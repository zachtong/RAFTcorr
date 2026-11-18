"""
Report generator for RAFT-DIC GUI.

Generates self-contained HTML analysis reports with inline CSS and base64 images.
Optionally exports PDF via weasyprint (graceful fallback if not installed).
"""

import base64
import io
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jinja2 import Environment, FileSystemLoader, select_autoescape

from raft_dic_gui.export_images import _get_component_data

logger = logging.getLogger(__name__)

# Optional PDF support
try:
    import weasyprint
    HAS_WEASYPRINT = True
except ImportError:
    HAS_WEASYPRINT = False

# All available sections in order
ALL_SECTIONS = [
    "header", "experiment", "parameters", "roi",
    "key_frames", "statistics", "probes",
    "line_probes", "area_probes",
]

# ---------------------------------------------------------------------------
# Theme system
# ---------------------------------------------------------------------------

THEMES = {
    "light": """
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #1a1a2e; background: #fff;
        }
        h1 { color: #16213e; }
        h2 { color: #0f3460; border-bottom: 2px solid #e0e0e0; }
        h3 { color: #333; }
        .meta { color: #666; }
        th { background: #f4f6f9; color: #333; }
        td { color: #444; }
        tr:hover td { background: #fafbfc; }
        th, td { border-bottom: 1px solid #e8e8e8; }
        .frame-card { border: 1px solid #e0e0e0; background: #fafafa; }
        .frame-card .label { color: #555; background: #f4f6f9; }
        .probe-card { border: 1px solid #e0e0e0; }
        .probe-card .probe-header { background: #f4f6f9; color: #333; }
        .probe-card .probe-header span { color: #888; }
        .footer { border-top: 1px solid #e0e0e0; color: #999; }
        .notes { color: #666; }
        .author { color: #555; }
    """,
    "dark": """
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #e0e0e0; background: #1a1a2e;
        }
        h1 { color: #7ec8e3; }
        h2 { color: #7ec8e3; border-bottom: 2px solid #333355; }
        h3 { color: #c0c0c0; }
        .meta { color: #999; }
        th { background: #252545; color: #e0e0e0; }
        td { color: #c0c0c0; }
        tr:nth-child(even) td { background: #252545; }
        tr:hover td { background: #303060; }
        th, td { border-bottom: 1px solid #333355; }
        .frame-card { border: 1px solid #333355; background: #252545; }
        .frame-card .label { color: #c0c0c0; background: #1e1e3e; }
        .probe-card { border: 1px solid #333355; }
        .probe-card .probe-header { background: #252545; color: #e0e0e0; }
        .probe-card .probe-header span { color: #999; }
        .footer { border-top: 1px solid #333355; color: #666; }
        .notes { color: #999; }
        .author { color: #aaa; }
        .no-data { color: #888; }
    """,
    "academic": """
        body {
            font-family: Georgia, "Times New Roman", Times, serif;
            color: #000; background: #fff;
        }
        h1 { color: #000; font-weight: bold; }
        h2 {
            color: #000; font-weight: bold;
            border-bottom: 1px solid #000;
            counter-increment: section;
        }
        h2::before { content: counter(section) ". "; }
        h3 { color: #222; }
        .meta { color: #333; }
        table { border: 1px solid #000; }
        th { background: #f0f0f0; color: #000; border: 1px solid #000; }
        td { color: #000; border: 1px solid #000; }
        tr:hover td { background: #f8f8f8; }
        .frame-card { border: 1px solid #000; background: #fff; }
        .frame-card .label { color: #000; background: #f0f0f0; }
        .probe-card { border: 1px solid #000; }
        .probe-card .probe-header { background: #f0f0f0; color: #000; }
        .probe-card .probe-header span { color: #555; }
        .footer { border-top: 1px solid #000; color: #555; }
        .notes { color: #333; }
        .author { color: #222; }
    """,
    "minimal": """
        body {
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #333; background: #fff;
        }
        h1 { color: #333; font-weight: 600; }
        h2 { color: #333; font-weight: 600; border-bottom: none; }
        h3 { color: #555; }
        .meta { color: #888; }
        th { background: transparent; color: #333; font-weight: 600; }
        td { color: #555; }
        tr:hover td { background: transparent; }
        th, td { border-bottom: none; padding: 6px 16px 6px 0; }
        table { border-collapse: separate; border-spacing: 0 4px; }
        .frame-card { border: none; background: #fafafa; border-radius: 4px; }
        .frame-card .label { color: #555; background: transparent; }
        .probe-card { border: none; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
        .probe-card .probe-header { background: transparent; color: #333; }
        .probe-card .probe-header span { color: #888; }
        .footer { border-top: none; color: #bbb; }
        .notes { color: #888; }
        .author { color: #888; }
    """,
}


def generate_report(
    session,
    output_path: str,
    sections: Optional[List[str]] = None,
    format: str = "html",
    theme: str = "light",
    custom_title: str = "",
    author: str = "",
    notes: str = "",
    key_frame_components: Optional[List[str]] = None,
    vis_settings: Optional[Dict] = None,
) -> Dict[str, Optional[str]]:
    """
    Generate an HTML (and optionally PDF) analysis report.

    Args:
        session: AppSession object
        output_path: Path to write the HTML file
        sections: List of section names to include (None = all)
        format: "html", "pdf", or "both"
        theme: One of "light", "dark", "academic", "minimal"
        custom_title: Custom report title (empty = default)
        author: Optional author/institution name
        notes: Optional multi-line notes
        key_frame_components: Components for key frames (None = ["magnitude"])
        vis_settings: Visualization settings dict (None = use session.vis_settings)

    Returns:
        Dict with "html_path" and/or "pdf_path" keys.
    """
    abs_path = os.path.abspath(output_path)
    active_sections = sections or ALL_SECTIONS

    # Resolve vis_settings
    vs = vis_settings or getattr(session, "vis_settings", {})

    # Determine physical unit state
    physical_ratio = vs.get("physical_ratio", 1.0)
    physical_unit = vs.get("physical_unit", "px")
    physical_enabled = physical_ratio != 1.0 and physical_unit != "px"

    context = {
        "sections": active_sections,
        "theme_css": THEMES.get(theme, THEMES["light"]),
    }

    if "header" in active_sections:
        context["header"] = _render_header_section(
            session,
            custom_title=custom_title,
            author=author,
            notes=notes,
        )
    if "experiment" in active_sections:
        context["experiment"] = _render_experiment_section(session)
    if "parameters" in active_sections:
        context["parameters"] = _render_parameters_section(session)
    if "roi" in active_sections:
        context["roi"] = _render_roi_section(session)
    if "key_frames" in active_sections:
        context["key_frames"] = _render_key_frames_section(
            session,
            components=key_frame_components,
            vis_settings=vs,
            physical_enabled=physical_enabled,
            physical_unit=physical_unit,
            physical_ratio=physical_ratio,
        )
    if "statistics" in active_sections:
        context["statistics"] = _render_statistics_section(
            session,
            physical_enabled=physical_enabled,
            physical_unit=physical_unit,
            physical_ratio=physical_ratio,
        )
    if "probes" in active_sections:
        context["probes"] = _render_probes_section(
            session,
            physical_enabled=physical_enabled,
            physical_unit=physical_unit,
            physical_ratio=physical_ratio,
        )
    if "line_probes" in active_sections:
        context["line_probes"] = _render_line_probes_section(
            session,
            vis_settings=vs,
            physical_enabled=physical_enabled,
            physical_unit=physical_unit,
            physical_ratio=physical_ratio,
        )
    if "area_probes" in active_sections:
        context["area_probes"] = _render_area_probes_section(
            session,
            vis_settings=vs,
            physical_enabled=physical_enabled,
            physical_unit=physical_unit,
            physical_ratio=physical_ratio,
        )

    html = _get_template().render(**context)

    result = {"html_path": None, "pdf_path": None}

    # Write HTML
    if format in ("html", "both"):
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(html)
        result["html_path"] = abs_path
        logger.info("Report generated (HTML): %s", abs_path)

    # Write PDF
    if format in ("pdf", "both"):
        pdf_path = os.path.splitext(abs_path)[0] + ".pdf"
        if HAS_WEASYPRINT:
            try:
                weasyprint.HTML(string=html).write_pdf(pdf_path)
                result["pdf_path"] = pdf_path
                logger.info("Report generated (PDF): %s", pdf_path)
            except Exception as e:
                logger.error("PDF generation failed: %s", e)
                # Fallback: write HTML anyway if only PDF was requested
                if format == "pdf":
                    with open(abs_path, "w", encoding="utf-8") as f:
                        f.write(html)
                    result["html_path"] = abs_path
                    logger.info("Fallback: wrote HTML instead of PDF: %s", abs_path)
        else:
            logger.warning("weasyprint not installed — cannot generate PDF, falling back to HTML")
            if format == "pdf":
                with open(abs_path, "w", encoding="utf-8") as f:
                    f.write(html)
                result["html_path"] = abs_path

    return result


# ---------------------------------------------------------------------------
# Helper: physical unit label suffix
# ---------------------------------------------------------------------------

def _unit_label(component: str, physical_enabled: bool, physical_unit: str) -> str:
    """Return a display label for a component, with physical unit if enabled."""
    disp_comps = {"u", "v", "magnitude"}
    if component.lower() in disp_comps:
        if physical_enabled:
            return f"{component} ({physical_unit})"
        return f"{component} (px)"
    # Strain components are dimensionless
    return component


def _is_displacement_component(comp: str) -> bool:
    return comp.lower() in {"u", "v", "magnitude"}


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_header_section(
    session,
    custom_title: str = "",
    author: str = "",
    notes: str = "",
) -> dict:
    return {
        "title": custom_title or "RAFT-DIC Analysis Report",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "software": "RAFT-DIC GUI",
        "model": session.config.model_label or os.path.basename(session.config.model_path),
        "author": author,
        "notes": notes,
    }


def _render_experiment_section(session) -> dict:
    return {
        "image_dir": session.image_dir,
        "num_images": len(session.image_files),
        "image_width": session.image_width,
        "image_height": session.image_height,
        "mode": session.config.mode,
        "num_displacement_frames": len(session.displacement_results),
        "num_strain_frames": len(session.strain_results),
    }


def _render_parameters_section(session) -> dict:
    config = session.config
    params = [
        ("Processing Mode", config.mode),
        ("Model", config.model_label or os.path.basename(config.model_path)),
        ("Smoothing", f"{'Enabled' if config.use_smooth else 'Disabled'} (sigma={config.sigma})"),
        ("Context Padding", f"{config.context_padding} px"),
        ("Tile Overlap", f"{config.tile_overlap} px"),
        ("Max Pixels", f"{config.p_max_pixels:,}"),
        ("Device", config.device),
    ]
    if config.mode == "incremental":
        params.append(("Key Frame Interval", str(config.key_frame_interval or "N/A")))
        params.append(("Median Filter", "Yes" if config.use_median_filter else "No"))
    return {"params": params}


def _render_roi_section(session) -> dict:
    result = {
        "has_roi": session.roi_rect is not None,
        "roi_rect": session.roi_rect,
        "image": None,
    }
    if session.roi_rect and session.reference_image is not None:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        rgb = cv2.cvtColor(session.reference_image, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        xmin, ymin, xmax, ymax = session.roi_rect
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                              linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        ax.set_title("Region of Interest")
        ax.axis("off")
        plt.tight_layout()
        result["image"] = _fig_to_base64(fig)
    return result


def _render_key_frames_section(
    session,
    components: Optional[List[str]] = None,
    vis_settings: Optional[Dict] = None,
    physical_enabled: bool = False,
    physical_unit: str = "px",
    physical_ratio: float = 1.0,
) -> dict:
    """Render first / peak / last frame snapshots for each requested component."""
    components = components or ["magnitude"]
    vs = vis_settings or {}
    colormap_name = vs.get("colormap", "turbo")

    rows = []
    n = len(session.displacement_results)
    if n == 0:
        return {"rows": rows}

    for comp in components:
        # Find per-component peak frame
        peak_idx = 0
        peak_val = 0
        for i in range(n):
            data = _get_component_data(comp, i, session.displacement_results, session.strain_results)
            if data is not None:
                fmax = np.nanmax(np.abs(data))
                if fmax > peak_val:
                    peak_val = fmax
                    peak_idx = i

        indices = [0, peak_idx, n - 1]
        labels = ["First Frame", f"Peak Frame (#{peak_idx + 1})", "Last Frame"]

        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for idx, lbl in zip(indices, labels):
            if idx not in seen:
                seen.add(idx)
                unique.append((idx, lbl))

        frames = []
        for idx, label in unique:
            data = _get_component_data(comp, idx, session.displacement_results, session.strain_results)
            if data is None:
                continue

            # Apply physical unit scaling for displacement components
            display_data = data.copy()
            if physical_enabled and _is_displacement_component(comp):
                display_data = display_data * physical_ratio

            fig, ax = plt.subplots(figsize=(5, 3.5), dpi=100)
            masked = np.ma.array(display_data, mask=np.isnan(display_data))
            cmap = plt.get_cmap(colormap_name).copy()
            cmap.set_bad(alpha=0)

            im = ax.imshow(masked, cmap=cmap)

            cbar_label = _unit_label(comp, physical_enabled, physical_unit)
            fig.colorbar(im, ax=ax, shrink=0.8, label=cbar_label)
            ax.set_title(f"{label} — {comp}", fontsize=10)
            ax.axis("off")
            plt.tight_layout()
            frames.append({
                "label": label,
                "frame_idx": idx + 1,
                "image": _fig_to_base64(fig),
            })

        rows.append({
            "component": comp,
            "frames": frames,
        })

    return {"rows": rows}


def _render_statistics_section(
    session,
    physical_enabled: bool = False,
    physical_unit: str = "px",
    physical_ratio: float = 1.0,
) -> dict:
    """Compute min/max/mean/std for displacement and strain components, plus trend charts."""
    stats = []
    trend_charts = []

    disp_components = ["u", "v", "magnitude"]
    disp_labels = ["U (horizontal)", "V (vertical)", "Magnitude"]

    n_disp = len(session.displacement_results)
    n_strain = len(session.strain_results)

    # --- Displacement stats + trends ---
    for comp, label in zip(disp_components, disp_labels):
        all_vals = []
        per_frame_min = []
        per_frame_max = []
        per_frame_mean = []
        per_frame_std = []

        for i in range(n_disp):
            data = _get_component_data(comp, i, session.displacement_results, session.strain_results)
            if data is not None:
                if physical_enabled and _is_displacement_component(comp):
                    data = data * physical_ratio
                valid = data[~np.isnan(data)]
                if valid.size > 0:
                    all_vals.append(valid)
                    per_frame_min.append(float(np.min(valid)))
                    per_frame_max.append(float(np.max(valid)))
                    per_frame_mean.append(float(np.mean(valid)))
                    per_frame_std.append(float(np.std(valid)))
                else:
                    per_frame_min.append(np.nan)
                    per_frame_max.append(np.nan)
                    per_frame_mean.append(np.nan)
                    per_frame_std.append(np.nan)
            else:
                per_frame_min.append(np.nan)
                per_frame_max.append(np.nan)
                per_frame_mean.append(np.nan)
                per_frame_std.append(np.nan)

        unit_suffix = f" ({physical_unit})" if physical_enabled else " (px)"
        display_label = label + unit_suffix

        if all_vals:
            combined = np.concatenate(all_vals)
            stats.append({
                "component": display_label,
                "min": f"{np.min(combined):.4f}",
                "max": f"{np.max(combined):.4f}",
                "mean": f"{np.mean(combined):.4f}",
                "std": f"{np.std(combined):.4f}",
            })

        # Trend chart
        if per_frame_mean and not all(np.isnan(v) for v in per_frame_mean):
            chart_img = _render_trend_chart(
                per_frame_min, per_frame_max, per_frame_mean, per_frame_std,
                ylabel=display_label,
                title=f"{label} Trend",
            )
            trend_charts.append({
                "component": label,
                "image": chart_img,
            })

    # --- Strain stats + trends (ALL components, no [:5] limit) ---
    for comp_name in session.strain_components:
        all_vals = []
        per_frame_min = []
        per_frame_max = []
        per_frame_mean = []
        per_frame_std = []

        for i in range(n_strain):
            data = _get_component_data(comp_name, i, session.displacement_results, session.strain_results)
            if data is not None:
                valid = data[~np.isnan(data)]
                if valid.size > 0:
                    all_vals.append(valid)
                    per_frame_min.append(float(np.min(valid)))
                    per_frame_max.append(float(np.max(valid)))
                    per_frame_mean.append(float(np.mean(valid)))
                    per_frame_std.append(float(np.std(valid)))
                else:
                    per_frame_min.append(np.nan)
                    per_frame_max.append(np.nan)
                    per_frame_mean.append(np.nan)
                    per_frame_std.append(np.nan)
            else:
                per_frame_min.append(np.nan)
                per_frame_max.append(np.nan)
                per_frame_mean.append(np.nan)
                per_frame_std.append(np.nan)

        if all_vals:
            combined = np.concatenate(all_vals)
            stats.append({
                "component": comp_name,
                "min": f"{np.min(combined):.6f}",
                "max": f"{np.max(combined):.6f}",
                "mean": f"{np.mean(combined):.6f}",
                "std": f"{np.std(combined):.6f}",
            })

        # Trend chart for strain
        if per_frame_mean and not all(np.isnan(v) for v in per_frame_mean):
            chart_img = _render_trend_chart(
                per_frame_min, per_frame_max, per_frame_mean, per_frame_std,
                ylabel=comp_name,
                title=f"{comp_name} Trend",
            )
            trend_charts.append({
                "component": comp_name,
                "image": chart_img,
            })

    return {"stats": stats, "trend_charts": trend_charts}


def _render_trend_chart(
    frame_min: List[float],
    frame_max: List[float],
    frame_mean: List[float],
    frame_std: List[float],
    ylabel: str = "",
    title: str = "",
) -> str:
    """Render a trend chart with min/max/mean lines and mean +/- std band."""
    n = len(frame_mean)
    frames_axis = list(range(1, n + 1))
    mean_arr = np.array(frame_mean, dtype=float)
    std_arr = np.array(frame_std, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 3), dpi=100)
    ax.plot(frames_axis, frame_min, color="blue", linestyle="--", linewidth=1, label="Min")
    ax.plot(frames_axis, frame_max, color="red", linestyle="--", linewidth=1, label="Max")
    ax.plot(frames_axis, frame_mean, color="black", linestyle="-", linewidth=1.5, label="Mean")
    ax.fill_between(
        frames_axis,
        mean_arr - std_arr,
        mean_arr + std_arr,
        color="gray",
        alpha=0.25,
        label="Mean \u00b1 Std",
    )
    ax.set_xlabel("Frame", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _fig_to_base64(fig)


def _render_probes_section(
    session,
    physical_enabled: bool = False,
    physical_unit: str = "px",
    physical_ratio: float = 1.0,
) -> dict:
    """Render time series charts for each point probe."""
    probes = session.probe_manager.probes
    point_probes = [p for p in probes if p.type == "point"]
    if not point_probes:
        return {"probe_charts": []}

    charts = []
    n = len(session.displacement_results)
    if n == 0:
        return {"probe_charts": []}

    # Build u and v data lists
    u_list = []
    v_list = []
    for i in range(n):
        u_data = _get_component_data("u", i, session.displacement_results, session.strain_results)
        v_data = _get_component_data("v", i, session.displacement_results, session.strain_results)
        u_list.append(u_data)
        v_list.append(v_data)

    from scipy.ndimage import map_coordinates
    roi_rect = session.roi_rect
    offset = (roi_rect[0], roi_rect[1]) if roi_rect else (0, 0)

    u_ylabel = f"U displacement ({physical_unit})" if physical_enabled else "U displacement (px)"
    v_ylabel = f"V displacement ({physical_unit})" if physical_enabled else "V displacement (px)"

    for p in point_probes:
        px = p.coords[0] - offset[0]
        py = p.coords[1] - offset[1]
        coords = np.array([[py], [px]])  # (2, 1) for map_coordinates

        u_series = []
        v_series = []
        for i in range(n):
            if u_list[i] is not None:
                u_val = map_coordinates(u_list[i], coords, order=1, mode="constant", cval=np.nan)[0]
                val = float(u_val)
                if physical_enabled:
                    val = val * physical_ratio
                u_series.append(val)
            else:
                u_series.append(np.nan)
            if v_list[i] is not None:
                v_val = map_coordinates(v_list[i], coords, order=1, mode="constant", cval=np.nan)[0]
                val = float(v_val)
                if physical_enabled:
                    val = val * physical_ratio
                v_series.append(val)
            else:
                v_series.append(np.nan)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), dpi=100)
        frames_axis = list(range(1, n + 1))
        ax1.plot(frames_axis, u_series, color=p.color, linewidth=1.5)
        ax1.set_xlabel("Frame", fontsize=9)
        ax1.set_ylabel(u_ylabel, fontsize=9)
        ax1.set_title(f"{p.label} — U", fontsize=10)
        ax1.grid(True, alpha=0.3)

        ax2.plot(frames_axis, v_series, color=p.color, linewidth=1.5)
        ax2.set_xlabel("Frame", fontsize=9)
        ax2.set_ylabel(v_ylabel, fontsize=9)
        ax2.set_title(f"{p.label} — V", fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        charts.append({
            "label": p.label,
            "type": p.type,
            "coords": f"({p.coords[0]:.0f}, {p.coords[1]:.0f})",
            "image": _fig_to_base64(fig),
        })

    return {"probe_charts": charts}


def _render_line_probes_section(
    session,
    vis_settings: Optional[Dict] = None,
    physical_enabled: bool = False,
    physical_unit: str = "px",
    physical_ratio: float = 1.0,
) -> dict:
    """Render summary time series and kymograph for each line probe."""
    vs = vis_settings or {}
    colormap_name = vs.get("colormap", "turbo")

    probes = session.probe_manager.probes
    line_probes = [p for p in probes if p.type == "line"]
    if not line_probes:
        return {"line_charts": []}

    n = len(session.displacement_results)
    if n == 0:
        return {"line_charts": []}

    # Build u and v data lists
    u_list = []
    v_list = []
    for i in range(n):
        u_list.append(_get_component_data("u", i, session.displacement_results, session.strain_results))
        v_list.append(_get_component_data("v", i, session.displacement_results, session.strain_results))

    roi_rect = session.roi_rect
    offset = (roi_rect[0], roi_rect[1]) if roi_rect else (0, 0)

    u_ylabel = f"U avg ({physical_unit})" if physical_enabled else "U avg (px)"
    v_ylabel = f"V avg ({physical_unit})" if physical_enabled else "V avg (px)"

    charts = []
    for p in line_probes:
        entry = {
            "label": p.label,
            "coords": f"({p.coords[0][0]:.0f},{p.coords[0][1]:.0f}) to ({p.coords[1][0]:.0f},{p.coords[1][1]:.0f})",
            "summary_image": None,
            "kymo_image": None,
        }

        # --- a) Summary time series (avg along line) ---
        u_avg = session.probe_manager.extract_line_series(
            u_list, p.id, metric="avg", num_points=100,
            scale_factors=(1.0, 1.0), offset=offset,
        )
        v_avg = session.probe_manager.extract_line_series(
            v_list, p.id, metric="avg", num_points=100,
            scale_factors=(1.0, 1.0), offset=offset,
        )

        if u_avg is not None and v_avg is not None:
            if physical_enabled:
                u_avg = u_avg * physical_ratio
                v_avg = v_avg * physical_ratio

            frames_axis = list(range(1, n + 1))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), dpi=100)
            ax1.plot(frames_axis, u_avg, color=p.color, linewidth=1.5)
            ax1.set_xlabel("Frame", fontsize=9)
            ax1.set_ylabel(u_ylabel, fontsize=9)
            ax1.set_title(f"{p.label} — U avg", fontsize=10)
            ax1.grid(True, alpha=0.3)

            ax2.plot(frames_axis, v_avg, color=p.color, linewidth=1.5)
            ax2.set_xlabel("Frame", fontsize=9)
            ax2.set_ylabel(v_ylabel, fontsize=9)
            ax2.set_title(f"{p.label} — V avg", fontsize=10)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            entry["summary_image"] = _fig_to_base64(fig)

        # --- b) Kymographs ---
        u_kymo, u_dist = session.probe_manager.extract_kymograph(
            u_list, p.id, num_points=100,
            scale_factors=(1.0, 1.0), offset=offset,
        )
        v_kymo, v_dist = session.probe_manager.extract_kymograph(
            v_list, p.id, num_points=100,
            scale_factors=(1.0, 1.0), offset=offset,
        )

        if u_kymo is not None and v_kymo is not None:
            if physical_enabled:
                u_kymo = u_kymo * physical_ratio
                v_kymo = v_kymo * physical_ratio

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), dpi=100)
            cmap = plt.get_cmap(colormap_name)

            im1 = ax1.imshow(u_kymo, aspect="auto", cmap=cmap,
                             extent=[1, n, u_dist[-1], u_dist[0]])
            ax1.set_xlabel("Frame", fontsize=9)
            ax1.set_ylabel("Position along line", fontsize=9)
            ax1.set_title(f"{p.label} — U kymograph", fontsize=10)
            fig.colorbar(im1, ax=ax1, shrink=0.8)

            im2 = ax2.imshow(v_kymo, aspect="auto", cmap=cmap,
                             extent=[1, n, v_dist[-1], v_dist[0]])
            ax2.set_xlabel("Frame", fontsize=9)
            ax2.set_ylabel("Position along line", fontsize=9)
            ax2.set_title(f"{p.label} — V kymograph", fontsize=10)
            fig.colorbar(im2, ax=ax2, shrink=0.8)

            plt.tight_layout()
            entry["kymo_image"] = _fig_to_base64(fig)

        charts.append(entry)

    return {"line_charts": charts}


def _render_area_probes_section(
    session,
    vis_settings: Optional[Dict] = None,
    physical_enabled: bool = False,
    physical_unit: str = "px",
    physical_ratio: float = 1.0,
) -> dict:
    """Render summary, stats table, and 2D snapshots for each area probe."""
    vs = vis_settings or {}
    colormap_name = vs.get("colormap", "turbo")

    probes = session.probe_manager.probes
    area_probes = [p for p in probes if p.type == "area"]
    if not area_probes:
        return {"area_charts": []}

    n = len(session.displacement_results)
    if n == 0:
        return {"area_charts": []}

    # Build u and v data lists
    u_list = []
    v_list = []
    for i in range(n):
        u_list.append(_get_component_data("u", i, session.displacement_results, session.strain_results))
        v_list.append(_get_component_data("v", i, session.displacement_results, session.strain_results))

    roi_rect = session.roi_rect
    offset = (roi_rect[0], roi_rect[1]) if roi_rect else (0, 0)

    u_ylabel = f"U ({physical_unit})" if physical_enabled else "U (px)"
    v_ylabel = f"V ({physical_unit})" if physical_enabled else "V (px)"

    charts = []
    for p in area_probes:
        shape_desc = p.coords.get("shape", "unknown") if isinstance(p.coords, dict) else "unknown"
        entry = {
            "label": p.label,
            "shape": shape_desc,
            "summary_image": None,
            "stats": [],
            "snapshots": [],
        }

        # --- a) Summary time series (avg + min/max envelope) ---
        u_avg = session.probe_manager.extract_area_series(
            u_list, p.id, metric="avg",
            scale_factors=(1.0, 1.0), offset=offset,
        )
        u_min = session.probe_manager.extract_area_series(
            u_list, p.id, metric="min",
            scale_factors=(1.0, 1.0), offset=offset,
        )
        u_max = session.probe_manager.extract_area_series(
            u_list, p.id, metric="max",
            scale_factors=(1.0, 1.0), offset=offset,
        )
        v_avg = session.probe_manager.extract_area_series(
            v_list, p.id, metric="avg",
            scale_factors=(1.0, 1.0), offset=offset,
        )
        v_min = session.probe_manager.extract_area_series(
            v_list, p.id, metric="min",
            scale_factors=(1.0, 1.0), offset=offset,
        )
        v_max = session.probe_manager.extract_area_series(
            v_list, p.id, metric="max",
            scale_factors=(1.0, 1.0), offset=offset,
        )

        if u_avg is not None and v_avg is not None:
            u_avg_arr = np.array(u_avg, dtype=float)
            u_min_arr = np.array(u_min, dtype=float)
            u_max_arr = np.array(u_max, dtype=float)
            v_avg_arr = np.array(v_avg, dtype=float)
            v_min_arr = np.array(v_min, dtype=float)
            v_max_arr = np.array(v_max, dtype=float)

            if physical_enabled:
                u_avg_arr = u_avg_arr * physical_ratio
                u_min_arr = u_min_arr * physical_ratio
                u_max_arr = u_max_arr * physical_ratio
                v_avg_arr = v_avg_arr * physical_ratio
                v_min_arr = v_min_arr * physical_ratio
                v_max_arr = v_max_arr * physical_ratio

            frames_axis = list(range(1, n + 1))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), dpi=100)

            ax1.plot(frames_axis, u_avg_arr, color=p.color, linewidth=1.5, label="Mean")
            ax1.fill_between(frames_axis, u_min_arr, u_max_arr, color=p.color, alpha=0.15, label="Min/Max")
            ax1.set_xlabel("Frame", fontsize=9)
            ax1.set_ylabel(u_ylabel, fontsize=9)
            ax1.set_title(f"{p.label} — U", fontsize=10)
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)

            ax2.plot(frames_axis, v_avg_arr, color=p.color, linewidth=1.5, label="Mean")
            ax2.fill_between(frames_axis, v_min_arr, v_max_arr, color=p.color, alpha=0.15, label="Min/Max")
            ax2.set_xlabel("Frame", fontsize=9)
            ax2.set_ylabel(v_ylabel, fontsize=9)
            ax2.set_title(f"{p.label} — V", fontsize=10)
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            entry["summary_image"] = _fig_to_base64(fig)

            # --- b) Statistics table ---
            for comp_label, avg_arr, min_arr, max_arr in [
                (u_ylabel, u_avg_arr, u_min_arr, u_max_arr),
                (v_ylabel, v_avg_arr, v_min_arr, v_max_arr),
            ]:
                valid = avg_arr[~np.isnan(avg_arr)]
                valid_min = min_arr[~np.isnan(min_arr)]
                valid_max = max_arr[~np.isnan(max_arr)]
                if valid.size > 0:
                    entry["stats"].append({
                        "component": comp_label,
                        "min": f"{np.min(valid_min):.4f}",
                        "max": f"{np.max(valid_max):.4f}",
                        "mean": f"{np.mean(valid):.4f}",
                        "std": f"{np.std(valid):.4f}",
                    })

        # --- c) 2D distribution snapshots ---
        selected_frames = _select_snapshot_frames(n, max_frames=6)
        # Build area mask once
        area_mask = _build_area_mask(p, u_list[0].shape if u_list[0] is not None else None, offset)

        for comp_name, comp_display in [("u", "U"), ("v", "V")]:
            data_list = u_list if comp_name == "u" else v_list
            frame_imgs = []
            for fi in selected_frames:
                if fi >= len(data_list) or data_list[fi] is None:
                    continue
                frame_data = data_list[fi].copy()
                if physical_enabled:
                    frame_data = frame_data * physical_ratio

                if area_mask is not None:
                    frame_data = np.where(area_mask, frame_data, np.nan)

                fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
                masked = np.ma.array(frame_data, mask=np.isnan(frame_data))
                cmap = plt.get_cmap(colormap_name)
                cmap.set_bad(alpha=0)
                im = ax.imshow(masked, cmap=cmap)
                fig.colorbar(im, ax=ax, shrink=0.8)
                ax.set_title(f"Frame #{fi + 1} — {comp_display}", fontsize=9)
                ax.axis("off")
                plt.tight_layout()
                frame_imgs.append({
                    "frame_idx": fi + 1,
                    "image": _fig_to_base64(fig),
                })

            if frame_imgs:
                entry["snapshots"].append({
                    "component": comp_display,
                    "frames": frame_imgs,
                })

        charts.append(entry)

    return {"area_charts": charts}


def _select_snapshot_frames(n: int, max_frames: int = 6) -> List[int]:
    """Select frames for 2D snapshots: equal-interval sampling, max max_frames."""
    if n <= max_frames:
        return list(range(n))
    # first, last, + (max_frames - 2) evenly spaced in between
    indices = set()
    indices.add(0)
    indices.add(n - 1)
    for i in range(1, max_frames - 1):
        idx = int(round(i * (n - 1) / (max_frames - 1)))
        indices.add(idx)
    return sorted(indices)


def _build_area_mask(probe, data_shape, offset):
    """Build a boolean mask for an area probe in data coordinates."""
    if data_shape is None:
        return None
    h, w = data_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    off_x, off_y = offset
    shape_type = probe.coords.get("shape", "")
    shape_data = probe.coords.get("data", [])

    def transform_point(x, y):
        return int(x - off_x), int(y - off_y)

    if shape_type == "rect":
        x0, y0, x1, y1 = shape_data
        tx0, ty0 = transform_point(x0, y0)
        tx1, ty1 = transform_point(x1, y1)
        tx0, tx1 = min(tx0, tx1), max(tx0, tx1)
        ty0, ty1 = min(ty0, ty1), max(ty0, ty1)
        cv2.rectangle(mask, (tx0, ty0), (tx1, ty1), 1, -1)
    elif shape_type == "circle":
        cx, cy, r = shape_data
        tcx, tcy = transform_point(cx, cy)
        tr = int(r)
        cv2.circle(mask, (tcx, tcy), tr, 1, -1)
    elif shape_type == "poly":
        pts = []
        for px, py in shape_data:
            pts.append(transform_point(px, py))
        pts = np.array([pts], dtype=np.int32)
        cv2.fillPoly(mask, pts, 1)

    return mask.astype(bool)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

def _get_template():
    template_dir = os.path.dirname(__file__)
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html"]),
    )
    return env.get_template("report_template.html")
