"""Probe CRUD and time series extraction endpoints."""

import csv
import io

import numpy as np
from flask import Blueprint, Response, jsonify, request

from raft_dic_gui.velocity import calculate_displacement_magnitude, calculate_velocity_field
from server.session import session
from server.validation import validate_choice

probes_bp = Blueprint("probes", __name__)


def _probe_to_dict(p):
    """Serialize a Probe object."""
    coords = p.coords
    # Convert numpy types to Python types
    if isinstance(coords, tuple):
        coords = [float(c) for c in coords]
    elif isinstance(coords, list):
        coords = [[float(c) for c in pt] if isinstance(pt, (list, tuple)) else pt for pt in coords]
    return {
        "id": p.id,
        "type": p.type,
        "coords": coords,
        "color": p.color,
        "label": p.label,
    }


def _build_data_list(component: str) -> list:
    """Build a list of 2D arrays for time series extraction.

    Handles displacement (u, v, magnitude, velocity) and strain components.
    """
    data_list = []

    if component in ("u", "v", "magnitude", "velocity"):
        for i, disp in enumerate(session.displacement_results):
            if component == "u":
                data_list.append(disp[:, :, 0])
            elif component == "v":
                data_list.append(disp[:, :, 1])
            elif component == "magnitude":
                data_list.append(calculate_displacement_magnitude(disp[:, :, 0], disp[:, :, 1]))
            elif component == "velocity":
                if i == 0:
                    data_list.append(np.zeros(disp.shape[:2]))
                else:
                    prev = session.displacement_results[i - 1]
                    data_list.append(calculate_velocity_field(
                        disp[:, :, 0], disp[:, :, 1],
                        prev[:, :, 0], prev[:, :, 1],
                    ))
    else:
        # Strain component
        for s in session.strain_results:
            if s and component in s:
                data_list.append(s[component])
            else:
                data_list.append(None)

    return data_list


def _compute_scale_offset():
    """Compute scale factors and offset for probe coordinate mapping."""
    scale_factors = (1.0, 1.0)
    offset = (0, 0)

    if session.roi_rect:
        xmin, ymin, xmax, ymax = session.roi_rect
        roi_w = xmax - xmin
        roi_h = ymax - ymin
        offset = (xmin, ymin)

        # If strain is downsampled, calculate scale
        if session.strain_results and session.strain_results[0]:
            sample = next(iter(session.strain_results[0].values()))
            if sample is not None:
                data_h, data_w = sample.shape
                sy = data_h / roi_h if roi_h > 0 else 1.0
                sx = data_w / roi_w if roi_w > 0 else 1.0
                scale_factors = (sy, sx)

    return scale_factors, offset


# --- CRUD Endpoints ---

@probes_bp.route("", methods=["GET"])
def list_probes():
    """List all probes."""
    probes = [_probe_to_dict(p) for p in session.probe_manager.probes]
    return jsonify({"probes": probes})


@probes_bp.route("/point", methods=["POST"])
def add_point():
    """Add a point probe."""
    data = request.get_json(force=True)
    x = float(data.get("x", 0))
    y = float(data.get("y", 0))

    probe = session.probe_manager.add_point(x, y)
    return jsonify(_probe_to_dict(probe))


@probes_bp.route("/line", methods=["POST"])
def add_line():
    """Add a line probe."""
    data = request.get_json(force=True)
    p1 = data.get("p1", [0, 0])
    p2 = data.get("p2", [0, 0])

    probe = session.probe_manager.add_line(tuple(p1), tuple(p2))
    return jsonify(_probe_to_dict(probe))


@probes_bp.route("/area", methods=["POST"])
def add_area():
    """Add an area probe."""
    data = request.get_json(force=True)
    try:
        shape_type = validate_choice(data.get("shape", "rect"), "shape", ("rect", "circle", "polygon"))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    coords = data.get("data", [])

    probe = session.probe_manager.add_area(shape_type, coords)
    return jsonify(_probe_to_dict(probe))


@probes_bp.route("/<int:probe_id>", methods=["DELETE"])
def remove_probe(probe_id: int):
    """Remove a specific probe by ID and type."""
    probe_type = request.args.get("type", "point")
    session.probe_manager.remove_probe_by_id_type(probe_id, probe_type)
    return jsonify({"ok": True})


@probes_bp.route("", methods=["DELETE"])
def clear_all():
    """Clear all probes."""
    probe_type = request.args.get("type")
    if probe_type:
        session.probe_manager.clear_by_type(probe_type)
    else:
        session.probe_manager.clear_all()
    return jsonify({"ok": True})


# --- Extraction Endpoints ---

@probes_bp.route("/timeseries", methods=["GET"])
def get_timeseries():
    """Extract time series for all point/line/area probes."""
    component = request.args.get("component", "u")
    metric = request.args.get("metric", "avg")
    probe_type = request.args.get("type", "point")

    data_list = _build_data_list(component)
    if not data_list:
        return jsonify({"error": "No data available for this component"}), 404

    # For displacement components, scale is 1:1 with ROI
    # For strain components with step > 1, we need proper scale_factors
    if component in ("u", "v", "magnitude", "velocity"):
        offset = (session.roi_rect[0], session.roi_rect[1]) if session.roi_rect else (0, 0)
        scale_factors = (1.0, 1.0)
    else:
        scale_factors, offset = _compute_scale_offset()

    series = {}

    if probe_type == "point":
        results = session.probe_manager.extract_time_series(
            data_list, scale_factors=scale_factors, offset=offset
        )
        # Convert numpy values to plain floats
        for pid, vals in results.items():
            series[str(pid)] = [float(v) if np.isfinite(v) else None for v in vals]

    elif probe_type == "line":
        for p in session.probe_manager.probes:
            if p.type == "line":
                vals = session.probe_manager.extract_line_series(
                    data_list, p.id, metric=metric,
                    scale_factors=scale_factors, offset=offset,
                )
                if vals is not None:
                    series[str(p.id)] = [
                        float(v) if np.isfinite(v) else None for v in vals
                    ]

    elif probe_type == "area":
        for p in session.probe_manager.probes:
            if p.type == "area":
                vals = session.probe_manager.extract_area_series(
                    data_list, p.id, metric=metric,
                    scale_factors=scale_factors, offset=offset,
                )
                if vals is not None:
                    series[str(p.id)] = [
                        float(v) if np.isfinite(v) else None for v in vals
                    ]

    return jsonify({"series": series})


@probes_bp.route("/extensometer/<int:line_id>", methods=["GET"])
def get_extensometer(line_id: int):
    """Compute virtual extensometer data for a line probe.

    Returns deformed length and strain for each frame.
    """
    if not session.displacement_results:
        return jsonify({"error": "No displacement results"}), 404

    # Find the line probe
    line_probe = None
    for p in session.probe_manager.probes:
        if p.type == "line" and p.id == line_id:
            line_probe = p
            break

    if line_probe is None:
        return jsonify({"error": f"Line probe {line_id} not found"}), 404

    # Extract endpoint coordinates (image pixels)
    p1 = np.array(line_probe.coords[0], dtype=float)
    p2 = np.array(line_probe.coords[1], dtype=float)

    # Initial (reference) length
    initial_length = float(np.linalg.norm(p2 - p1))

    if session.roi_rect is None:
        return jsonify({"error": "No ROI defined"}), 400

    x0, y0, _, _ = session.roi_rect

    # For each frame, compute deformed endpoint positions and distance
    lengths = []
    strains = []
    for disp in session.displacement_results:
        # Get displacement at p1 and p2 (ROI-relative coordinates)
        r1 = int(round(p1[1] - y0))
        c1 = int(round(p1[0] - x0))
        r2 = int(round(p2[1] - y0))
        c2 = int(round(p2[0] - x0))

        h, w = disp.shape[:2]
        r1 = max(0, min(r1, h - 1))
        c1 = max(0, min(c1, w - 1))
        r2 = max(0, min(r2, h - 1))
        c2 = max(0, min(c2, w - 1))

        # Deformed positions = original + displacement
        d1 = p1 + np.array([disp[r1, c1, 0], disp[r1, c1, 1]])
        d2 = p2 + np.array([disp[r2, c2, 0], disp[r2, c2, 1]])

        deformed_length = float(np.linalg.norm(d2 - d1))
        lengths.append(deformed_length)
        strains.append((deformed_length - initial_length) / initial_length if initial_length > 0 else 0)

    return jsonify({
        "initial_length": initial_length,
        "lengths": lengths,
        "strains": strains,
        "num_frames": len(lengths),
    })


@probes_bp.route("/kymograph/<int:line_id>", methods=["GET"])
def get_kymograph(line_id: int):
    """Get kymograph data for a line probe."""
    component = request.args.get("component", "u")

    data_list = _build_data_list(component)
    if not data_list:
        return jsonify({"error": "No data available"}), 404

    if component in ("u", "v", "magnitude", "velocity"):
        offset = (session.roi_rect[0], session.roi_rect[1]) if session.roi_rect else (0, 0)
        scale_factors = (1.0, 1.0)
    else:
        scale_factors, offset = _compute_scale_offset()

    kymograph, dist_axis = session.probe_manager.extract_kymograph(
        data_list, line_id, scale_factors=scale_factors, offset=offset,
    )

    if kymograph is None:
        return jsonify({"error": "Line probe not found"}), 404

    return jsonify({
        "data": kymograph.tolist(),
        "dist_axis": dist_axis.tolist(),
        "shape": list(kymograph.shape),
    })


@probes_bp.route("/export/csv", methods=["GET"])
def export_csv():
    """Export probe time series as a downloadable CSV file."""
    component = request.args.get("component", "u")
    probe_type = request.args.get("type", "point")
    metric = request.args.get("metric", "avg")
    include_extensometer = request.args.get("extensometer", "false").lower() == "true"

    data_list = _build_data_list(component)
    if not data_list:
        return jsonify({"error": "No data available for this component"}), 404

    # Determine scale and offset (same logic as get_timeseries)
    if component in ("u", "v", "magnitude", "velocity"):
        offset = (session.roi_rect[0], session.roi_rect[1]) if session.roi_rect else (0, 0)
        scale_factors = (1.0, 1.0)
    else:
        scale_factors, offset = _compute_scale_offset()

    # Extract series keyed by probe id
    series: dict[str, list] = {}
    prefix = {"point": "probe", "line": "line", "area": "area"}.get(probe_type, "probe")

    if probe_type == "point":
        results = session.probe_manager.extract_time_series(
            data_list, scale_factors=scale_factors, offset=offset
        )
        for pid, vals in results.items():
            series[str(pid)] = [float(v) if np.isfinite(v) else None for v in vals]

    elif probe_type == "line":
        for p in session.probe_manager.probes:
            if p.type == "line":
                vals = session.probe_manager.extract_line_series(
                    data_list, p.id, metric=metric,
                    scale_factors=scale_factors, offset=offset,
                )
                if vals is not None:
                    series[str(p.id)] = [
                        float(v) if np.isfinite(v) else None for v in vals
                    ]

    elif probe_type == "area":
        for p in session.probe_manager.probes:
            if p.type == "area":
                vals = session.probe_manager.extract_area_series(
                    data_list, p.id, metric=metric,
                    scale_factors=scale_factors, offset=offset,
                )
                if vals is not None:
                    series[str(p.id)] = [
                        float(v) if np.isfinite(v) else None for v in vals
                    ]

    if not series:
        return jsonify({"error": "No probe data to export"}), 404

    # Compute extensometer data for line probes if requested
    extensometer_data: dict[str, dict] = {}
    if include_extensometer and probe_type == "line" and session.displacement_results:
        for p in session.probe_manager.probes:
            if p.type == "line" and str(p.id) in series:
                p1 = np.array(p.coords[0], dtype=float)
                p2 = np.array(p.coords[1], dtype=float)
                initial_length = float(np.linalg.norm(p2 - p1))

                if session.roi_rect is None:
                    continue

                x0, y0, _, _ = session.roi_rect
                lengths = []
                strains = []
                for disp in session.displacement_results:
                    r1 = int(round(p1[1] - y0))
                    c1 = int(round(p1[0] - x0))
                    r2 = int(round(p2[1] - y0))
                    c2 = int(round(p2[0] - x0))

                    h, w = disp.shape[:2]
                    r1 = max(0, min(r1, h - 1))
                    c1 = max(0, min(c1, w - 1))
                    r2 = max(0, min(r2, h - 1))
                    c2 = max(0, min(c2, w - 1))

                    d1 = p1 + np.array([disp[r1, c1, 0], disp[r1, c1, 1]])
                    d2 = p2 + np.array([disp[r2, c2, 0], disp[r2, c2, 1]])

                    deformed_length = float(np.linalg.norm(d2 - d1))
                    lengths.append(deformed_length)
                    strains.append(
                        (deformed_length - initial_length) / initial_length
                        if initial_length > 0 else 0.0
                    )

                extensometer_data[str(p.id)] = {
                    "lengths": lengths,
                    "strains": strains,
                }

    # Build CSV
    buf = io.StringIO()
    writer = csv.writer(buf)

    # Determine number of frames from the first series
    probe_ids = sorted(series.keys(), key=lambda x: int(x))
    num_frames = len(next(iter(series.values())))

    # Header row
    header = ["frame"]
    for pid in probe_ids:
        header.append(f"{prefix}_{pid}_{component}")
    for pid in probe_ids:
        if pid in extensometer_data:
            header.append(f"{prefix}_{pid}_strain")
            header.append(f"{prefix}_{pid}_deformed_length")
    writer.writerow(header)

    # Data rows
    for i in range(num_frames):
        row: list = [i + 1]
        for pid in probe_ids:
            val = series[pid][i] if i < len(series[pid]) else None
            row.append(val if val is not None else "")
        for pid in probe_ids:
            if pid in extensometer_data:
                ext = extensometer_data[pid]
                if i < len(ext["strains"]):
                    row.append(ext["strains"][i])
                    row.append(ext["lengths"][i])
                else:
                    row.append("")
                    row.append("")
        writer.writerow(row)

    csv_content = buf.getvalue()
    filename = f"probes_{probe_type}_{component}.csv"

    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
