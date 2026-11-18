"""
Session save/load for RAFT-DIC GUI.

File format: .raftproj = ZIP archive containing:
  - metadata.json   (config, paths, ROI, probes, vis settings)
  - arrays.npz      (displacement, strain, roi_mask — compressed)

Each displacement frame stored separately as displacement_NNN to avoid
allocating a large contiguous 4D array in memory.
"""

import io
import json
import logging
import os
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
SOFTWARE_ID = "RAFT-DIC GUI"


def save_session(session, file_path: str) -> str:
    """
    Serialize the current session state to a .raftproj file.

    Returns the absolute path of the saved file.
    Raises ValueError if there are no results to save.
    """
    if not session.displacement_results:
        raise ValueError("No displacement results to save.")

    metadata = _build_metadata(session)
    arrays = _collect_arrays(session)

    # Write ZIP atomically: write to temp then rename
    abs_path = os.path.abspath(file_path)
    tmp_path = abs_path + ".tmp"

    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # metadata.json
            meta_bytes = json.dumps(metadata, indent=2, ensure_ascii=False).encode("utf-8")
            zf.writestr("metadata.json", meta_bytes)

            # arrays.npz — write numpy arrays into an in-memory buffer, then add to ZIP
            buf = io.BytesIO()
            np.savez_compressed(buf, **arrays)
            buf.seek(0)
            zf.writestr("arrays.npz", buf.read())

        # Atomic rename (Windows: need to remove target first if exists)
        if os.path.exists(abs_path):
            os.replace(tmp_path, abs_path)
        else:
            os.rename(tmp_path, abs_path)

    except Exception:
        # Clean up temp file on failure
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    logger.info("Session saved to %s (%.1f MB)", abs_path, os.path.getsize(abs_path) / 1e6)
    return abs_path


def load_session(file_path: str, session) -> dict:
    """
    Deserialize a .raftproj file and restore session state.

    The caller is responsible for calling session.reset() and clearing
    render caches before this function, and for reloading images afterward.

    Returns dict with:
      - warnings: list of warning strings
      - summary: dict with restored state info
    """
    from raft_dic_gui.config import DICConfig
    from raft_dic_gui.probe_manager import ProbeManager

    abs_path = os.path.abspath(file_path)
    warnings = []

    with zipfile.ZipFile(abs_path, "r") as zf:
        # Read metadata
        with zf.open("metadata.json") as f:
            metadata = json.loads(f.read().decode("utf-8"))

        schema_ver = metadata.get("schema_version", 0)
        if schema_ver > SCHEMA_VERSION:
            warnings.append(
                f"File uses schema v{schema_ver}, this software supports v{SCHEMA_VERSION}. "
                "Some data may not load correctly."
            )

        # Read arrays
        with zf.open("arrays.npz") as f:
            npz_data = np.load(io.BytesIO(f.read()), allow_pickle=False)
            arrays = dict(npz_data)

    # --- Restore config ---
    config_data = metadata.get("config", {})
    # Only set fields that exist in DICConfig
    for key, val in config_data.items():
        if hasattr(session.config, key):
            setattr(session.config, key, val)

    # --- Restore image info ---
    session.image_dir = metadata.get("image_dir", "")
    session.image_files = metadata.get("image_files", [])
    session.image_width = metadata.get("image_width", 0)
    session.image_height = metadata.get("image_height", 0)

    # Check if image directory still exists
    if session.image_dir and not os.path.isdir(session.image_dir):
        warnings.append(
            f"Image directory not found: {session.image_dir}. "
            "Displacement data is loaded but background images will be unavailable."
        )

    # --- Restore ROI ---
    roi = metadata.get("roi_rect")
    if roi:
        session.roi_rect = tuple(roi)
    session.roi_confirmed = metadata.get("roi_confirmed", False)

    envelope = metadata.get("envelope_rect")
    if envelope:
        session.envelope_rect = tuple(envelope)

    if "roi_mask" in arrays:
        session.roi_mask = arrays["roi_mask"].astype(bool)

    # --- Restore displacement results ---
    num_disp = metadata.get("num_displacement_frames", 0)
    displacement_results = []
    for i in range(num_disp):
        key = f"displacement_{i:03d}"
        if key in arrays:
            displacement_results.append(arrays[key].astype(np.float32))
        else:
            warnings.append(f"Missing displacement frame {i}")
            displacement_results.append(None)

    session.displacement_results = displacement_results

    # --- Restore strain results ---
    num_strain = metadata.get("num_strain_frames", 0)
    strain_components = metadata.get("strain_components", [])
    strain_results = []

    for i in range(num_strain):
        frame_dict = {}
        for comp in strain_components:
            key = f"strain_{i:03d}_{comp}"
            if key in arrays:
                frame_dict[comp] = arrays[key].astype(np.float32)
        # Only add if we got at least one component
        if frame_dict:
            strain_results.append(frame_dict)
        else:
            strain_results.append(None)

    session.strain_results = strain_results
    session.strain_components = strain_components

    # --- Restore probes ---
    probes_data = metadata.get("probes", [])
    if probes_data:
        session.probe_manager.load_from_list(probes_data)

    # --- Restore vis settings ---
    vis = metadata.get("vis_settings")
    if vis:
        session.vis_settings.update(vis)

    # Bump result version so caches are invalidated
    session.result_version += 1

    summary = {
        "num_displacement_frames": len(displacement_results),
        "num_strain_frames": len(strain_results),
        "strain_components": strain_components,
        "has_roi": session.roi_rect is not None,
        "roi_confirmed": session.roi_confirmed,
        "num_probes": len(probes_data),
        "image_dir": session.image_dir,
        "image_dir_exists": os.path.isdir(session.image_dir) if session.image_dir else False,
        "image_width": session.image_width,
        "image_height": session.image_height,
    }

    logger.info("Session loaded: %d disp frames, %d strain frames, %d probes",
                len(displacement_results), len(strain_results), len(probes_data))

    return {"warnings": warnings, "summary": summary}


def get_session_info(session) -> dict:
    """Return a summary of the current session state."""
    return {
        "has_results": bool(session.displacement_results),
        "num_displacement_frames": len(session.displacement_results),
        "num_strain_frames": len(session.strain_results),
        "strain_components": list(session.strain_components),
        "has_roi": session.roi_rect is not None,
        "roi_confirmed": session.roi_confirmed,
        "num_probes": len(session.probe_manager.probes),
        "image_dir": session.image_dir,
        "image_width": session.image_width,
        "image_height": session.image_height,
        "config": {
            "mode": session.config.mode,
            "model_label": session.config.model_label,
            "use_smooth": session.config.use_smooth,
            "sigma": session.config.sigma,
        },
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_metadata(session) -> dict:
    """Build the metadata dict for saving."""
    config_dict = asdict(session.config)
    # Remove non-serializable fields
    config_dict.pop("model_metadata", None)

    probes = session.probe_manager.to_list()

    return {
        "schema_version": SCHEMA_VERSION,
        "software": SOFTWARE_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config_dict,
        "image_dir": session.image_dir,
        "image_files": list(session.image_files),
        "image_width": session.image_width,
        "image_height": session.image_height,
        "roi_rect": list(session.roi_rect) if session.roi_rect else None,
        "roi_confirmed": session.roi_confirmed,
        "envelope_rect": list(session.envelope_rect) if session.envelope_rect else None,
        "vis_settings": dict(session.vis_settings),
        "probes": probes,
        "strain_components": list(session.strain_components),
        "num_displacement_frames": len(session.displacement_results),
        "num_strain_frames": len(session.strain_results),
    }


def _collect_arrays(session) -> dict:
    """Collect all numpy arrays for saving, frame by frame."""
    arrays = {}

    # Displacement: each frame as displacement_NNN (H, W, 2) float32
    for i, disp in enumerate(session.displacement_results):
        if disp is not None:
            arr = disp if isinstance(disp, np.ndarray) else np.load(disp)
            arrays[f"displacement_{i:03d}"] = arr.astype(np.float32)

    # Strain: each frame+component as strain_NNN_component (H, W) float32
    for i, strain_dict in enumerate(session.strain_results):
        if strain_dict is None:
            continue
        for comp_name, comp_data in strain_dict.items():
            if comp_data is not None:
                arr = comp_data if isinstance(comp_data, np.ndarray) else np.load(comp_data)
                arrays[f"strain_{i:03d}_{comp_name}"] = arr.astype(np.float32)

    # ROI mask
    if session.roi_mask is not None:
        arrays["roi_mask"] = session.roi_mask.astype(bool)

    return arrays
