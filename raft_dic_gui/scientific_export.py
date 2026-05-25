"""Streaming exporters for full-field DIC results.

Replaces the in-memory `save_scientific_results` implementation which
allocated 12 simultaneous (T, H, W) float32 stacks. For 2000-frame
sequences at typical ROIs (1000x1000) that peaked at ~96 GB RAM and OOM'd.

This module writes frame-by-frame:
  - HDF5 (.h5/.hdf5): chunked + gzip, one frame per chunk along the time axis
  - MAT v7.3 (.mat): same HDF5 layout with MATLAB_class attrs and transposed
    dataspace so MATLAB reads arrays as (T, H, W) natively
  - NPZ (.npz): pre-allocated .npy memmaps per variable, then bundled into the
    final ZIP container at the end (~2x disk peak, O(H*W) RAM peak)

Peak RAM cost: O(H_roi * W_roi) per frame -- a few hundred MB for typical
configurations, regardless of T.
"""

from __future__ import annotations

import os
import shutil
import struct
import tempfile
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np
from numpy.lib.format import open_memmap


STRAIN_KEYS: Tuple[str, ...] = (
    "exx", "eyy", "exy", "e1", "e2", "max_shear", "von_mises", "rotation",
)


# ---------------------------------------------------------------------------
# Per-frame processor
# ---------------------------------------------------------------------------

class _FrameProcessor:
    """Crop, mask, and compute derived quantities for a single frame.

    Holds prev_u/prev_v across calls so velocity (finite difference) can be
    computed without keeping the full series in memory.
    """

    def __init__(
        self,
        roi_mask: np.ndarray,
        roi_rect: Tuple[int, int, int, int],
        strain_h: int,
        strain_w: int,
        upsample_strain: bool,
        dt: float,
    ) -> None:
        self.xmin, self.ymin, self.xmax, self.ymax = roi_rect
        self.H, self.W = roi_mask.shape
        self.mask_crop = roi_mask[self.ymin:self.ymax, self.xmin:self.xmax]
        self.H_roi, self.W_roi = self.mask_crop.shape
        self.valid = self.mask_crop > 0
        self.strain_h = strain_h
        self.strain_w = strain_w
        self.upsample_strain = upsample_strain
        self.dt = dt if dt > 0 else 1.0
        self.prev_u: Optional[np.ndarray] = None
        self.prev_v: Optional[np.ndarray] = None

    def process(
        self,
        t: int,
        displacement_field: Any,
        strain_dict: Optional[Dict[str, np.ndarray]],
    ) -> Optional[Dict[str, np.ndarray]]:
        """Return per-frame arrays keyed by variable name, or None on failure.

        Output keys: U, V, Magnitude, Velocity, plus any strain components
        present in `strain_dict` and recognized in `STRAIN_KEYS`.
        """
        try:
            d = displacement_field
            if isinstance(d, str):
                d = np.load(d)

            if d is None or getattr(d, "size", 0) == 0:
                return None

            d_crop = self._crop_displacement(t, d)
            if d_crop is None:
                return None

            u_frame = d_crop[..., 0].astype(np.float32, copy=True)
            v_frame = d_crop[..., 1].astype(np.float32, copy=True)
            u_frame[~self.valid] = np.nan
            v_frame[~self.valid] = np.nan

            mag = np.sqrt(u_frame * u_frame + v_frame * v_frame)

            if self.prev_u is not None and self.prev_v is not None:
                du = u_frame - self.prev_u
                dv = v_frame - self.prev_v
                vel = np.sqrt(du * du + dv * dv) / self.dt
            else:
                vel = np.zeros_like(u_frame)

            result: Dict[str, np.ndarray] = {
                "U": u_frame, "V": v_frame, "Magnitude": mag, "Velocity": vel,
            }
            result.update(self._process_strain(strain_dict))

            self.prev_u = u_frame
            self.prev_v = v_frame
            return result

        except Exception as exc:
            print(f"[Error] Frame {t}: {exc}")
            return None

    def _crop_displacement(self, t: int, d: np.ndarray) -> Optional[np.ndarray]:
        h_d, w_d = d.shape[:2]
        if h_d == self.H_roi and w_d == self.W_roi:
            d_crop = d
        elif h_d == self.H and w_d == self.W:
            d_crop = d[self.ymin:self.ymax, self.xmin:self.xmax]
        else:
            print(f"[Warning] Frame {t}: shape {d.shape} incompatible "
                  f"with ROI {self.H_roi}x{self.W_roi} or full "
                  f"{self.H}x{self.W}. Using raw.")
            d_crop = d
        if d_crop.shape[0] != self.H_roi or d_crop.shape[1] != self.W_roi:
            print(f"[Error] Frame {t}: cropped shape {d_crop.shape} != "
                  f"mask {self.H_roi}x{self.W_roi}. Skipping.")
            return None
        return d_crop

    def _process_strain(
        self, strain_dict: Optional[Dict[str, np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        if not strain_dict:
            return {}
        out: Dict[str, np.ndarray] = {}
        for k in STRAIN_KEYS:
            val = strain_dict.get(k)
            if val is None:
                continue
            h_s, w_s = val.shape
            if h_s == self.strain_h and w_s == self.strain_w:
                val_crop = val
            elif h_s == self.H and w_s == self.W:
                val_crop = val[self.ymin:self.ymax, self.xmin:self.xmax]
            elif h_s == self.H_roi and w_s == self.W_roi:
                val_crop = val
            elif self.upsample_strain:
                val_crop = cv2.resize(
                    val.astype(np.float32),
                    (self.strain_w, self.strain_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                continue

            val_final = val_crop.astype(np.float32, copy=True)
            if val_final.shape == self.valid.shape:
                val_final[~self.valid] = np.nan
            out[k] = val_final
        return out


# ---------------------------------------------------------------------------
# Helpers for export geometry
# ---------------------------------------------------------------------------

def _derive_shapes(
    displacement_results: Sequence,
    strain_results: Sequence,
    roi_mask: np.ndarray,
    roi_rect: Tuple[int, int, int, int],
    upsample_strain: bool,
    metadata: Dict[str, Any],
) -> Tuple[int, int, int, int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve T, H_roi, W_roi, strain_h, strain_w, plus static grids/mask.

    Mutates a copy of metadata to record `strain_native_size` when applicable.
    Returns the same metadata dict back to the caller (potentially mutated).
    """
    xmin, ymin, xmax, ymax = roi_rect
    mask_crop = roi_mask[ymin:ymax, xmin:xmax]
    H_roi, W_roi = mask_crop.shape

    x_range = np.arange(xmin, xmax)
    y_range = np.arange(ymin, ymax)
    X_ref, Y_ref = np.meshgrid(x_range, y_range)

    if not upsample_strain and strain_results:
        first_strain = next((s for s in strain_results if s), None)
        if first_strain:
            first_key = next(iter(first_strain))
            native_h, native_w = first_strain[first_key].shape
            metadata["strain_native_size"] = (native_h, native_w)
            return (
                len(displacement_results), H_roi, W_roi,
                native_h, native_w, mask_crop, X_ref, Y_ref,
            )

    return (
        len(displacement_results), H_roi, W_roi,
        H_roi, W_roi, mask_crop, X_ref, Y_ref,
    )


# ---------------------------------------------------------------------------
# HDF5 streaming writer (plain .h5 and MAT v7.3)
# ---------------------------------------------------------------------------

_NUMPY_TO_MATLAB_CLASS: Dict[np.dtype, bytes] = {
    np.dtype("float32"): b"single",
    np.dtype("float64"): b"double",
    np.dtype("int8"): b"int8",
    np.dtype("int16"): b"int16",
    np.dtype("int32"): b"int32",
    np.dtype("int64"): b"int64",
    np.dtype("uint8"): b"uint8",
    np.dtype("uint16"): b"uint16",
    np.dtype("uint32"): b"uint32",
    np.dtype("uint64"): b"uint64",
    np.dtype("bool"): b"logical",
}


def _set_matlab_class(dset: h5py.Dataset, dtype: np.dtype) -> None:
    """Tag a dataset with the MATLAB_class attr expected by MAT v7.3 readers."""
    cls = _NUMPY_TO_MATLAB_CLASS.get(np.dtype(dtype))
    if cls is None:
        return
    dset.attrs.create("MATLAB_class", cls)
    if cls == b"logical":
        dset.attrs.create("MATLAB_int_decode", np.int32(1))


def _disk_shape(np_shape: Tuple[int, ...], mat_v73: bool) -> Tuple[int, ...]:
    """Return the on-disk dataspace shape.

    For .mat v7.3 we reverse axes so MATLAB (which iterates dims in F-order)
    sees the same logical shape as numpy.
    """
    return tuple(np_shape[::-1]) if mat_v73 else tuple(np_shape)


def _pick_chunks(np_shape: Tuple[int, ...], mat_v73: bool) -> Tuple[int, ...]:
    """One frame per chunk along the time axis."""
    if mat_v73:
        return tuple(np_shape[-1:0:-1]) + (1,)  # = (..., W, H, 1)
    return (1,) + tuple(np_shape[1:])


def _write_frame(
    dset: h5py.Dataset, t: int, frame: np.ndarray, mat_v73: bool,
) -> None:
    """Write a single (H, W) frame at time index t into a streaming dataset."""
    if mat_v73:
        dset[:, :, t] = frame.T
    else:
        dset[t] = frame


def _create_streaming(
    f: h5py.File,
    name: str,
    np_shape: Tuple[int, ...],
    dtype: np.dtype,
    mat_v73: bool,
    fill: float = np.nan,
) -> h5py.Dataset:
    shape = _disk_shape(np_shape, mat_v73)
    chunks = _pick_chunks(np_shape, mat_v73)
    fill_value = fill if np.issubdtype(np.dtype(dtype), np.floating) else 0
    dset = f.create_dataset(
        name, shape=shape, dtype=dtype,
        chunks=chunks, compression="gzip", compression_opts=4, shuffle=True,
        fillvalue=fill_value,
    )
    if mat_v73:
        _set_matlab_class(dset, dtype)
    return dset


def _write_static_array(
    f: h5py.File, name: str, arr: np.ndarray, mat_v73: bool,
) -> None:
    """Write a small static array (coordinates, masks, scalars)."""
    arr = np.asarray(arr)
    if mat_v73 and arr.ndim >= 1:
        # MATLAB sees reversed axes -- store transposed so it reads back as `arr`
        data = np.ascontiguousarray(arr.T)
    else:
        data = arr
    dset = f.create_dataset(name, data=data)
    if mat_v73:
        _set_matlab_class(dset, arr.dtype)


def _write_matlab_string(f: h5py.File, name: str, value: str) -> None:
    """Store a Python string as a MATLAB char array (uint16 row vector)."""
    arr = np.array([ord(c) for c in value], dtype=np.uint16)
    if arr.size == 0:
        arr = np.zeros((0, 0), dtype=np.uint16)
        dset = f.create_dataset(name, data=arr)
    else:
        # MATLAB char vectors are (1, N); HDF5 stores transposed: (N, 1)
        dset = f.create_dataset(name, data=arr.reshape(-1, 1))
    dset.attrs.create("MATLAB_class", b"char")
    dset.attrs.create("MATLAB_int_decode", np.int32(2))


def _write_metadata_entry(
    f: h5py.File, key: str, value: Any, mat_v73: bool,
) -> None:
    """Best-effort metadata serialization for HDF5 / MAT v7.3."""
    try:
        if value is None:
            return
        if isinstance(value, str):
            if mat_v73:
                _write_matlab_string(f, key, value)
            else:
                f.create_dataset(key, data=value)
            return
        if isinstance(value, bool):
            arr = np.array(value, dtype=bool)
        elif isinstance(value, (int, np.integer)):
            arr = np.array(value, dtype=np.int64)
        elif isinstance(value, (float, np.floating)):
            arr = np.array(value, dtype=np.float64)
        elif isinstance(value, (list, tuple)):
            inferred = np.asarray(value)
            if inferred.dtype.kind in ("U", "O", "S"):
                joined = "\n".join(str(v) for v in value)
                if mat_v73:
                    _write_matlab_string(f, key, joined)
                else:
                    f.create_dataset(
                        key,
                        data=np.array(
                            [str(v) for v in value],
                            dtype=h5py.string_dtype(encoding="utf-8"),
                        ),
                    )
                return
            arr = inferred
        else:
            arr = np.asarray(value)
        _write_static_array(f, key, arr, mat_v73)
    except Exception as exc:
        print(f"[Warning] Could not serialize metadata key '{key}': {exc}")


def _write_matlab_userblock(file_path: str) -> None:
    """Write the 512-byte MATLAB v7.3 header into the HDF5 userblock.

    Without this, MATLAB will still open the file (since it's valid HDF5 with
    MATLAB_class attrs), but some MATLAB versions emit a warning. Writing the
    proper header avoids that.
    """
    timestamp = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    text = (f"MATLAB 7.3 MAT-file, Platform: PCWIN64, "
            f"Created on: {timestamp} HDF5 schema 1.00").encode("ascii")
    text = text[:116].ljust(116, b" ")
    subsystem = b"\x00" * 8
    version = struct.pack("<H", 0x0200)
    endian = b"IM"  # little-endian
    header = text + subsystem + version + endian  # 128 bytes total

    with open(file_path, "r+b") as fp:
        fp.seek(0)
        fp.write(header)


def _write_hdf5_streaming(
    file_path: str,
    *,
    displacement_results: Sequence,
    strain_results: Sequence,
    roi_mask: np.ndarray,
    roi_rect: Tuple[int, int, int, int],
    metadata: Dict[str, Any],
    image_files: Optional[List[str]],
    upsample_strain: bool,
    warped_masks: Optional[List[np.ndarray]],
    mat_v73: bool,
) -> str:
    metadata = dict(metadata)
    (T, H_roi, W_roi, strain_h, strain_w, mask_crop,
     X_ref, Y_ref) = _derive_shapes(
        displacement_results, strain_results, roi_mask, roi_rect,
        upsample_strain, metadata,
    )
    dt = float(metadata.get("time_step", 1.0) or 1.0)
    processor = _FrameProcessor(
        roi_mask, roi_rect, strain_h, strain_w, upsample_strain, dt,
    )

    userblock_size = 512 if mat_v73 else 0
    with h5py.File(file_path, "w", userblock_size=userblock_size) as f:
        u_dset = _create_streaming(f, "U", (T, H_roi, W_roi), np.float32, mat_v73)
        v_dset = _create_streaming(f, "V", (T, H_roi, W_roi), np.float32, mat_v73)
        mag_dset = _create_streaming(
            f, "Magnitude", (T, H_roi, W_roi), np.float32, mat_v73,
        )
        vel_dset = _create_streaming(
            f, "Velocity", (T, H_roi, W_roi), np.float32, mat_v73, fill=0.0,
        )
        strain_dsets: Dict[str, h5py.Dataset] = {
            k: _create_streaming(
                f, k, (T, strain_h, strain_w), np.float32, mat_v73,
            )
            for k in STRAIN_KEYS
        }

        _write_static_array(f, "X_ref", X_ref.astype(np.int32), mat_v73)
        _write_static_array(f, "Y_ref", Y_ref.astype(np.int32), mat_v73)
        _write_static_array(f, "ROI_mask", mask_crop.astype(np.uint8), mat_v73)

        for key, value in metadata.items():
            _write_metadata_entry(f, key, value, mat_v73)

        if image_files:
            _write_metadata_entry(f, "image_files", list(image_files), mat_v73)

        if warped_masks:
            for i, wm in enumerate(warped_masks):
                _write_static_array(
                    f, f"warped_mask_{i}", np.asarray(wm).astype(np.uint8),
                    mat_v73,
                )
            _write_metadata_entry(
                f, "num_warped_masks", int(len(warped_masks)), mat_v73,
            )

        # Frame-by-frame streaming
        for t in range(T):
            strain_t = strain_results[t] if t < len(strain_results) else None
            frame_data = processor.process(
                t, displacement_results[t], strain_t,
            )
            if frame_data is None:
                continue
            _write_frame(u_dset, t, frame_data["U"], mat_v73)
            _write_frame(v_dset, t, frame_data["V"], mat_v73)
            _write_frame(mag_dset, t, frame_data["Magnitude"], mat_v73)
            _write_frame(vel_dset, t, frame_data["Velocity"], mat_v73)
            for k in STRAIN_KEYS:
                if k in frame_data:
                    _write_frame(strain_dsets[k], t, frame_data[k], mat_v73)
            del frame_data

    if mat_v73:
        _write_matlab_userblock(file_path)

    return file_path


# ---------------------------------------------------------------------------
# NPZ streaming writer
# ---------------------------------------------------------------------------

def _write_npz_streaming(
    file_path: str,
    *,
    displacement_results: Sequence,
    strain_results: Sequence,
    roi_mask: np.ndarray,
    roi_rect: Tuple[int, int, int, int],
    metadata: Dict[str, Any],
    image_files: Optional[List[str]],
    upsample_strain: bool,
    warped_masks: Optional[List[np.ndarray]],
) -> str:
    """Stream-write to .npz via per-variable memmaps, then bundle into a ZIP.

    Peak RAM: O(H_roi * W_roi). Peak disk: ~2x final compressed size (temp
    memmaps + final .npz). Both are cleaned up before return.
    """
    metadata = dict(metadata)
    (T, H_roi, W_roi, strain_h, strain_w, mask_crop,
     X_ref, Y_ref) = _derive_shapes(
        displacement_results, strain_results, roi_mask, roi_rect,
        upsample_strain, metadata,
    )
    dt = float(metadata.get("time_step", 1.0) or 1.0)
    processor = _FrameProcessor(
        roi_mask, roi_rect, strain_h, strain_w, upsample_strain, dt,
    )

    tmpdir = tempfile.mkdtemp(prefix="raft_export_")
    try:
        memmaps: Dict[str, np.memmap] = {}

        def _alloc(name: str, shape: Tuple[int, ...], fill: float = np.nan) -> np.memmap:
            mm = open_memmap(
                os.path.join(tmpdir, f"{name}.npy"),
                mode="w+", dtype=np.float32, shape=shape,
            )
            if not np.isnan(fill):
                mm[:] = fill
            else:
                # Touch only chunks lazily; explicit NaN fill is needed because
                # memmaps initialize to zero by default.
                mm[:] = np.nan
            return mm

        memmaps["U"] = _alloc("U", (T, H_roi, W_roi))
        memmaps["V"] = _alloc("V", (T, H_roi, W_roi))
        memmaps["Magnitude"] = _alloc("Magnitude", (T, H_roi, W_roi))
        memmaps["Velocity"] = _alloc("Velocity", (T, H_roi, W_roi), fill=0.0)
        for k in STRAIN_KEYS:
            memmaps[k] = _alloc(k, (T, strain_h, strain_w))

        # Frame-by-frame streaming
        for t in range(T):
            strain_t = strain_results[t] if t < len(strain_results) else None
            frame_data = processor.process(
                t, displacement_results[t], strain_t,
            )
            if frame_data is None:
                continue
            memmaps["U"][t] = frame_data["U"]
            memmaps["V"][t] = frame_data["V"]
            memmaps["Magnitude"][t] = frame_data["Magnitude"]
            memmaps["Velocity"][t] = frame_data["Velocity"]
            for k in STRAIN_KEYS:
                if k in frame_data:
                    memmaps[k][t] = frame_data[k]
            del frame_data

        # Flush and close memmaps before zipping (Windows file handles).
        for mm in memmaps.values():
            mm.flush()
        memmaps.clear()

        # Static + metadata
        np.save(os.path.join(tmpdir, "X_ref.npy"), X_ref)
        np.save(os.path.join(tmpdir, "Y_ref.npy"), Y_ref)
        np.save(os.path.join(tmpdir, "ROI_mask.npy"), mask_crop)
        for k, v in metadata.items():
            try:
                np.save(
                    os.path.join(tmpdir, f"{k}.npy"),
                    np.asarray(v) if not isinstance(v, str) else v,
                    allow_pickle=True,
                )
            except Exception as exc:
                print(f"[Warning] Could not save metadata '{k}': {exc}")
        if image_files:
            np.save(
                os.path.join(tmpdir, "image_files.npy"),
                np.array(image_files, dtype=object),
                allow_pickle=True,
            )
        if warped_masks:
            for i, wm in enumerate(warped_masks):
                np.save(
                    os.path.join(tmpdir, f"warped_mask_{i}.npy"),
                    np.asarray(wm).astype(np.uint8),
                )
            np.save(
                os.path.join(tmpdir, "num_warped_masks.npy"),
                np.int32(len(warped_masks)),
            )

        with zipfile.ZipFile(
            file_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True,
        ) as zf:
            for fn in sorted(os.listdir(tmpdir)):
                zf.write(os.path.join(tmpdir, fn), arcname=fn)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return file_path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def save_scientific_results_streaming(
    displacement_results: Sequence,
    strain_results: Sequence,
    roi_mask: np.ndarray,
    roi_rect: Tuple[int, int, int, int],
    metadata: Dict[str, Any],
    file_path: str,
    image_files: Optional[List[str]] = None,
    upsample_strain: bool = True,
    warped_masks: Optional[List[np.ndarray]] = None,
) -> str:
    """Streaming replacement for `save_scientific_results`.

    Dispatches by extension:
      .npz                          -> _write_npz_streaming
      .mat                          -> _write_hdf5_streaming(mat_v73=True)
      .h5 / .hdf5                   -> _write_hdf5_streaming(mat_v73=False)
      anything else                 -> .npz (with ext appended)

    Peak RAM cost is O(H_roi * W_roi) per frame regardless of T.
    """
    if not displacement_results:
        raise ValueError("displacement_results is empty")
    if roi_mask is None or roi_rect is None:
        raise ValueError("roi_mask and roi_rect are required")

    ext = os.path.splitext(file_path)[1].lower()

    common_kwargs = dict(
        displacement_results=displacement_results,
        strain_results=strain_results or [],
        roi_mask=roi_mask,
        roi_rect=roi_rect,
        metadata=metadata or {},
        image_files=image_files,
        upsample_strain=upsample_strain,
        warped_masks=warped_masks,
    )

    if ext == ".npz":
        return _write_npz_streaming(file_path, **common_kwargs)
    if ext == ".mat":
        return _write_hdf5_streaming(file_path, mat_v73=True, **common_kwargs)
    if ext in (".h5", ".hdf5"):
        return _write_hdf5_streaming(file_path, mat_v73=False, **common_kwargs)

    # Unknown extension -- default to .npz
    new_path = file_path + ".npz"
    return _write_npz_streaming(new_path, **common_kwargs)
