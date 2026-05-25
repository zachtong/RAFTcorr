"""Round-trip correctness tests for streaming scientific export.

Verifies that the streaming writers produce files whose contents match the
mathematical specification (U, V, Magnitude, Velocity, strain components) for
each supported format: .npz, .h5, .mat (v7.3).
"""

from __future__ import annotations

import os
from typing import List

import h5py
import numpy as np
import pytest

from raft_dic_gui.scientific_export import (
    STRAIN_KEYS,
    save_scientific_results_streaming,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_results():
    """Build a small but realistic displacement + strain sequence.

    ROI is centered inside a larger image so the cropping path is exercised.
    """
    H, W = 16, 20
    xmin, ymin, xmax, ymax = 2, 1, 14, 15  # ROI = 14 x 12
    roi_rect = (xmin, ymin, xmax, ymax)
    roi_mask = np.zeros((H, W), dtype=bool)
    roi_mask[ymin:ymax, xmin:xmax] = True
    # Carve a hole to exercise the NaN-mask path
    roi_mask[5:7, 6:8] = False

    T = 5
    rng = np.random.default_rng(seed=42)
    displacement_results: List[np.ndarray] = []
    for t in range(T):
        disp = rng.standard_normal((H, W, 2)).astype(np.float32) * (t + 1)
        displacement_results.append(disp)

    H_roi = ymax - ymin
    W_roi = xmax - xmin
    strain_results = []
    for t in range(T):
        strain_results.append({
            k: rng.standard_normal((H_roi, W_roi)).astype(np.float32) * 0.001
            for k in ("exx", "eyy", "exy")
        })

    metadata = {
        "model_path": "/fake/model.pth",
        "mode": "Full Image",
        "time_step": 0.5,
        "von_mises_assumption": "plane_stress_2D",
    }

    return {
        "displacement_results": displacement_results,
        "strain_results": strain_results,
        "roi_mask": roi_mask,
        "roi_rect": roi_rect,
        "metadata": metadata,
        "T": T, "H": H, "W": W,
        "H_roi": H_roi, "W_roi": W_roi,
    }


def _expected_u_v(data) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation: produce the (T, H_roi, W_roi) U/V stacks."""
    T = data["T"]
    H_roi, W_roi = data["H_roi"], data["W_roi"]
    xmin, ymin, xmax, ymax = data["roi_rect"]
    mask_crop = data["roi_mask"][ymin:ymax, xmin:xmax]
    U = np.full((T, H_roi, W_roi), np.nan, dtype=np.float32)
    V = np.full((T, H_roi, W_roi), np.nan, dtype=np.float32)
    for t in range(T):
        d_crop = data["displacement_results"][t][ymin:ymax, xmin:xmax]
        U[t][mask_crop] = d_crop[..., 0][mask_crop]
        V[t][mask_crop] = d_crop[..., 1][mask_crop]
    return U, V


# ---------------------------------------------------------------------------
# .npz round-trip
# ---------------------------------------------------------------------------

class TestNpzStreaming:
    def test_roundtrip_basic(self, synthetic_results, tmp_path):
        out = str(tmp_path / "result.npz")
        save_scientific_results_streaming(
            displacement_results=synthetic_results["displacement_results"],
            strain_results=synthetic_results["strain_results"],
            roi_mask=synthetic_results["roi_mask"],
            roi_rect=synthetic_results["roi_rect"],
            metadata=synthetic_results["metadata"],
            file_path=out,
        )
        assert os.path.exists(out)

        with np.load(out, allow_pickle=True) as f:
            assert set(["U", "V", "Magnitude", "Velocity",
                        "X_ref", "Y_ref", "ROI_mask"]).issubset(f.files)
            for k in STRAIN_KEYS:
                assert k in f.files

            U_exp, V_exp = _expected_u_v(synthetic_results)
            np.testing.assert_allclose(f["U"], U_exp, equal_nan=True)
            np.testing.assert_allclose(f["V"], V_exp, equal_nan=True)

            mag_exp = np.sqrt(U_exp ** 2 + V_exp ** 2)
            np.testing.assert_allclose(f["Magnitude"], mag_exp, equal_nan=True)

            dt = synthetic_results["metadata"]["time_step"]
            vel = f["Velocity"]
            assert np.all(vel[0] == 0)
            for t in range(1, synthetic_results["T"]):
                du = U_exp[t] - U_exp[t - 1]
                dv = V_exp[t] - V_exp[t - 1]
                expected_vel = np.sqrt(du ** 2 + dv ** 2) / dt
                np.testing.assert_allclose(vel[t], expected_vel, equal_nan=True)

    def test_metadata_preserved(self, synthetic_results, tmp_path):
        out = str(tmp_path / "meta.npz")
        save_scientific_results_streaming(
            displacement_results=synthetic_results["displacement_results"],
            strain_results=synthetic_results["strain_results"],
            roi_mask=synthetic_results["roi_mask"],
            roi_rect=synthetic_results["roi_rect"],
            metadata=synthetic_results["metadata"],
            file_path=out,
        )
        with np.load(out, allow_pickle=True) as f:
            assert "mode" in f.files
            assert str(f["mode"]) == "Full Image"
            assert float(f["time_step"]) == 0.5

    def test_no_strain_results(self, synthetic_results, tmp_path):
        out = str(tmp_path / "no_strain.npz")
        save_scientific_results_streaming(
            displacement_results=synthetic_results["displacement_results"],
            strain_results=[],
            roi_mask=synthetic_results["roi_mask"],
            roi_rect=synthetic_results["roi_rect"],
            metadata=synthetic_results["metadata"],
            file_path=out,
        )
        with np.load(out, allow_pickle=True) as f:
            # Strain arrays should exist but be all NaN
            assert "exx" in f.files
            assert np.all(np.isnan(f["exx"]))


# ---------------------------------------------------------------------------
# .h5 round-trip
# ---------------------------------------------------------------------------

class TestH5Streaming:
    def test_roundtrip_basic(self, synthetic_results, tmp_path):
        out = str(tmp_path / "result.h5")
        save_scientific_results_streaming(
            displacement_results=synthetic_results["displacement_results"],
            strain_results=synthetic_results["strain_results"],
            roi_mask=synthetic_results["roi_mask"],
            roi_rect=synthetic_results["roi_rect"],
            metadata=synthetic_results["metadata"],
            file_path=out,
        )
        assert os.path.exists(out)

        with h5py.File(out, "r") as f:
            assert "U" in f and "V" in f and "Magnitude" in f and "Velocity" in f
            U = f["U"][:]
            V = f["V"][:]
            U_exp, V_exp = _expected_u_v(synthetic_results)
            np.testing.assert_allclose(U, U_exp, equal_nan=True)
            np.testing.assert_allclose(V, V_exp, equal_nan=True)

    def test_chunked_one_frame_per_chunk(self, synthetic_results, tmp_path):
        out = str(tmp_path / "chunked.h5")
        save_scientific_results_streaming(
            displacement_results=synthetic_results["displacement_results"],
            strain_results=synthetic_results["strain_results"],
            roi_mask=synthetic_results["roi_mask"],
            roi_rect=synthetic_results["roi_rect"],
            metadata=synthetic_results["metadata"],
            file_path=out,
        )
        with h5py.File(out, "r") as f:
            T = synthetic_results["T"]
            H_roi = synthetic_results["H_roi"]
            W_roi = synthetic_results["W_roi"]
            assert f["U"].shape == (T, H_roi, W_roi)
            # Chunking on time axis: one frame per chunk
            assert f["U"].chunks == (1, H_roi, W_roi)
            assert f["U"].compression == "gzip"


# ---------------------------------------------------------------------------
# .mat v7.3 round-trip (read back via h5py)
# ---------------------------------------------------------------------------

class TestMatV73Streaming:
    def test_roundtrip_with_transpose(self, synthetic_results, tmp_path):
        out = str(tmp_path / "result.mat")
        save_scientific_results_streaming(
            displacement_results=synthetic_results["displacement_results"],
            strain_results=synthetic_results["strain_results"],
            roi_mask=synthetic_results["roi_mask"],
            roi_rect=synthetic_results["roi_rect"],
            metadata=synthetic_results["metadata"],
            file_path=out,
        )
        assert os.path.exists(out)

        # .mat v7.3 stores arrays transposed so MATLAB reads them naturally.
        # From Python we have to transpose back to recover (T, H, W).
        with h5py.File(out, "r") as f:
            U_disk = f["U"][:]
            T = synthetic_results["T"]
            H_roi = synthetic_results["H_roi"]
            W_roi = synthetic_results["W_roi"]
            assert U_disk.shape == (W_roi, H_roi, T)  # reversed axes

            U = U_disk.transpose(2, 1, 0)
            U_exp, _ = _expected_u_v(synthetic_results)
            np.testing.assert_allclose(U, U_exp, equal_nan=True)

            # MATLAB_class attr present (h5py 3.x may auto-decode bytes -> str)
            def _as_bytes(v):
                return v.encode("ascii") if isinstance(v, str) else v
            assert _as_bytes(f["U"].attrs["MATLAB_class"]) == b"single"
            assert _as_bytes(f["ROI_mask"].attrs["MATLAB_class"]) == b"uint8"

    def test_userblock_signature(self, synthetic_results, tmp_path):
        out = str(tmp_path / "with_header.mat")
        save_scientific_results_streaming(
            displacement_results=synthetic_results["displacement_results"],
            strain_results=synthetic_results["strain_results"],
            roi_mask=synthetic_results["roi_mask"],
            roi_rect=synthetic_results["roi_rect"],
            metadata=synthetic_results["metadata"],
            file_path=out,
        )
        with open(out, "rb") as fp:
            header = fp.read(116)
        assert header.startswith(b"MATLAB 7.3 MAT-file")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_unknown_extension_defaults_to_npz(self, synthetic_results, tmp_path):
        out = str(tmp_path / "result.xyz")
        result_path = save_scientific_results_streaming(
            displacement_results=synthetic_results["displacement_results"],
            strain_results=synthetic_results["strain_results"],
            roi_mask=synthetic_results["roi_mask"],
            roi_rect=synthetic_results["roi_rect"],
            metadata=synthetic_results["metadata"],
            file_path=out,
        )
        assert result_path.endswith(".npz")
        assert os.path.exists(result_path)

    def test_empty_displacement_raises(self, synthetic_results, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            save_scientific_results_streaming(
                displacement_results=[],
                strain_results=[],
                roi_mask=synthetic_results["roi_mask"],
                roi_rect=synthetic_results["roi_rect"],
                metadata={},
                file_path=str(tmp_path / "empty.npz"),
            )


# ---------------------------------------------------------------------------
# Memory bound smoke test
# ---------------------------------------------------------------------------

class TestMemoryBound:
    """Streaming should keep peak in-memory allocation O(H*W), not O(T*H*W).

    We can't easily measure RSS portably; instead we verify the writer can
    process a tall sequence whose full-stack would clearly exceed our scratch
    arrays by writing 200 frames at modest ROI.
    """

    def test_many_frames(self, tmp_path):
        H, W = 8, 10
        T = 200
        roi_rect = (0, 0, W, H)
        roi_mask = np.ones((H, W), dtype=bool)
        disp = [
            np.ones((H, W, 2), dtype=np.float32) * t
            for t in range(T)
        ]
        strain = [
            {"exx": np.ones((H, W), dtype=np.float32) * 1e-4}
            for _ in range(T)
        ]

        out = str(tmp_path / "many.h5")
        save_scientific_results_streaming(
            displacement_results=disp, strain_results=strain,
            roi_mask=roi_mask, roi_rect=roi_rect,
            metadata={"time_step": 1.0}, file_path=out,
        )
        with h5py.File(out, "r") as f:
            assert f["U"].shape == (T, H, W)
            assert f["U"][0, 0, 0] == 0.0
            assert f["U"][T - 1, 0, 0] == T - 1
            # Velocity[t>=1] = sqrt((t-(t-1))^2 + same^2)/dt = sqrt(2)
            np.testing.assert_allclose(
                f["Velocity"][1, 0, 0], np.sqrt(2.0), atol=1e-6,
            )
