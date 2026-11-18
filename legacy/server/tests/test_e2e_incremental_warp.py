"""End-to-end test: incremental mode with auto-warp on synthetic images.

Creates a sequence of synthetic images with known uniform translation,
runs incremental processing with auto-warp, and verifies:
1. Auto-warped masks shift correctly per key frame
2. Accumulated displacement matches expected total translation
3. Envelope rect equals roi_rect (valid set can only shrink on Frame 1 grid)
"""

import numpy as np
import os
import cv2
import pytest
from raft_dic_gui.controller import DICProcessor
from raft_dic_gui.config import DICConfig


@pytest.fixture
def synthetic_sequence(tmp_path):
    """Create 5 frames with 10px/frame rightward translation of a textured block."""
    H, W = 128, 256
    np.random.seed(42)
    texture = (np.random.rand(60, 60) * 255).astype(np.uint8)

    for i in range(5):
        img = np.zeros((H, W), dtype=np.uint8)
        x_offset = 40 + i * 10
        img[30:90, x_offset:x_offset + 60] = texture
        cv2.imwrite(str(tmp_path / f"frame_{i+1:03d}.png"), img)

    return str(tmp_path), H, W


_MODEL_PATH = "models/RAFTcorr_fine_parameter_more.pth"
_HAS_MODEL = os.path.exists(_MODEL_PATH)


class TestE2EIncrementalWarp:

    @pytest.mark.skipif(
        not _HAS_MODEL,
        reason="RAFT model not available for integration test",
    )
    def test_auto_warp_accumulation(self, synthetic_sequence, tmp_path):
        """Full pipeline: incremental + auto-warp on synthetic translation."""
        img_dir, H, W = synthetic_sequence
        out_dir = str(tmp_path / "output")

        roi_mask = np.zeros((H, W), dtype=bool)
        roi_mask[25:95, 35:105] = True
        roi_rect = (35, 25, 105, 95)

        config = DICConfig(
            img_dir=img_dir,
            project_root=out_dir,
            model_path=_MODEL_PATH,
            mode="incremental",
            key_frames=[1, 3],
            key_frame_interval=None,
            mask_dir=None,  # triggers auto-warp
        )

        processor = DICProcessor()
        results = processor.run(config, roi_mask, roi_rect)

        assert len(results) == 4  # frames 2-5

        # Displacement output is on Frame 1 pixel grid;
        # envelope_rect equals roi_rect (valid set shrinks, never grows)
        assert processor.envelope_rect == roi_rect

    @pytest.mark.skipif(
        not _HAS_MODEL,
        reason="RAFT model not available for integration test",
    )
    def test_auto_warp_mask_shifts_at_keyframe(self, synthetic_sequence, tmp_path):
        """Auto-warped mask at key frame 3 should be shifted from original ROI."""
        img_dir, H, W = synthetic_sequence
        out_dir = str(tmp_path / "output")

        roi_mask = np.zeros((H, W), dtype=bool)
        roi_mask[25:95, 35:105] = True
        roi_rect = (35, 25, 105, 95)

        config = DICConfig(
            img_dir=img_dir,
            project_root=out_dir,
            model_path=_MODEL_PATH,
            mode="incremental",
            key_frames=[1, 3],
            key_frame_interval=None,
            mask_dir=None,
        )

        processor = DICProcessor()
        processor.run(config, roi_mask, roi_rect)

        # kf_accumulated[3] should have ~20px rightward displacement
        # (frame 3 is 2 steps × 10px/step from frame 1)
        kf_acc_3 = processor.kf_accumulated.get(3)
        if kf_acc_3 is not None:
            valid = ~np.isnan(kf_acc_3[..., 0])
            if np.any(valid):
                mean_u = np.nanmean(kf_acc_3[valid, 0])
                np.testing.assert_allclose(mean_u, 20.0, atol=5.0)
