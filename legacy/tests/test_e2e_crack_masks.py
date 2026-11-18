"""End-to-end test: growing crack with per-frame masks.

Generates 6 synthetic speckle frames where:
- Left half: stationary
- Right half: translates +2px/frame rightward
- A vertical crack at the center grows wider each frame

With per-frame masks excluding the crack region, verifies:
1. Displacement coverage matches current frame's specimen boundary
   (not the previous frame's larger boundary)
2. Strain is ~0 everywhere (rigid body translation on each side,
   no cross-crack VSG fitting leakage)
"""

import os
import shutil

import cv2
import numpy as np
import pytest

from raft_dic_gui.config import DICConfig
from raft_dic_gui.controller import DICProcessor
from raft_dic_gui.strain import calculate_strain_field

# ---------------------------------------------------------------------------
# Model availability
# ---------------------------------------------------------------------------
_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "active",
    "RAFTcorr_fine_parameter_more.pth",
)
_HAS_MODEL = os.path.exists(_MODEL_PATH)

N_FRAMES = 6
H, W = 200, 300
CRACK_COL = W // 2           # crack at center column
SHIFT_PER_FRAME = 2          # px rightward per frame for right half
CRACK_GROWTH = 2             # crack widens by this many px per frame


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def _make_speckle(h, w, seed=42):
    """Generate a repeatable speckle pattern."""
    rng = np.random.RandomState(seed)
    # Fine noise + blur → speckle-like texture
    noise = rng.randint(0, 256, (h, w), dtype=np.uint8)
    speckle = cv2.GaussianBlur(noise, (5, 5), 1.2)
    return speckle


def _warp_image(img, u_field, v_field):
    """Warp image using displacement field (backward mapping)."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = xx - u_field.astype(np.float32)
    map_y = yy - v_field.astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def generate_crack_sequence(out_dir):
    """Generate N_FRAMES speckle images + masks with growing crack.

    Returns (img_dir, mask_dir, expected_crack_widths)
    """
    img_dir = os.path.join(out_dir, "images")
    mask_dir = os.path.join(out_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    base_speckle = _make_speckle(H, W, seed=42)

    crack_widths = []

    for i in range(N_FRAMES):
        # Cumulative displacement for right half
        cum_shift = i * SHIFT_PER_FRAME  # frame 0 = 0, frame 1 = 2, ...
        crack_width = i * CRACK_GROWTH   # frame 0 = 0, frame 1 = 2, ...
        crack_widths.append(crack_width)

        # Build displacement field
        u_field = np.zeros((H, W), dtype=np.float64)
        v_field = np.zeros((H, W), dtype=np.float64)

        # Right half shifts rightward (uniform translation)
        u_field[:, CRACK_COL:] = cum_shift

        # Warp base speckle
        warped = _warp_image(base_speckle, u_field, v_field)

        # Insert crack: set crack region to constant gray (no texture)
        if crack_width > 0:
            c_start = CRACK_COL - crack_width // 2 + cum_shift // 2
            c_end = c_start + crack_width
            c_start = max(0, c_start)
            c_end = min(W, c_end)
            warped[:, c_start:c_end] = 128  # featureless → bad correlation

        # Save image
        cv2.imwrite(os.path.join(img_dir, f"frame_{i + 1:03d}.png"), warped)

        # Build mask: True everywhere except crack + thin border
        mask = np.ones((H, W), dtype=np.uint8) * 255
        # Exclude image borders (avoid edge artifacts)
        border = 8
        mask[:border, :] = 0
        mask[-border:, :] = 0
        mask[:, :border] = 0
        mask[:, -border:] = 0
        # Exclude crack region (with small safety margin)
        if crack_width > 0:
            margin = 3
            m_start = max(0, c_start - margin)
            m_end = min(W, c_end + margin)
            mask[:, m_start:m_end] = 0

        cv2.imwrite(os.path.join(mask_dir, f"mask_{i + 1:03d}.png"), mask)

    return img_dir, mask_dir, crack_widths


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def crack_data(tmp_path_factory):
    """Generate synthetic data and run DIC once for all tests."""
    out_dir = str(tmp_path_factory.mktemp("crack_e2e"))
    img_dir, mask_dir, crack_widths = generate_crack_sequence(out_dir)

    if not _HAS_MODEL:
        pytest.skip("RAFT model not found")

    # Initial ROI = full image minus border
    border = 8
    roi_mask = np.ones((H, W), dtype=bool)
    roi_mask[:border, :] = False
    roi_mask[-border:, :] = False
    roi_mask[:, :border] = False
    roi_mask[:, -border:] = False
    y_idx, x_idx = np.where(roi_mask)
    roi_rect = (int(x_idx.min()), int(y_idx.min()),
                int(x_idx.max()) + 1, int(y_idx.max()) + 1)

    config = DICConfig(
        img_dir=img_dir,
        project_root=os.path.join(out_dir, "output"),
        model_path=_MODEL_PATH,
        mode="incremental",
        key_frame_interval=1,  # every frame is a keyframe
        mask_dir=mask_dir,
        use_smooth=True,
        sigma=2.0,
        device="cuda",
    )

    processor = DICProcessor()
    results = processor.run(config, roi_mask, roi_rect)

    return {
        "results": results,
        "roi_rect": roi_rect,
        "crack_widths": crack_widths,
        "mask_dir": mask_dir,
        "img_dir": img_dir,
    }


# ---------------------------------------------------------------------------
# Test 1: Current-frame mask — coverage matches specimen boundary
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_MODEL, reason="RAFT model not found")
class TestDeformedViewCoverage:
    """Verify displacement NaN pattern matches current frame's mask."""

    def test_results_exist(self, crack_data):
        results = crack_data["results"]
        assert len(results) == N_FRAMES - 1, \
            f"Expected {N_FRAMES - 1} displacement results, got {len(results)}"

    def test_frame2_crack_region_is_nan(self, crack_data):
        """Frame 2 has a 2px crack. Pixels in the crack region should be NaN."""
        disp = crack_data["results"][0]  # result for frame 2
        u = disp[:, :, 0]
        valid = ~np.isnan(u)

        # Load frame 2's mask to compare
        mask_path = os.path.join(crack_data["mask_dir"], "mask_002.png")
        mask_2 = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0

        # Crop both to roi_rect for comparison
        x0, y0, x1, y1 = crack_data["roi_rect"]
        mask_2_crop = mask_2[y0:y1, x0:x1]

        # Valid displacement pixels should not exceed mask_2
        n_valid = valid.sum()
        n_mask = mask_2_crop.sum()
        print(f"\nFrame 2: valid disp = {n_valid}, mask = {n_mask}")
        assert n_valid <= n_mask, \
            f"Displacement coverage ({n_valid}) exceeds frame 2 mask ({n_mask})"

    def test_later_frames_coverage_decreases(self, crack_data):
        """As the crack grows, valid displacement area should decrease."""
        results = crack_data["results"]
        valid_counts = []
        for i, disp in enumerate(results):
            n = int((~np.isnan(disp[:, :, 0])).sum())
            valid_counts.append(n)
            print(f"Frame {i + 2}: valid pixels = {n}")

        # Coverage should generally decrease (crack growing)
        # Allow some tolerance for small fluctuations
        assert valid_counts[-1] < valid_counts[0], \
            "Coverage should decrease as crack grows"

    def test_each_frame_within_its_mask(self, crack_data):
        """For every frame, displacement coverage <= that frame's mask."""
        results = crack_data["results"]
        x0, y0, x1, y1 = crack_data["roi_rect"]

        for i, disp in enumerate(results):
            frame_num = i + 2
            mask_path = os.path.join(
                crack_data["mask_dir"], f"mask_{frame_num:03d}.png"
            )
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0
            mask_crop = mask[y0:y1, x0:x1]

            valid = ~np.isnan(disp[:, :, 0])
            n_valid = int(valid.sum())
            n_mask = int(mask_crop.sum())

            assert n_valid <= n_mask, \
                f"Frame {frame_num}: valid disp ({n_valid}) > mask ({n_mask})"


# ---------------------------------------------------------------------------
# Test 2: Strain — no cross-crack fitting leakage
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_MODEL, reason="RAFT model not found")
class TestStrainCrackIsolation:
    """Verify strain near crack is not contaminated by cross-crack fitting."""

    def _compute_strain(self, disp, roi_rect):
        """Compute strain for a displacement result."""
        x0, y0, x1, y1 = roi_rect
        # Embed back into full image for strain calculation
        full = np.full((H, W, 2), np.nan)
        dh, dw = disp.shape[:2]
        full[y0:y0 + dh, x0:x0 + dw, :] = disp
        return calculate_strain_field(
            full, method='engineering', vsg_size=15, step=1,
            boundary_erosion=0,  # disable erosion to test raw VSG
        )

    def test_strain_magnitude_reasonable(self, crack_data):
        """Each side is rigid translation → strain should be near zero.

        Any large strain would indicate cross-crack fitting leakage.
        """
        results = crack_data["results"]
        roi_rect = crack_data["roi_rect"]

        for i, disp in enumerate(results):
            frame_num = i + 2
            strain = self._compute_strain(disp, roi_rect)
            if strain is None:
                continue

            exx = strain['exx']
            valid_exx = exx[~np.isnan(exx)]
            if len(valid_exx) == 0:
                continue

            max_abs_exx = np.max(np.abs(valid_exx))
            p99_exx = np.percentile(np.abs(valid_exx), 99)

            print(f"Frame {frame_num}: max |exx| = {max_abs_exx:.4f}, "
                  f"p99 |exx| = {p99_exx:.4f}")

            # Rigid translation → exx should be very small
            # Allow tolerance for accumulated RAFT noise in incremental mode:
            #   - Unit tests confirm connectivity fix gives |exx| < 1e-6
            #   - Real RAFT noise accumulates ~0.04/frame (p99)
            #   - Without fix, cross-crack fitting would give exx ≈ 2*N/15 ≈ 0.53
            #     for frame 5, so 0.30 threshold still catches real leakage
            assert p99_exx < 0.30, \
                f"Frame {frame_num}: p99 |exx| = {p99_exx:.4f} too high, " \
                f"possible cross-crack fitting leakage"

    def test_strain_near_crack_not_extreme(self, crack_data):
        """Pixels closest to the crack should not have extreme strain.

        Without connectivity fix, cross-crack fitting would produce
        huge strain near the crack boundary.
        """
        results = crack_data["results"]
        roi_rect = crack_data["roi_rect"]

        for i, disp in enumerate(results):
            frame_num = i + 2
            if crack_data["crack_widths"][i + 1] == 0:
                continue  # no crack yet

            strain = self._compute_strain(disp, roi_rect)
            if strain is None:
                continue

            exx = strain['exx']

            # Check a band near the crack (±20 px from center column)
            x0 = roi_rect[0]
            crack_local = CRACK_COL - x0
            band_start = max(0, crack_local - 20)
            band_end = min(exx.shape[1], crack_local + 20)
            near_crack = exx[:, band_start:band_end]
            valid_near = near_crack[~np.isnan(near_crack)]

            if len(valid_near) == 0:
                continue

            max_near = np.max(np.abs(valid_near))
            print(f"Frame {frame_num}: max |exx| near crack = {max_near:.4f}")

            # Without connectivity fix, cross-crack fitting would give
            # exx >> 0.5 (accumulated delta_u across 15px VSG window).
            # With the fix, values should be accumulated RAFT noise only.
            # Threshold 0.35 still catches real leakage (expected >0.5).
            assert max_near < 0.35, \
                f"Frame {frame_num}: extreme strain near crack ({max_near:.4f}), " \
                f"likely cross-crack fitting leakage"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
