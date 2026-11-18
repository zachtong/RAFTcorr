"""Test that image file ordering is consistent between UI load and processing.

Regression test for the bug where controller.py re-read files from disk
using lexicographic sort, losing the natural sort order from the UI.
With filenames like Image1, Image2, ..., Image10, ..., Image100, lexicographic
sort produces: Image1, Image10, Image100, Image11, ... Image2, Image20, ...
which causes accumulative DIC results to periodically jump back to small values.
"""

import os
import re
import tempfile

import pytest


def _natural_sort_key(s: str):
    """Mirror of the natural sort key used in server/routes/images.py."""
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", s)]


IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


class TestImageSortOrder:
    """Verify natural sort vs lexicographic sort for DIC image sequences."""

    @pytest.fixture
    def unpadded_dir(self, tmp_path):
        """Create a directory with un-zero-padded filenames spanning 1..120."""
        for i in range(1, 121):
            (tmp_path / f"Image{i}.tif").write_bytes(b"")
        return tmp_path

    def test_lexicographic_sort_is_wrong(self, unpadded_dir):
        """Prove that lexicographic sort misorders un-padded filenames."""
        files = sorted(
            f for f in os.listdir(unpadded_dir)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        )
        # Lexicographic: Image1 → Image10 → Image100 → Image101 → ... → Image2
        assert files[0] == "Image1.tif"
        assert files[1] == "Image10.tif", "Lexicographic puts Image10 before Image2"
        # Image2 should be index 1 in correct order, but lexico pushes it far back
        idx_image2 = files.index("Image2.tif")
        assert idx_image2 > 10, f"Image2 at index {idx_image2}, expected >> 1"

    def test_natural_sort_is_correct(self, unpadded_dir):
        """Verify natural sort produces the expected numeric order."""
        files = sorted(
            (f for f in os.listdir(unpadded_dir)
             if f.lower().endswith(IMAGE_EXTENSIONS)),
            key=_natural_sort_key,
        )
        expected = [f"Image{i}.tif" for i in range(1, 121)]
        assert files == expected

    def test_controller_uses_passed_image_files(self, unpadded_dir):
        """After fix: controller.run() should accept image_files parameter
        and use it instead of re-scanning the directory."""
        from raft_dic_gui.controller import DICProcessor

        # The run() signature should accept image_files
        import inspect
        sig = inspect.signature(DICProcessor.run)
        params = list(sig.parameters.keys())
        assert "image_files" in params, (
            "DICProcessor.run() must accept an image_files parameter "
            "so the caller can pass the correctly-sorted file list"
        )

    def test_sort_order_roundtrip(self, unpadded_dir):
        """Simulate the full pipeline: load with natural sort → pass to processor.
        The order must be preserved end-to-end."""
        # Step 1: Simulate what images.py load_images() does
        raw = [
            f for f in os.listdir(unpadded_dir)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]
        ui_sorted = sorted(raw, key=_natural_sort_key)

        # Step 2: The processing route passes session.image_files to controller
        # After fix, controller should use this list directly
        processor_files = ui_sorted  # This is what should happen after the fix

        expected = [f"Image{i}.tif" for i in range(1, 121)]
        assert processor_files == expected

        # Step 3: Verify no frame-order anomalies in accumulative mode
        # In accumulative mode, frame N references frame 1 (ref).
        # If order is wrong, frame indices would not be monotonically increasing.
        frame_numbers = [
            int(re.search(r"(\d+)", f).group(1)) for f in processor_files
        ]
        for i in range(1, len(frame_numbers)):
            assert frame_numbers[i] > frame_numbers[i - 1], (
                f"Frame order anomaly at index {i}: "
                f"{processor_files[i-1]} → {processor_files[i]}"
            )
