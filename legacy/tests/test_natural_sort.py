"""Tests for natural sort functionality in image loading."""

import re


def _natural_sort_key(s: str):
    """Copy of the function from server/routes/images.py for isolated testing."""
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", s)]


class TestNaturalSortKey:
    """Unit tests for the natural sort key function."""

    def test_pure_numeric_filenames(self):
        """1.tif, 2.tif, ..., 11.tif should sort numerically."""
        files = ["1.tif", "11.tif", "2.tif", "20.tif", "3.tif"]
        result = sorted(files, key=_natural_sort_key)
        assert result == ["1.tif", "2.tif", "3.tif", "11.tif", "20.tif"]

    def test_prefix_with_numbers(self):
        """image1.png, image2.png, ..., image11.png should sort naturally."""
        files = ["image1.png", "image11.png", "image2.png", "image20.png", "image3.png"]
        result = sorted(files, key=_natural_sort_key)
        assert result == ["image1.png", "image2.png", "image3.png", "image11.png", "image20.png"]

    def test_1005_after_1(self):
        """1005 should come AFTER 1, not before."""
        files = ["1005.tif", "1.tif", "2.tif", "100.tif"]
        result = sorted(files, key=_natural_sort_key)
        assert result == ["1.tif", "2.tif", "100.tif", "1005.tif"]

    def test_lexicographic_fallback(self):
        """Without natural sort, str.lower gives lexicographic order."""
        files = ["image1.png", "image11.png", "image2.png", "image20.png"]
        result = sorted(files, key=str.lower)
        # Lexicographic: "image1" < "image11" < "image2" < "image20"
        assert result == ["image1.png", "image11.png", "image2.png", "image20.png"]

    def test_mixed_extensions(self):
        files = ["frame1.tif", "frame10.tif", "frame2.jpg", "frame3.png"]
        result = sorted(files, key=_natural_sort_key)
        assert result == ["frame1.tif", "frame2.jpg", "frame3.png", "frame10.tif"]

    def test_leading_zeros(self):
        """Files with leading zeros should sort correctly."""
        files = ["img001.tif", "img010.tif", "img002.tif", "img100.tif"]
        result = sorted(files, key=_natural_sort_key)
        assert result == ["img001.tif", "img002.tif", "img010.tif", "img100.tif"]

    def test_no_numbers(self):
        """Pure alphabetic names should sort alphabetically."""
        files = ["bravo.png", "alpha.png", "charlie.png"]
        result = sorted(files, key=_natural_sort_key)
        assert result == ["alpha.png", "bravo.png", "charlie.png"]

    def test_case_insensitive(self):
        """Natural sort should be case-insensitive."""
        files = ["Image2.png", "image1.png", "IMAGE3.png"]
        result = sorted(files, key=_natural_sort_key)
        assert result == ["image1.png", "Image2.png", "IMAGE3.png"]
