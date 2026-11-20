"""The version is written in three places; a release must change all three together."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from raftcorr import __version__

ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    path = ROOT / name
    if not path.is_file():
        pytest.skip(f"{name} is not in this copy (e.g. an sdist)")
    return path.read_text(encoding="utf-8")


def test_citation_version_is_the_package_version():
    m = re.search(r'^version:\s*"?([^"\s]+)"?\s*$', _read("CITATION.cff"), re.MULTILINE)
    assert m is not None, "CITATION.cff has no version"
    assert m.group(1) == __version__


def test_changelog_starts_with_the_package_version():
    m = re.search(r"^## (\S+)", _read("CHANGELOG.md"), re.MULTILINE)
    assert m is not None, "CHANGELOG.md has no version heading"
    assert m.group(1) == __version__
