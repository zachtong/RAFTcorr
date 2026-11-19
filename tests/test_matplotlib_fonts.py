"""Charts in Chinese, Japanese and Korean: matplotlib must find the CJK glyphs of axis labels.

matplotlib falls back glyph by glyph only through the families listed in ``font.family``; a generic
family ("sans-serif") resolves to a single font, and with Segoe UI first every CJK label of the
Analysis charts came out as boxes on Windows.
"""

from __future__ import annotations

import warnings

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from matplotlib import font_manager  # noqa: E402

from raftcorr.gui.support import matplotlib_fonts  # noqa: E402

LABELS = {"zh_CN": "引伸计应变", "zh_TW": "引伸計應變", "ja": "伸び計ひずみ", "ko": "신율계 변형률"}
#: A font that covers each script, among those the fallback chain names.
COVERING = {
    "zh_CN": ("Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "Source Han Sans CN", "WenQuanYi Zen Hei"),
    "zh_TW": ("Microsoft JhengHei", "PingFang TC", "Noto Sans CJK TC"),
    "ja": ("Yu Gothic UI", "Hiragino Sans", "Noto Sans CJK JP"),
    "ko": ("Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans CJK KR"),
}


@pytest.fixture
def configured():
    saved = matplotlib.rcParams.copy()
    matplotlib_fonts.configure_matplotlib_fonts()
    yield
    matplotlib.rcParams.update(saved)


def _installed() -> set[str]:
    return {f.name for f in font_manager.fontManager.ttflist}


@pytest.mark.parametrize("lang", sorted(LABELS))
def test_cjk_axis_labels_render_without_missing_glyphs(configured, lang):
    if not set(COVERING[lang]) & _installed():
        pytest.skip(f"no font for {lang} on this computer")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(2, 1.5))
    ax.set_ylabel(LABELS[lang])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.canvas.draw()
    plt.close(fig)
    missing = [str(w.message) for w in caught if "missing from font" in str(w.message)]
    assert missing == []


def test_the_font_list_names_only_installed_fonts_and_ends_with_a_universal_one(configured):
    family = matplotlib.rcParams["font.family"]
    assert isinstance(family, list) and family
    assert set(family) <= _installed() | {"DejaVu Sans"}
    assert family[-1] == "DejaVu Sans" and matplotlib.rcParams["axes.unicode_minus"] is False
