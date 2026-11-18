"""Generate the RAFTcorr application icons (SVG source -> PNG, ICO, ICNS).

"RAFT / DIC" in the layout of pyALDIC's "AL / DIC" (a dark rounded square, a white word over a
slanted coloured "DIC") so the two programs read as one family, with RAFTcorr's own colour (the
website's accent blue). Rendered with the native font database (run it without
QT_QPA_PLATFORM=offscreen, which lacks the heavy Segoe UI face).

    python scripts/make_icons.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src" / "raftcorr" / "gui" / "assets" / "icon"
FONT = "Segoe UI, SF Pro Display, Inter, -apple-system, Helvetica, Arial, sans-serif"
BACKGROUND = "#1e293b"
ACCENT = "#4a9eff"
QT_FONT = "Segoe UI Black"

SVG = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <rect x="0" y="0" width="512" height="512" rx="112" ry="112" fill="{BACKGROUND}"></rect>
  <text x="256" y="232" text-anchor="middle" font-family="{FONT}" font-weight="900" font-size="150" letter-spacing="-4" fill="#ffffff">RAFT</text>
  <g transform="translate(256 420) skewX(-10) translate(-256 -420)">
    <text x="256" y="440" text-anchor="middle" font-family="{FONT}" font-weight="900" font-size="180" letter-spacing="-6" fill="{ACCENT}">DIC</text>
  </g>
</svg>
"""


def render(svg: bytes, size: int):
    """The SVG rasterised by Qt at *size* x *size* pixels, as a PIL image."""
    from PIL import Image
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(0)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    QSvgRenderer(QByteArray(svg)).render(p)
    p.end()
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    return Image.open(io.BytesIO(bytes(buf.data()))).convert("RGBA")


def main() -> None:
    from PySide6.QtGui import QGuiApplication

    app = QGuiApplication.instance() or QGuiApplication(sys.argv)  # noqa: F841 - fonts need an app
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RAFTcorr.svg").write_text(SVG, encoding="utf-8")
    # Qt's SVG renderer (SVG Tiny 1.2) takes one family and no clip paths: name the heavy face.
    big = render(SVG.replace(FONT, QT_FONT).encode("utf-8"), 1024)
    big.resize((256, 256), resample=3).save(OUT / "RAFTcorr-256.png")
    big.save(OUT / "RAFTcorr.ico", sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
    try:
        big.save(OUT / "RAFTcorr.icns")
    except Exception as exc:  # ICNS writing depends on the Pillow build; Windows does not need it
        print(f"RAFTcorr.icns skipped: {exc}")
    print(f"icons written to {OUT}")


if __name__ == "__main__":
    main()
