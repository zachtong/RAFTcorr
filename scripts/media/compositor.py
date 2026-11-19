"""Frame compositor of the scripted RAFTcorr quick-start video.

Every frame is composed on a master canvas twice the output size and then reduced with an exact 2:1
area filter. The application is captured at a device-pixel ratio of 2, so at the default camera the
window maps 1:1 onto the master canvas; camera zooms only ever up-sample the capture (no aliasing
shimmer on speckle images), and the final 2:1 reduction anti-aliases the pointer and the captions.

Time is a virtual clock (``Video.t``). Animations (pointer glide, camera zoom, cross-fades, click
ripples, caption fades) are tweens evaluated at the frame time, so several of them can overlap.

Adapted from the pyALDIC tutorial-video compositor (same author).
"""
from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor, QFont, QFontMetricsF, QImage, QPainter, QPainterPath, QPen,
)

OUT_W, OUT_H = 1920, 1080
SS = 2                      # master canvas = SS x output
FPS = 60
DPR = 2                     # device-pixel ratio of every capture

# The "screen" that shows the app, in OUTPUT pixels (x, y, w, h).
VIEW = (240, 30, 1440, 900)
DESK_W, DESK_H = 1600, 1000           # app window size in logical pixels

BG = (243, 244, 246)        # page background (light gray)
ACCENT = QColor("#4f46e5")
TEXT = QColor("#111827")
TEXT_MUTED = QColor("#6b7280")

FONT_CJK = "Microsoft YaHei UI"
FONT_LATIN = "Segoe UI"


# ----------------------------------------------------------------- easing
def ease_in_out(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u) if u < 1 else 1.0


def ease_in_out_cubic(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return 4 * u ** 3 if u < 0.5 else 1 - (-2 * u + 2) ** 3 / 2


def ease_out(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return 1 - (1 - u) ** 3


@dataclass(frozen=True)
class Tween:
    """A value moving from ``v0`` to ``v1`` over [t0, t0 + dur]."""
    v0: tuple
    v1: tuple
    t0: float
    dur: float
    bow: float = 0.0        # perpendicular arc (fraction of distance), points only
    ease: Callable[[float], float] = ease_in_out_cubic

    def at(self, t: float) -> tuple:
        if self.dur <= 0 or t >= self.t0 + self.dur:
            return self.v1
        if t <= self.t0:
            return self.v0
        u = self.ease((t - self.t0) / self.dur)
        out = [a + (b - a) * u for a, b in zip(self.v0, self.v1)]
        if self.bow and len(out) == 2:
            dx, dy = self.v1[0] - self.v0[0], self.v1[1] - self.v0[1]
            s = math.sin(math.pi * u) * self.bow
            out[0] += -dy * s
            out[1] += dx * s
        return tuple(out)

    def done(self, t: float) -> bool:
        return t >= self.t0 + self.dur


def still(v: tuple) -> Tween:
    return Tween(v, v, 0.0, 0.0)


FULL_CAM = (0.0, 0.0, float(DESK_W), float(DESK_H))


def cam_around(cx: float, cy: float, zoom: float) -> tuple:
    """A 16:10 camera rect centred on (cx, cy), clamped to the desktop."""
    w, h = DESK_W / zoom, DESK_H / zoom
    x = min(max(cx - w / 2, 0.0), DESK_W - w)
    y = min(max(cy - h / 2, 0.0), DESK_H - h)
    return (x, y, w, h)


# --------------------------------------------------------- static layers
def qimage_view(arr: np.ndarray) -> QImage:
    """Wrap a contiguous HxWx3 (RGB) or HxWx4 (RGBA premultiplied) array."""
    h, w, c = arr.shape
    fmt = QImage.Format.Format_RGB888 if c == 3 else QImage.Format.Format_RGBA8888_Premultiplied
    return QImage(arr.data, w, h, arr.strides[0], fmt)


def qimage_to_rgb(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    h, w, bpl = img.height(), img.width(), img.bytesPerLine()
    buf = np.frombuffer(img.constBits(), np.uint8, count=h * bpl).reshape(h, bpl)
    return buf[:, : w * 3].reshape(h, w, 3).copy()


def _rounded_mask(w: int, h: int, r: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (r, 0), (w - r - 1, h - 1), 255, -1)
    cv2.rectangle(m, (0, r), (w - 1, h - r - 1), 255, -1)
    for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)):
        cv2.circle(m, (cx, cy), r, 255, -1, lineType=cv2.LINE_AA)
    return (m.astype(np.float32) / 255.0)[..., None]


def build_background() -> np.ndarray:
    """Page background with a soft shadow under the screen viewport."""
    W, H = OUT_W * SS, OUT_H * SS
    bg = np.empty((H, W, 3), np.float32)
    bg[:] = BG
    x, y, w, h = (v * SS for v in VIEW)
    shadow = np.zeros((H, W), np.float32)
    cv2.rectangle(shadow, (x + 10, y + 26), (x + w - 10, y + h + 14), 1.0, -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 34) * 0.30
    bg *= (1.0 - shadow[..., None])
    return bg.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------- cursor
_ARROW = [(0, 0), (0, 17.2), (4.1, 13.3), (6.9, 19.9), (9.6, 18.8),
          (6.8, 12.3), (12.4, 12.3)]


def _arrow_path(scale: float) -> QPainterPath:
    p = QPainterPath()
    p.moveTo(*(_ARROW[0][0] * scale, _ARROW[0][1] * scale))
    for x, y in _ARROW[1:]:
        p.lineTo(x * scale, y * scale)
    p.closeSubpath()
    return p


def draw_cursor(p: QPainter, x: float, y: float, press: float) -> None:
    """Classic arrow pointer, tip at (x, y) in master pixels."""
    s = 1.45 * SS * (1.0 - 0.12 * press)
    path = _arrow_path(s)
    p.save()
    p.translate(x + 1.2 * SS, y + 2.0 * SS)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(0, 0, 0, 55))
    p.drawPath(path)
    p.restore()
    p.save()
    p.translate(x, y)
    p.setPen(QPen(QColor(20, 20, 20), 1.15 * SS, Qt.PenStyle.SolidLine,
                  Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.setBrush(QColor(255, 255, 255))
    p.drawPath(path)
    p.restore()


def draw_ripple(p: QPainter, x: float, y: float, u: float) -> None:
    """Click feedback: a soft accent ring that expands and fades (u in 0..1)."""
    r = (7 + 20 * ease_out(u)) * SS
    a = 1.0 - u
    c = QColor(ACCENT)
    c.setAlphaF(0.16 * a)
    p.setBrush(c)
    ring = QColor(ACCENT)
    ring.setAlphaF(0.55 * a)
    p.setPen(QPen(ring, 2.0 * SS))
    p.drawEllipse(QPointF(x, y), r, r)


# --------------------------------------------------------------- caption
@dataclass(frozen=True)
class Caption:
    step: str          # e.g. "1 / 6"
    title: str
    sub: str
    tag: str = ""      # e.g. "4x" speed badge


def _font(size_out: float, weight: QFont.Weight, family: str) -> QFont:
    f = QFont(family)
    f.setPixelSize(round(size_out * SS))
    f.setWeight(weight)
    f.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    return f


def render_caption(cap: Caption, cjk: bool) -> np.ndarray:
    """Caption block (RGBA premultiplied) centred in the bottom band."""
    W = OUT_W * SS
    band_top = (VIEW[1] + VIEW[3]) * SS
    H = OUT_H * SS - band_top
    layer = np.zeros((H, W, 4), np.uint8)
    img = qimage_view(layer)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    fam = FONT_CJK if cjk else FONT_LATIN
    f_step = _font(17, QFont.Weight.DemiBold, FONT_LATIN)
    f_title = _font(27, QFont.Weight.DemiBold, fam)
    f_sub = _font(19, QFont.Weight.Normal, fam)
    f_tag = _font(14, QFont.Weight.DemiBold, FONT_LATIN)

    m_step, m_title = QFontMetricsF(f_step), QFontMetricsF(f_title)
    m_sub, m_tag = QFontMetricsF(f_sub), QFontMetricsF(f_tag)
    gap = 14 * SS
    pill_pad = 9 * SS
    step_w = m_step.horizontalAdvance(cap.step) + 2 * pill_pad
    title_w = m_title.horizontalAdvance(cap.title)
    tag_w = (m_tag.horizontalAdvance(cap.tag) + 2 * 7 * SS) if cap.tag else 0
    row_w = step_w + gap + title_w + ((gap + tag_w) if cap.tag else 0)
    x0 = (W - row_w) / 2
    y_title = 36 * SS                       # baseline-ish centre line of row 1
    pill_h = m_step.height() + 6 * SS

    # step pill
    pill = QRectF(x0, y_title - pill_h / 2, step_w, pill_h)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(ACCENT)
    p.drawRoundedRect(pill, pill_h / 2, pill_h / 2)
    p.setFont(f_step)
    p.setPen(QColor("white"))
    p.drawText(pill, Qt.AlignmentFlag.AlignCenter, cap.step)

    # title
    p.setFont(f_title)
    p.setPen(TEXT)
    tr = QRectF(x0 + step_w + gap, y_title - m_title.height() / 2, title_w + 4, m_title.height())
    p.drawText(tr, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, cap.title)

    # optional speed tag
    if cap.tag:
        th = m_tag.height() + 6 * SS
        tag = QRectF(tr.left() + title_w + gap, y_title - th / 2, tag_w, th)
        p.setPen(QPen(TEXT_MUTED, 1.2 * SS))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(tag, 5 * SS, 5 * SS)
        p.setFont(f_tag)
        p.drawText(tag, Qt.AlignmentFlag.AlignCenter, cap.tag)

    # subtitle line
    p.setFont(f_sub)
    p.setPen(TEXT_MUTED)
    sr = QRectF(0, y_title + 24 * SS, W, m_sub.height() + 4)
    p.drawText(sr, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, cap.sub)
    p.end()
    return layer


def render_card(lines: list[tuple[str, float, QFont.Weight, QColor, str]],
                icon: QImage | None, gap_out: float = 18) -> np.ndarray:
    """Full-frame title/end card (opaque RGB) with centred text lines.

    ``lines`` = [(text, size_out, weight, color, family), ...].
    """
    W, H = OUT_W * SS, OUT_H * SS
    arr = np.empty((H, W, 3), np.uint8)
    arr[:] = (250, 250, 251)
    img = qimage_view(arr)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    fonts = [_font(sz, wt, fam) for _, sz, wt, _, fam in lines]
    heights = [QFontMetricsF(f).height() for f in fonts]
    icon_h = 150 * SS if icon is not None else 0
    total = icon_h + (gap_out * 2 * SS if icon is not None else 0) + sum(heights) \
        + gap_out * SS * (len(lines) - 1)
    y = (H - total) / 2 - 20 * SS
    if icon is not None:
        p.drawImage(QRectF((W - icon_h) / 2, y, icon_h, icon_h), icon)
        y += icon_h + gap_out * 2 * SS
    for (text, _, _, color, _), f, h in zip(lines, fonts, heights):
        p.setFont(f)
        p.setPen(color)
        p.drawText(QRectF(0, y, W, h), Qt.AlignmentFlag.AlignCenter, text)
        y += h + gap_out * SS
    p.end()
    return arr


# ----------------------------------------------------------------- video
@dataclass
class Overlay:
    """A secondary top-level window shown on the desktop (e.g. a dialog)."""
    image: np.ndarray       # RGB at DPR 2
    x: float                # desktop logical position of its top-left
    y: float


class Video:
    """Accumulates frames on a virtual clock and pipes them to ffmpeg.

    Rendering is layered so the per-frame cost stays low: the background
    plus the camera view of the desktop is cached as a *base frame* keyed by
    (desktop version, cross-fade phase, camera); cursor, ripples, captions
    and cards are drawn on a copy of it.  Versions are explicit counters --
    never ``id()`` -- because a freed capture's address can be reused by the
    next one and would silently serve a stale frame.
    """

    def __init__(self, path: str, ffmpeg: str, cjk: bool) -> None:
        self.cjk = cjk
        self.t = 0.0
        self.frames = 0
        self._bg = build_background()
        vx, vy, vw, vh = (v * SS for v in VIEW)
        self._vp = (vx, vy, vw, vh)
        self._corner_r = 14 * SS
        self._corner_mask = _rounded_mask(vw, vh, self._corner_r)
        self.desktop: np.ndarray | None = None
        self._overlays: list[Overlay] = []
        self._desk_ver = 0
        self._desk_cache: tuple[int, np.ndarray] | None = None
        self._xfade: tuple[np.ndarray, float, float] | None = None
        self._base_key = None
        self._base: np.ndarray | None = None
        self.cursor = still((DESK_W * 0.55, DESK_H * 0.6))
        self.cursor_visible = True
        self.camera = still(FULL_CAM)
        self.ripples: list[tuple[float, float, float]] = []
        self._cap_cache: dict[Caption, tuple] = {}
        self.caption: Caption | None = None
        self._cap_prev: Caption | None = None
        self._cap_t0 = -10.0
        self.card: np.ndarray | None = None
        self._card_ver = 0
        self.card_alpha = still((0.0,))
        self._last_key = None
        self._last_bytes: bytes | None = None
        self._proc = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OUT_W}x{OUT_H}",
             "-r", str(FPS), "-i", "-",
             "-vf", "scale=out_color_matrix=bt709:out_range=tv",
             "-c:v", "libx264", "-preset", "slow", "-crf", "16",
             "-pix_fmt", "yuv420p", "-colorspace", "bt709",
             "-color_primaries", "bt709", "-color_trc", "bt709",
             "-movflags", "+faststart", path],
            stdin=subprocess.PIPE)

    # -- state setters ------------------------------------------------
    @property
    def overlays(self) -> list[Overlay]:
        return self._overlays

    @overlays.setter
    def overlays(self, value: list[Overlay]) -> None:
        self._overlays = list(value)
        self._desk_ver += 1

    def set_desktop(self, img: np.ndarray, fade: float = 0.0) -> None:
        if fade > 0 and self.desktop is not None:
            self._xfade = (self._composed_desktop(), self.t, fade)
        self.desktop = img
        self._desk_ver += 1

    def caption_to(self, cap: Caption | None) -> None:
        self._cap_prev, self.caption, self._cap_t0 = self.caption, cap, self.t

    def move(self, x: float, y: float, dur: float = 0.0, wait: bool = True) -> None:
        """Glide the cursor to (x, y); ``dur`` <= 0 picks a natural duration."""
        p0 = self.cursor.at(self.t)
        dist = math.hypot(x - p0[0], y - p0[1])
        dur = dur if dur > 0 else min(0.9, 0.38 + dist / 1500)
        self.cursor = Tween(p0, (x, y), self.t, dur, bow=0.06)
        if wait:
            self.hold(dur)

    def click(self, settle: float = 0.18) -> None:
        x, y = self.cursor.at(self.t)
        self.ripples.append((x, y, self.t))
        self.hold(settle)

    def zoom(self, cam: tuple, dur: float = 0.8, wait: bool = False) -> None:
        self.camera = Tween(self.camera.at(self.t), cam, self.t, dur)
        if wait:
            self.hold(dur)

    def show_card(self, card: np.ndarray, fade: float) -> None:
        self.card = card
        self._card_ver += 1
        self.card_alpha = Tween(self.card_alpha.at(self.t), (1.0,), self.t, fade,
                                ease=ease_in_out)

    def hide_card(self, fade: float) -> None:
        self.card_alpha = Tween(self.card_alpha.at(self.t), (0.0,), self.t, fade,
                                ease=ease_in_out)

    # -- timeline -----------------------------------------------------
    def hold(self, dur: float) -> None:
        n = max(0, round(dur * FPS))
        for _ in range(n):
            self._emit()

    def _emit(self) -> None:
        key = self._state_key()
        if key == self._last_key and self._last_bytes is not None:
            data = self._last_bytes
        else:
            data = self._render().tobytes()
            self._last_key, self._last_bytes = key, data
        self._proc.stdin.write(data)
        self.frames += 1
        self.t = self.frames / FPS

    def close(self) -> None:
        self._proc.stdin.close()
        if self._proc.wait() != 0:
            raise RuntimeError("ffmpeg failed to encode the video")

    # -- rendering ----------------------------------------------------
    def _xfade_phase(self) -> float | None:
        if self._xfade is None:
            return None
        _, t0, dur = self._xfade
        if self.t >= t0 + dur:
            self._xfade = None
            return None
        return round(ease_in_out((self.t - t0) / dur), 5)

    def _state_key(self):
        t = self.t
        ripples = tuple((x, y, round(t - t0, 4)) for x, y, t0 in self.ripples
                        if t - t0 < 0.45)
        cap_phase = round(min(t - self._cap_t0, 1.0), 4)
        return (self._desk_ver, self._xfade_phase(), self.camera.at(t),
                self.cursor.at(t) if self.cursor_visible else None, ripples,
                self.caption, self._cap_prev, cap_phase,
                self._card_ver, self.card_alpha.at(t))

    def _composed_desktop(self) -> np.ndarray:
        if self._desk_cache is None or self._desk_cache[0] != self._desk_ver:
            out = self.desktop
            if self._overlays:
                out = out.copy()
                for o in self._overlays:
                    _paste_window(out, o.image, int(round(o.x * DPR)), int(round(o.y * DPR)))
            self._desk_cache = (self._desk_ver, out)
        return self._desk_cache[1]

    def _base_frame(self) -> np.ndarray:
        """Background + rounded camera view of the desktop (cached)."""
        phase = self._xfade_phase()
        cam = self.camera.at(self.t)
        key = (self._desk_ver, phase, cam)
        if key == self._base_key:
            return self._base
        desk = self._composed_desktop()
        if phase is not None:
            desk = cv2.addWeighted(self._xfade[0], 1.0 - phase, desk, phase, 0.0)
        cx, cy, cw, _ = cam
        vx, vy, vw, vh = self._vp
        k = vw / cw                                  # master px per logical px
        view = _camera_view(desk, k / DPR, cx * k, cy * k, vw, vh)
        frame = self._bg.copy()
        frame[vy:vy + vh, vx:vx + vw] = view
        r = self._corner_r                           # blend only the 4 corners
        for ys in (slice(0, r), slice(vh - r, vh)):
            for xs in (slice(0, r), slice(vw - r, vw)):
                m = self._corner_mask[ys, xs]
                bg = self._bg[vy:vy + vh, vx:vx + vw][ys, xs]
                frame[vy:vy + vh, vx:vx + vw][ys, xs] = \
                    (bg * (1.0 - m) + view[ys, xs] * m).astype(np.uint8)
        self._base_key, self._base = key, frame
        return frame

    def _render(self) -> np.ndarray:
        t = self.t
        frame = self._base_frame().copy()
        cx, cy, cw, _ = self.camera.at(t)
        vx, vy, vw, vh = self._vp
        k = vw / cw

        def to_master(px: float, py: float) -> tuple[float, float]:
            return vx + (px - cx) * k, vy + (py - cy) * k

        self.ripples = [r for r in self.ripples if t - r[2] < 0.45]
        if self.ripples or self.cursor_visible:
            qi = qimage_view(frame)
            p = QPainter(qi)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setClipRect(QRectF(vx, vy, vw, vh))
            for rx, ry, t0 in self.ripples:
                draw_ripple(p, *to_master(rx, ry), (t - t0) / 0.45)
            if self.cursor_visible:
                press = 0.0
                for _, _, t0 in self.ripples:
                    press = max(press, math.sin(math.pi * min((t - t0) / 0.22, 1.0)))
                draw_cursor(p, *to_master(*self.cursor.at(t)), press)
            p.end()

        self._blend_captions(frame)
        a = self.card_alpha.at(t)[0]
        if self.card is not None and a > 0:
            frame = cv2.addWeighted(frame, 1.0 - a, self.card, a, 0.0) if a < 1 else self.card
        return cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)

    def _cap_layer(self, cap: Caption) -> tuple:
        """Caption cropped to its opaque bbox: (y0, x0, rgb_premul, alpha) floats."""
        if cap not in self._cap_cache:
            lay = render_caption(cap, self.cjk)
            ys, xs = np.nonzero(lay[..., 3])
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            crop = lay[y0:y1, x0:x1].astype(np.float32)
            self._cap_cache[cap] = (int(y0), int(x0), crop[..., :3], crop[..., 3:4] / 255.0)
        return self._cap_cache[cap]

    def _blend_captions(self, frame: np.ndarray) -> None:
        """Old caption fades out (0.22 s), then the new one fades/slides in."""
        dt = self.t - self._cap_t0
        out_d, in_d = 0.22, 0.32
        band_top = (VIEW[1] + VIEW[3]) * SS
        layers = []
        if self._cap_prev is not None and dt < out_d:
            layers.append((self._cap_prev, 1.0 - ease_in_out(dt / out_d), 0.0))
        if self.caption is not None:
            u = ease_in_out((dt - out_d) / in_d) if self._cap_prev is not None else \
                ease_in_out(dt / in_d)
            if u > 0:
                layers.append((self.caption, u, (1.0 - u) * 8 * SS))
        for cap, alpha, dy in layers:
            y0, x0, rgb, a = self._cap_layer(cap)
            h, w = a.shape[:2]
            ty = band_top + y0 + int(round(dy))
            ty1 = min(ty + h, frame.shape[0])
            n = ty1 - ty
            dst = frame[ty:ty1, x0:x0 + w].astype(np.float32)
            out = dst * (1.0 - a[:n] * alpha) + rgb[:n] * alpha
            frame[ty:ty1, x0:x0 + w] = out.clip(0, 255).astype(np.uint8)


def _camera_view(desk: np.ndarray, s: float, ox: float, oy: float,
                 vw: int, vh: int) -> np.ndarray:
    """Scale ``desk`` by ``s`` and crop the (vw x vh) window at master (ox, oy).

    Down-scaling goes through an area filter first (no aliasing on speckle
    images); the remaining sub-pixel offset and any up-scaling use bilinear
    warping, which keeps slow camera moves free of 1-px jitter.
    """
    if s < 0.999:
        h, w = desk.shape[:2]
        desk = cv2.resize(desk, (max(1, round(w * s)), max(1, round(h * s))),
                          interpolation=cv2.INTER_AREA)
        s_rest = 1.0
    else:
        s_rest = s
    m = np.array([[s_rest, 0.0, -ox], [0.0, s_rest, -oy]], np.float32)
    return cv2.warpAffine(desk, m, (vw, vh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _paste_window(dst: np.ndarray, img: np.ndarray, x: int, y: int) -> None:
    """Paste a secondary window with a soft drop shadow (in place)."""
    h, w = img.shape[:2]
    pad = 60
    H, W = dst.shape[:2]
    sx0, sy0 = max(x - pad, 0), max(y - pad, 0)
    sx1, sy1 = min(x + w + pad, W), min(y + h + pad, H)
    sh = np.zeros((sy1 - sy0, sx1 - sx0), np.float32)
    cv2.rectangle(sh, (x - sx0 + 6, y - sy0 + 18), (x - sx0 + w - 6, y - sy0 + h + 8), 1.0, -1)
    sh = cv2.GaussianBlur(sh, (0, 0), 22) * 0.35
    region = dst[sy0:sy1, sx0:sx1].astype(np.float32)
    dst[sy0:sy1, sx0:sx1] = (region * (1 - sh[..., None])).astype(np.uint8)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    dst[y0:y1, x0:x1] = img[y0 - y:y1 - y, x0 - x:x1 - x]
