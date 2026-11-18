"""Self-test: every installed model on a synthetic speckle pair with a known translation."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import synthetic as syn
from .callbacks import Callbacks, coerce

SHIFT = (2.30, -1.20)
TOLERANCE_PX = 0.1


@dataclass(frozen=True)
class SelfTestReport:
    ok: bool
    rows: tuple[tuple[str, float, float, float, str], ...]  # model, u, v, seconds, status

    @property
    def text(self) -> str:
        lines = [f"Imposed translation: u = {SHIFT[0]:+.2f}, v = {SHIFT[1]:+.2f} px (256 x 256 synthetic speckle)"]
        for key, u, v, sec, status in self.rows:
            if np.isfinite(u):
                lines.append(f"RAFTcorr-{key}: u = {u:+.3f}, v = {v:+.3f} px, {sec:.2f} s  {status}")
            else:
                lines.append(f"RAFTcorr-{key}: {status}")
        lines.append("All installed models pass." if self.ok else "Some models FAILED.")
        return "\n".join(lines)


def run_selftest(device: str = "auto", callbacks: Callbacks | None = None) -> SelfTestReport:
    from .engine.backend import Predictor
    from .models import available_models, load_model

    cb = coerce(callbacks)
    ref, dfm = syn.render_pair(256, 256, syn.translation(*SHIFT), seed=11)
    rows = []
    ok = True
    avail = available_models()
    for i, (key, path) in enumerate(avail.items()):
        cb.progress(i / len(avail), f"self-test RAFTcorr-{key}")
        if path is None:
            rows.append((key, float("nan"), float("nan"), 0.0, "not installed"))
            continue
        try:
            pred = Predictor(load_model(path, device), iters=20)
            t0 = time.perf_counter()
            flow = pred.predict(ref, dfm)[:, 32:-32, 32:-32]
            sec = time.perf_counter() - t0
            u, v = float(np.median(flow[0])), float(np.median(flow[1]))
            good = abs(u - SHIFT[0]) < TOLERANCE_PX and abs(v - SHIFT[1]) < TOLERANCE_PX
            ok &= good
            rows.append((key, u, v, sec, "ok" if good else "WRONG"))
            del pred
        except Exception as exc:  # report every model
            ok = False
            rows.append((key, float("nan"), float("nan"), 0.0, f"error: {exc}"))
    cb.progress(1.0, "self-test done")
    if all(r[4] == "not installed" for r in rows):
        ok = False
    return SelfTestReport(ok, tuple(rows))


def gui_selftest() -> tuple[bool, str]:
    """Open and close the main window offscreen: Qt, the icons and the seven translations load.

    For a fresh installation (``raftcorr selftest --gui``); the window never shows on screen.
    """
    import os
    from pathlib import Path

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    lines: list[str] = []
    try:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        from .gui.app import MainWindow
        from .gui.icons import icon_app

        win = MainWindow()
        win.show()
        app.processEvents()
        title = win.windowTitle()
        win.close()
        app.processEvents()
        compiled = Path(__file__).resolve().parent / "i18n" / "compiled"
        catalogs = sorted(p.stem.removeprefix("raftcorr_") for p in compiled.glob("raftcorr_*.qm"))
        checks = [
            ("main window", bool(title), title),
            ("application icon", not icon_app().isNull(), ""),
            ("translations", len(catalogs) == 7, ", ".join(catalogs)),
        ]
    except Exception as exc:  # report, do not raise: this is a diagnostic
        return False, f"GUI self-test: error: {type(exc).__name__}: {exc}"
    ok = all(good for _name, good, _detail in checks)
    for name, good, detail in checks:
        lines.append(f"  {name:18s} {'ok' if good else 'MISSING'}  {detail}".rstrip())
    return ok, "GUI self-test: " + ("ok" if ok else "FAILED") + "\n" + "\n".join(lines)
