"""Drive the GUI end to end without a user and save screenshots.

    python scripts/gui_smoke.py <image folder> [--model s4] [--out reports_v2/gui] [--offscreen]

Loads the folder, draws a rectangular ROI (central 60 %), runs the correlation, computes strain,
places a point and a line probe, and saves screenshots of the main window at each stage.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def wait(app, pred, timeout: float = 600.0) -> None:
    t0 = time.time()
    while not pred():
        app.processEvents()
        time.sleep(0.02)
        if time.time() - t0 > timeout:
            raise TimeoutError("timed out")
    for _ in range(20):
        app.processEvents()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--model", default="s4")
    ap.add_argument("--out", default=str(ROOT / "reports_v2" / "gui"))
    ap.add_argument("--offscreen", action="store_true")
    ap.add_argument("--theme", default="light")
    ap.add_argument("--field", default="exx")
    ap.add_argument("--deformed", action="store_true")
    ap.add_argument("--roi", default="0.2,0.2,0.8,0.8", help="x0,y0,x1,y1 as fractions of the image")
    ap.add_argument("--mode", default="accumulative")
    args = ap.parse_args()
    if args.offscreen:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np

    from raftcorr.gui_basic import theme
    from raftcorr.gui_basic.app import build_app
    from raftcorr.gui_basic.main_window import MainWindow
    from raftcorr.post.probes import Probe, color_for
    from raftcorr.roi.shapes import Rect, apply_shape

    app = build_app([])
    theme.set_current(args.theme)
    app.setStyleSheet(theme.stylesheet())
    win = MainWindow()
    win.resize(1600, 1000)
    win.show()
    app.processEvents()
    win.grab().save(str(out / "0_empty.png"))

    assert win.load_folder(Path(args.folder))
    st = win.state
    h, w = st.sequence.shape
    fx0, fy0, fx1, fy1 = (float(t) for t in args.roi.split(","))
    roi = apply_shape(np.zeros((h, w), bool), Rect(fx0 * w, fy0 * h, fx1 * w, fy1 * h), "add")
    st.set_roi(0, roi)
    from raftcorr.config import TrackingConfig

    st.set_run_config(replace(st.run_config, model=args.model, tracking=TrackingConfig(mode=args.mode)))
    win.set_tool("rect")
    wait(app, lambda: True)
    win.grab().save(str(out / "1_roi.png"))
    win.set_tool("pan")

    t0 = time.time()
    win.run_correlation()
    wait(app, lambda: win._worker is None and st.run_state == "idle")
    print(f"correlation: {time.time() - t0:.1f} s, frames done {int(st.series.done.sum())}/{st.series.n_frames}")
    st.set_frame(st.series.n_frames - 1)
    st.update_view(field="u")
    wait(app, lambda: True)
    win.grab().save(str(out / "2_u.png"))

    win.compute_strain()
    wait(app, lambda: win._worker is None and st.run_state == "idle")
    print(f"strain done: {st.strain is not None}")
    cx, cy = 0.5 * (fx0 + fx1) * w, 0.5 * (fy0 + fy1) * h
    dx, dy = 0.25 * (fx1 - fx0) * w, 0.25 * (fy1 - fy0) * h
    st.add_probe(Probe(1, "point", ((cx, cy),), color_for(0)))
    st.add_probe(Probe(2, "line", ((cx - dx, cy + dy), (cx + dx, cy + dy)), color_for(1)))
    st.add_probe(Probe(3, "rect", ((cx - dx, cy - 1.5 * dy), (cx, cy - 0.5 * dy)), color_for(2)))
    st.update_view(field=args.field, configuration="deformed" if args.deformed else "reference")
    win.tabs.setCurrentWidget(win.chart)
    wait(app, lambda: True)
    win.grab().save(str(out / "3_strain_probes.png"))
    st.update_view(configuration="deformed")
    wait(app, lambda: True)
    win.grab().save(str(out / "4_deformed.png"))
    print(f"screenshots in {out}")
    log = win.logview.toPlainText()
    (out / "log.txt").write_text(log, encoding="utf-8")
    win.close()


if __name__ == "__main__":
    main()
