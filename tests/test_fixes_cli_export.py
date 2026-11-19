"""Fixes in the command line, the library, the exports and the packaging (no model weights needed).

* ``raftcorr run --strain --project`` stores the strain settings it computed with;
* every frame number shown to a person counts from 1 (suggest report, strain progress, image-size error);
* bad input ends with one line on stderr and exit code 2, internal errors keep their traceback;
* the stop contract of :mod:`raftcorr.callbacks`;
* exported colour bars use the names the windows show;
* the GUI always writes a log file;
* install.bat switches an installed PyTorch build that no longer fits the driver.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import raftcorr.engine.pipeline as pipeline_module
from raftcorr.callbacks import FunctionCallbacks, StopRequested
from raftcorr.cli.main import main
from raftcorr.engine.mapping import Grid
from raftcorr.results import DisplacementSeries, RunInfo
from raftcorr.schedule import FrameSchedule

ROOT = Path(__file__).resolve().parents[1]
H, W = 64, 80


# -- helpers --------------------------------------------------------------------------------------


def _series(n: int, names: tuple[str, ...] | None = None) -> DisplacementSeries:
    """A uniform 1 % stretch on the whole image, every frame computed."""
    grid = Grid(0, 0, W, H, H, W)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = np.stack([0.01 * k * xx for k in range(n)]).astype(np.float32)
    v = np.zeros_like(u)
    info = RunInfo(software_version="test", model_name="fake", model_sha256="ab" * 32, downsample_factor=8,
                   iters=1, corr_mode="volume", device="cpu", config={})
    return DisplacementSeries(grid, np.ones((H, W), bool), u, v, np.ones(n, bool),
                              names or tuple(f"f{k}.png" for k in range(n)), FrameSchedule.accumulative(n), info)


class _FakePipeline:
    """Stands in for the tracking: a model that loads instantly and a synthetic series."""

    def __init__(self, config=None, callbacks=None, loaded=None) -> None:
        self.config = config
        self.loaded = SimpleNamespace(name="fake")

    def run(self, images, roi=None, **_kw) -> DisplacementSeries:
        return _series(len(images), tuple(images.names))


class _Boom(_FakePipeline):
    def run(self, images, roi=None, **_kw):
        raise RuntimeError("internal failure")


@pytest.fixture()
def folder(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    d = tmp_path / "imgs"
    d.mkdir()
    for i in range(3):
        Image.fromarray(rng.integers(0, 255, (H, W), dtype=np.uint8)).save(d / f"img{i + 1}.png")
    return d


@pytest.fixture()
def fake_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_module, "Pipeline", _FakePipeline)


# -- 1. the project keeps the strain settings it was computed with --------------------------------


def test_run_project_stores_the_strain_settings_used(folder, tmp_path, fake_tracking):
    from raftcorr.io.project import load_project

    out = tmp_path / "p.raftcorr"
    assert main(["run", str(folder), "--strain", "--gauge", "21", "--measure", "infinitesimal",
                 "--edge-trim", "0.3", "--project", str(out), "--quiet"]) == 0
    prj = load_project(out)
    assert prj.strain is not None and prj.strain.settings.gauge_px == 21
    cfg = prj.strain_config
    assert (cfg.gauge_px, cfg.measure, cfg.edge_trim_alpha) == (21, "infinitesimal", 0.3)


# -- 2. frame numbers count from 1 ----------------------------------------------------------------


def test_suggest_report_numbers_frames_from_one(folder, fake_tracking, monkeypatch, capsys):
    from raftcorr.models import advisor

    def fake_survey(_pipe, _seq, _mask, pairs, _cb):
        assert pairs == [(0, 1), (0, 2)]                       # the advisor's data stay 0-based
        return advisor.Advice("s2", 30.0, ((1, 10.0), (2, 30.0)), "0-based reason text")

    monkeypatch.setattr(advisor, "survey", fake_survey)
    assert main(["suggest", str(folder)]) == 0
    out = capsys.readouterr().out
    assert "RAFTcorr-s2" in out
    assert "frames 2, 3" in out and "0-based reason text" not in out


def test_strain_progress_numbers_frames_from_one():
    from raftcorr.post.strain import compute_strain

    messages: list[str] = []
    compute_strain(_series(3), callbacks=FunctionCallbacks(progress=lambda _f, m: messages.append(m)))
    assert messages == ["strain: frame 1 (1/3)", "strain: frame 2 (2/3)", "strain: frame 3 (3/3)"]


def test_index_options_say_they_are_zero_based(capsys):
    with pytest.raises(SystemExit):
        main(["run", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "0-based index" in text


def test_image_size_error_counts_from_one_and_names_the_file(tmp_path):
    from raftcorr.io.images import ImageSequence

    d = tmp_path / "mixed"
    d.mkdir()
    Image.fromarray(np.zeros((H, W), np.uint8)).save(d / "a1.png")
    Image.fromarray(np.zeros((H, W + 2), np.uint8)).save(d / "a2.png")
    with pytest.raises(ValueError) as info:
        ImageSequence.from_folder(d)
    msg = str(info.value)
    assert "frame 0" not in msg
    assert "frame 1" in msg and "a1.png" in msg and "a2.png" in msg and f"{W + 2} x {H}" in msg


# -- 3. bad input: one line on stderr, exit 2 -----------------------------------------------------


def _clean_error(capsys, argv: list[str]) -> str:
    assert main(argv) == 2
    err = capsys.readouterr().err.strip()
    assert err and "\n" not in err and "Traceback" not in err, err
    return err


@pytest.mark.parametrize("extra", [
    ["--keys", "5", "--mode", "custom"],             # out of range for 3 frames
    ["--keys", "a,b", "--mode", "custom"],           # not numbers
    ["--every", "0", "--mode", "every_n"],
    ["--iters", "0"],
    ["--smooth", "-1"],
    ["--config", "missing.json"],
    ["--roi", "missing.png"],
    ["--device", "nonsense"],
])
def test_run_rejects_bad_options_cleanly(folder, fake_tracking, capsys, extra):
    _clean_error(capsys, ["run", str(folder), *extra, "--quiet"])


def test_run_rejects_a_missing_folder_cleanly(tmp_path, fake_tracking, capsys):
    assert "not a folder" in _clean_error(capsys, ["run", str(tmp_path / "nope"), "--quiet"])


def test_run_rejects_a_folder_without_images_cleanly(tmp_path, fake_tracking, capsys):
    _clean_error(capsys, ["run", str(tmp_path), "--quiet"])


def test_run_rejects_a_single_image_cleanly(tmp_path, fake_tracking, capsys):
    Image.fromarray(np.zeros((H, W), np.uint8)).save(tmp_path / "only.png")
    _clean_error(capsys, ["run", str(tmp_path), "--quiet"])


def test_run_rejects_an_empty_roi_cleanly(folder, tmp_path, fake_tracking, capsys):
    roi = tmp_path / "empty.png"
    Image.fromarray(np.zeros((H, W), np.uint8)).save(roi)
    _clean_error(capsys, ["run", str(folder), "--roi", str(roi), "--quiet"])


def test_run_rejects_a_bad_config_file_cleanly(folder, tmp_path, fake_tracking, capsys):
    cfg = tmp_path / "bad.json"
    cfg.write_text("{not json", encoding="utf-8")
    _clean_error(capsys, ["run", str(folder), "--config", str(cfg), "--quiet"])


@pytest.mark.parametrize("model", ["s9", "missing_checkpoint.pth"])
def test_run_rejects_an_unknown_model_cleanly(folder, capsys, model):
    # The real Pipeline: an unknown key or a missing file fails before any download or tracking.
    _clean_error(capsys, ["run", str(folder), "--model", model, "--quiet"])


def test_suggest_rejects_bad_input_cleanly(tmp_path, folder, fake_tracking, capsys):
    _clean_error(capsys, ["suggest", str(tmp_path / "nope")])
    _clean_error(capsys, ["suggest", str(folder), "--device", "nonsense"])


def test_models_download_rejects_an_unknown_key_cleanly(capsys):
    assert "s9" in _clean_error(capsys, ["models", "--download", "s9"])


def test_selftest_rejects_a_bad_device_cleanly(capsys):
    _clean_error(capsys, ["selftest", "--device", "nonsense"])


def test_internal_errors_keep_their_traceback(folder, monkeypatch):
    monkeypatch.setattr(pipeline_module, "Pipeline", _Boom)
    with pytest.raises(RuntimeError, match="internal failure"):
        main(["run", str(folder), "--quiet"])


# -- 4. the stop contract the callbacks docstring states ------------------------------------------


def test_pipeline_run_returns_partial_results_when_stopped(folder):
    from raftcorr.config import RunConfig
    from raftcorr.io.images import ImageSequence

    loaded = SimpleNamespace(name="fake", sha256="0" * 64, downsample_factor=8, corr_mode="volume", device="cpu")
    stop = FunctionCallbacks(should_stop=lambda: True)
    series = pipeline_module.Pipeline(RunConfig(device="cpu"), stop, loaded).run(ImageSequence.from_folder(folder))
    assert series.done.tolist() == [True, False, False]
    assert any("stopped" in note for note in series.info.notes)


def test_compute_strain_raises_when_stopped():
    from raftcorr.post.strain import compute_strain

    with pytest.raises(StopRequested):
        compute_strain(_series(3), callbacks=FunctionCallbacks(should_stop=lambda: True))


def test_callbacks_docstring_states_the_stop_contract():
    import raftcorr.callbacks as callbacks

    doc = " ".join((callbacks.__doc__ or "").split())
    assert "makes the pipeline raise" not in doc
    assert "Pipeline.run" in doc and "StopRequested" in doc


# -- 5. exported colour bars use the names on screen ----------------------------------------------


@pytest.mark.parametrize(("field", "label"), [
    ("u", "U (px)"), ("v", "V (px)"), ("magnitude", "Magnitude (px)"),
    ("exx", "εxx (-)"), ("max_shear", "max shear (-)"), ("rotation", "ω rot (deg)"),
])
def test_exported_colorbar_uses_the_display_name(tmp_path, monkeypatch, field, label):
    import raftcorr.export.figures as figures
    from raftcorr.config import StrainConfig
    from raftcorr.post.strain import compute_strain
    from raftcorr.viz.fields import FieldProvider, Units

    series = _series(2)
    provider = FieldProvider(series, compute_strain(series, StrainConfig(gauge_px=11)))
    seen: list[str] = []
    real = figures.render_figure

    def spy(rgba, style, rng, bar_label, title, **kw):
        seen.append(bar_label)
        return real(rgba, style, rng, bar_label, title, **kw)

    monkeypatch.setattr(figures, "render_figure", spy)
    style = figures.FrameStyle(field=field, background="none")
    figures.export_frames(tmp_path / "img", provider, None, [1], style, Units())
    figures.export_animation(tmp_path / "a.gif", provider, None, [0, 1], style, Units())
    assert seen and set(seen) == {label}
    assert figures.colorbar_label(field, Units()) == label


def test_report_thumbnails_use_the_display_name(tmp_path, monkeypatch):
    import raftcorr.export.report as report
    from raftcorr.viz.fields import FieldProvider

    seen: list[str] = []
    real = report.render_figure

    def spy(rgba, style, rng, bar_label, title, **kw):
        seen.append(bar_label)
        return real(rgba, style, rng, bar_label, title, **kw)

    monkeypatch.setattr(report, "render_figure", spy)
    report.export_html_report(tmp_path / "r.html", FieldProvider(_series(2)))
    assert seen and all(not s.startswith(("u (", "v (", "magnitude (")) for s in seen), seen


# -- 7. the GUI always writes a log file ----------------------------------------------------------


def test_gui_logging_writes_a_rotating_file_when_not_frozen(tmp_path, monkeypatch):
    from logging.handlers import RotatingFileHandler

    from raftcorr.gui import app

    monkeypatch.delattr("sys.frozen", raising=False)
    monkeypatch.setattr(app, "user_data_dir", lambda: tmp_path / "data")
    monkeypatch.setattr(app, "_LOG_FILE", None)
    root = logging.getLogger()
    before = list(root.handlers)
    level = root.level
    try:
        app._configure_logging()
        app._configure_logging()                               # idempotent: one file handler
        added = [h for h in root.handlers if h not in before]
        files = [h for h in added if isinstance(h, RotatingFileHandler)]
        assert len(files) == 1
        logging.getLogger("raftcorr.test").info("hello log")
        files[0].flush()
        log_file = tmp_path / "data" / "logs" / "RAFTcorr.log"
        assert app._LOG_FILE == log_file
        assert "hello log" in log_file.read_text(encoding="utf-8")
    finally:
        for h in [h for h in root.handlers if h not in before]:
            root.removeHandler(h)
            h.close()
        root.setLevel(level)


def test_user_data_dir_docstring_names_no_missing_file():
    from raftcorr.gui import app

    for name in re.findall(r"packaging/[\w/.]+", app.user_data_dir.__doc__ or ""):
        assert (ROOT / name).exists(), name


# -- 8. install.bat switches a PyTorch build that does not fit -----------------------------------


def _setup_runtime():
    import importlib.util

    path = ROOT / "packaging" / "windows" / "setup_runtime.py"
    spec = importlib.util.spec_from_file_location("setup_runtime_fix", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("installed, build, action", [
    (None, "cu128", "install"),            # no PyTorch yet
    ("2.11.0+cu128", "cu128", "keep"),     # re-run: the right build downloads nothing
    ("2.11.0+cpu", "cu128", "replace"),    # CPU build, and the driver now allows CUDA
    ("2.11.0+cu128", "cu126", "replace"),  # driver downgrade
    ("2.11.0", "cpu", "replace"),          # a PyPI wheel without a build tag
])
def test_torch_action_keeps_the_right_build_and_replaces_any_other(installed, build, action):
    assert _setup_runtime().torch_action(installed, build) == action


def test_both_installers_replace_a_mismatched_torch_build():
    """install.bat asks setup_runtime.py, so one rule serves the Setup (and its Repair) and the
    batch file; a wrong build is uninstalled first, because "pip install --upgrade" keeps an
    installed torch of the same or a newer version as "already satisfied"."""
    bat = (ROOT / "install.bat").read_text(encoding="utf-8")
    assert "--torch-action" in bat and "pip uninstall -y torch" in bat
    src = (ROOT / "packaging" / "windows" / "setup_runtime.py").read_text(encoding="utf-8")
    assert "torch_action(" in src and '"uninstall", "-y", "torch"' in src
