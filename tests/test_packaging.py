"""The installers' own logic that runs without building one: the PyTorch build chosen for a GPU,
the GUI self-test an installation runs, and what the wheel declares."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _setup_runtime():
    path = ROOT / "packaging" / "windows" / "setup_runtime.py"
    spec = importlib.util.spec_from_file_location("setup_runtime", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OLD_HEADER = "| NVIDIA-SMI 551.86     Driver Version: 551.86     CUDA Version: 12.4     |\n| NVIDIA GeForce RTX 4090 |"
NEW_HEADER = "| NVIDIA-SMI 610.60     KMD Version: 610.60     CUDA UMD Version: 13.3     |\n| NVIDIA GeForce RTX 5090 |"


@pytest.mark.parametrize(("output", "build"), [
    (OLD_HEADER, "cu126"),                                       # "CUDA Version:" (older drivers)
    (NEW_HEADER, "cu128"),                                       # "CUDA UMD Version:" (newer drivers)
    ("CUDA Version: 12.8 |", "cu128"),
    ("CUDA Version: 11.7 |", "cu118"),
    ("CUDA Version: 12.2 | NVIDIA GeForce RTX 5070", "cu128"),   # RTX 50 needs CUDA 12.8 kernels
    ("CUDA Version: 10.2 |", "cpu"),
    ("NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.", "cpu"),
    (None, "cpu"),                                               # no nvidia-smi at all
])
def test_the_pytorch_build_follows_the_driver(output, build):
    assert _setup_runtime().parse_smi(output)[0] == build


def test_every_build_has_an_index():
    mod = _setup_runtime()
    for output in (OLD_HEADER, NEW_HEADER, "CUDA Version: 11.8", None):
        assert mod.TORCH_INDEX[mod.parse_smi(output)[0]].startswith("https://download.pytorch.org/whl/")


def test_cpu_can_be_forced():
    assert _setup_runtime().torch_build(force_cpu=True)[0] == "cpu"


def test_setup_runtime_needs_an_app_dir(capsys):
    with pytest.raises(SystemExit):
        _setup_runtime().main([])


def test_setup_runtime_reports_a_missing_wheel(tmp_path):
    assert _setup_runtime().main(["--app-dir", str(tmp_path), "--no-models"]) == 2
    assert "no RAFTcorr wheel" in (tmp_path / "install.log").read_text(encoding="utf-8")
    assert not (tmp_path / "runtime.ok").exists()


def test_gui_selftest_opens_the_main_window(qapp):
    from raftcorr.selftest import gui_selftest

    ok, text = gui_selftest()
    assert ok, text
    assert "main window" in text and "translations" in text


def test_cli_selftest_gui_without_models(qapp, capsys):
    from raftcorr.cli.main import main

    assert main(["selftest", "--gui", "--no-models"]) == 0
    assert "GUI self-test: ok" in capsys.readouterr().out


def test_one_install_brings_the_whole_application():
    tomllib = pytest.importorskip("tomllib")          # Python 3.11+
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    names = {d.split(">")[0].split("=")[0].split("[")[0].strip().lower() for d in project["dependencies"]}
    assert {"torch", "pyside6-essentials", "h5py", "imageio", "imageio-ffmpeg"} <= names
    assert "pyside6" not in names        # the full package adds WebEngine and the Addons, unused
    assert project["scripts"]["raftcorr"] == "raftcorr.cli.main:main"
    assert project["optional-dependencies"]["all"] == []       # old instructions keep working


def test_every_qt_module_used_is_in_pyside6_essentials():
    """pip installs PySide6-Essentials only: a module from the Addons would fail on import."""
    import importlib.metadata as md
    import re

    try:
        files = md.files("PySide6-Essentials") or []
    except md.PackageNotFoundError:
        pytest.skip("PySide6-Essentials is not installed as its own distribution here")
    shipped = {m.group(1) for f in files if (m := re.match(r"PySide6/(Qt\w+)\.(?:pyd|abi3\.so|so)$", f.as_posix()))}
    assert {"QtCore", "QtGui", "QtWidgets"} <= shipped, sorted(shipped)
    used = {m for p in (ROOT / "src").rglob("*.py")
            for m in re.findall(r"PySide6\.(Qt\w+)", p.read_text(encoding="utf-8"))}
    assert used and used <= shipped, f"not in PySide6-Essentials: {sorted(used - shipped)}"


@pytest.mark.skipif(sys.platform != "win32", reason="the installer script is for Windows")
def test_installer_registers_the_command_the_application_expects():
    """The .iss writes the same ProgID and command raftcorr.gui.file_association uses."""
    from raftcorr.gui import file_association as fa

    iss = (ROOT / "packaging" / "windows" / "raftcorr.iss").read_text(encoding="utf-8")
    assert f'Software\\Classes\\{fa.EXT}"' in iss and f'ValueData: "{fa.PROGID}"' in iss
    assert '"""{app}\\python\\pythonw.exe"" -m raftcorr gui ""%1"""' in iss
