"""Install RAFTcorr's Python packages into the installer's private Python (Windows).

Run by the Setup program right after it has copied its files, and again by the Start menu entry
"Repair RAFTcorr". The installer is small because PyTorch is not inside it: the CUDA build alone
is larger than a release asset may be, and which build fits depends on the computer. This script
chooses it the way install.bat does:

1. the NVIDIA driver's CUDA version (``nvidia-smi``) picks the PyTorch wheel: CUDA 12.8 for a
   driver that supports 12.8 or later (RTX 50 series included), 12.6, 11.8, or the CPU build when
   there is no NVIDIA GPU;
2. PyTorch from the PyTorch index, then the RAFTcorr wheel shipped with the installer, whose other
   dependencies come from PyPI;
3. the four model files from the release (not fatal: the application also fetches a model the
   first time it is used).

Everything is written to ``install.log`` in the installation folder; ``runtime.ok`` marks success.

    python setup_runtime.py --app-dir "C:\\...\\RAFTcorr" [--cpu] [--no-models]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import shutil
import subprocess
import sys
from pathlib import Path

TORCH_INDEX = {
    "cu128": "https://download.pytorch.org/whl/cu128",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu118": "https://download.pytorch.org/whl/cu118",
    "cpu": "https://download.pytorch.org/whl/cpu",
}


class _Tee:
    """Print to the console window and append to the log."""

    def __init__(self, log: Path) -> None:
        self._fh = open(log, "a", encoding="utf-8")

    def __call__(self, text: str) -> None:
        print(text, flush=True)
        self._fh.write(text + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def parse_smi(output: str | None) -> tuple[str, str]:
    """``(build, reason)`` from the output of ``nvidia-smi`` (None: no NVIDIA driver).

    The header says "CUDA Version: 12.4" on older drivers and "CUDA UMD Version: 13.3" on newer ones.
    """
    if output is None:
        return "cpu", "no NVIDIA driver found: CPU build (RAFTcorr runs, but slowly)"
    m = re.search(r"CUDA[A-Za-z ]*Version:\s*(\d+)\.(\d+)", output)
    if not m:
        return "cpu", "nvidia-smi reports no CUDA version: CPU build"
    major, minor = int(m.group(1)), int(m.group(2))
    if re.search(r"RTX\s*50", output) or major >= 13 or (major == 12 and minor >= 8):
        return "cu128", f"driver supports CUDA {major}.{minor}: PyTorch with CUDA 12.8"
    if major == 12:
        return "cu126", f"driver supports CUDA {major}.{minor}: PyTorch with CUDA 12.6"
    if major == 11:
        return "cu118", f"driver supports CUDA {major}.{minor}: PyTorch with CUDA 11.8"
    return "cpu", f"driver supports only CUDA {major}.{minor}: CPU build (update the NVIDIA driver for the GPU)"


def torch_build(force_cpu: bool = False) -> tuple[str, str]:
    """``(build, reason)``: which PyTorch wheel this computer should get."""
    if force_cpu:
        return "cpu", "CPU build requested"
    smi = shutil.which("nvidia-smi") or r"C:\Windows\System32\nvidia-smi.exe"
    try:
        out = subprocess.run([smi], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        out = None
    return parse_smi(out)


def _run(say: _Tee, args: list[str]) -> int:
    say("$ " + " ".join(args))
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    assert proc.stdout is not None
    for line in proc.stdout:
        say(line.rstrip())
    return proc.wait()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--app-dir", help="the installation folder (required unless --detect)")
    ap.add_argument("--cpu", action="store_true", help="install the CPU build of PyTorch")
    ap.add_argument("--no-models", action="store_true", help="do not download the model files")
    ap.add_argument("--detect", action="store_true",
                    help="print '<build>|<reason>' (cu128, cu126, cu118 or cpu) and exit; used by install.bat")
    args = ap.parse_args(argv)
    if args.detect:
        build, reason = torch_build(args.cpu)
        print(f"{build}|{reason}")
        return 0
    if not args.app_dir:
        ap.error("--app-dir is required")
    app = Path(args.app_dir)
    marker = app / "runtime.ok"
    marker.unlink(missing_ok=True)
    say = _Tee(app / "install.log")
    py = sys.executable
    try:
        say(f"RAFTcorr setup, {_dt.datetime.now():%Y-%m-%d %H:%M:%S}, Python {sys.version.split()[0]}")
        wheels = sorted((app / "wheels").glob("raftcorr-*.whl"))
        if not wheels:
            say(f"ERROR: no RAFTcorr wheel in {app / 'wheels'}")
            return 2
        if _run(say, [py, "-m", "pip", "--version"]) != 0 and _run(say, [py, "-m", "ensurepip", "--upgrade"]) != 0:
            say("ERROR: pip is not available")
            return 3
        build, reason = torch_build(args.cpu)
        say(f"[1/3] PyTorch: {reason}")
        pip = [py, "-m", "pip", "install", "--upgrade", "--no-cache-dir", "--no-warn-script-location",
               "--disable-pip-version-check"]
        if _run(say, pip + ["torch", "--index-url", TORCH_INDEX[build]]) != 0:
            say("ERROR: PyTorch could not be installed (check the internet connection)")
            return 4
        say(f"[2/3] RAFTcorr {wheels[-1].name} and its dependencies")
        if _run(say, pip + [str(wheels[-1])]) != 0:
            say("ERROR: RAFTcorr could not be installed (check the internet connection)")
            return 5
        if args.no_models:
            say("[3/3] model files: skipped")
        else:
            say("[3/3] model files")
            if _run(say, [py, "-m", "raftcorr", "models", "--download"]) != 0:
                say("WARNING: some model files were not downloaded; RAFTcorr fetches a model when it is first used")
        _run(say, [py, "-m", "raftcorr", "info"])
        marker.write_text(f"{wheels[-1].name}\ntorch {build}\n", encoding="utf-8")
        say("Done.")
        return 0
    finally:
        say.close()


if __name__ == "__main__":
    sys.exit(main())
