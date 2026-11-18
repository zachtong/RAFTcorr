"""Build the Windows installer, dist-installers/RAFTcorr-Windows-Setup.exe.

Stages, under build/windows/stage:

* ``python/``  a relocatable CPython 3.11 (python-build-standalone, the build uv uses), which the
  installed RAFTcorr runs on;
* ``wheels/``  the RAFTcorr wheel of this checkout;
* ``setup_runtime.py``, ``bin/raftcorr.cmd``, ``RAFTcorr.ico``, ``LICENSE.txt``;

then compiles packaging/windows/raftcorr.iss with Inno Setup. PyTorch and the other dependencies
are installed when the installer runs (see setup_runtime.py).

    python packaging/build_windows.py [--iscc PATH] [--python-archive URL_OR_FILE]

Needs pip and Inno Setup 6 (ISCC.exe); the GitHub Actions workflow build-installers.yml runs it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "build" / "windows" / "stage"
OUT = ROOT / "dist-installers"
PBS_API = "https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest"
PBS_ASSET = re.compile(r"cpython-3\.11\.\d+\+\d+-x86_64-pc-windows-msvc-install_only\.tar\.gz$")
DEFAULT_ISCC = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Inno Setup 6" / "ISCC.exe"


def version() -> str:
    text = (ROOT / "src" / "raftcorr" / "_version.py").read_text(encoding="utf-8")
    return re.search(r'__version__\s*=\s*"([^"]+)"', text).group(1)


def numeric(ver: str) -> str:
    """``2.0.0.dev0`` -> ``2.0.0``: Windows version info takes numbers only."""
    return ".".join(re.findall(r"\d+", ver)[:3])


def _get(url: str) -> bytes:
    headers = {"User-Agent": "raftcorr-build"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token and "api.github.com" in url:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300) as resp:
        return resp.read()


def python_archive_url() -> str:
    release = json.loads(_get(PBS_API))
    for asset in release["assets"]:
        if PBS_ASSET.search(asset["name"]):
            return asset["browser_download_url"]
    raise SystemExit(f"no CPython 3.11 Windows install_only archive in {release['tag_name']}")


def stage_python(source: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "python.tar.gz"
        if Path(source).is_file():
            shutil.copy2(source, archive)
        else:
            print(f"downloading {source}")
            archive.write_bytes(_get(source))
        with tarfile.open(archive) as tar:
            tar.extractall(STAGE, filter="data")       # the archive holds python/...
    for exe in ("python.exe", "pythonw.exe"):
        if not (STAGE / "python" / exe).is_file():
            raise SystemExit(f"the Python archive has no python/{exe}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iscc", default=str(DEFAULT_ISCC))
    ap.add_argument("--python-archive", help="python-build-standalone archive (URL or file); default: latest")
    args = ap.parse_args()
    ver = version()
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "bin").mkdir(parents=True)
    OUT.mkdir(exist_ok=True)

    subprocess.run([sys.executable, "-m", "pip", "wheel", str(ROOT), "--no-deps", "-w", str(STAGE / "wheels")],
                   check=True)
    stage_python(args.python_archive or python_archive_url())
    win = ROOT / "packaging" / "windows"
    shutil.copy2(win / "setup_runtime.py", STAGE / "setup_runtime.py")
    shutil.copy2(win / "raftcorr.cmd", STAGE / "bin" / "raftcorr.cmd")
    shutil.copy2(ROOT / "src" / "raftcorr" / "gui" / "assets" / "icon" / "RAFTcorr.ico", STAGE / "RAFTcorr.ico")
    shutil.copy2(ROOT / "LICENSE.md", STAGE / "LICENSE.txt")

    subprocess.run([args.iscc, f"/DAppVersion={ver}", f"/DNumericVersion={numeric(ver)}", f"/DStageDir={STAGE}",
                    f"/DOutputDir={OUT}", str(win / "raftcorr.iss")], check=True)
    setup = OUT / "RAFTcorr-Windows-Setup.exe"
    print(f"built {setup} ({setup.stat().st_size / 1e6:.1f} MB), RAFTcorr {ver}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
