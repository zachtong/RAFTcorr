"""Build the macOS disk image, dist-installers/RAFTcorr-macOS.dmg (Apple silicon).

Run in a clean virtual environment that holds PyTorch (the macOS wheel from PyPI is the CPU/MPS
build), RAFTcorr with its dependencies and PyInstaller (packaging/requirements-build-macos.txt):

    python packaging/build_macos.py

PyInstaller writes build/macos/dist/RAFTcorr.app; the disk image holds the app and a link to
/Applications, so installing is a drag. The model files are not inside: RAFTcorr downloads a
model the first time it is used (or ``RAFTcorr models --download``).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "build" / "macos"
OUT = ROOT / "dist-installers"
MAX_ASSET_BYTES = 2 * 1024**3          # GitHub's limit for one release asset


def main() -> int:
    if sys.platform != "darwin":
        raise SystemExit("the disk image is built on macOS")
    if WORK.exists():
        shutil.rmtree(WORK)
    OUT.mkdir(exist_ok=True)
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
                    "--distpath", str(WORK / "dist"), "--workpath", str(WORK / "work"),
                    str(ROOT / "packaging" / "macos" / "raftcorr.spec")], check=True)
    app = WORK / "dist" / "RAFTcorr.app"
    stage = WORK / "dmg"
    stage.mkdir()
    shutil.copytree(app, stage / "RAFTcorr.app", symlinks=True)
    (stage / "Applications").symlink_to("/Applications")
    dmg = OUT / "RAFTcorr-macOS.dmg"
    dmg.unlink(missing_ok=True)
    subprocess.run(["hdiutil", "create", "-volname", "RAFTcorr", "-srcfolder", str(stage), "-ov",
                    "-format", "UDZO", str(dmg)], check=True)
    size = dmg.stat().st_size
    print(f"built {dmg} ({size / 1e6:.0f} MB)")
    if size > MAX_ASSET_BYTES:
        raise SystemExit("the disk image is larger than a GitHub release asset may be (2 GiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
