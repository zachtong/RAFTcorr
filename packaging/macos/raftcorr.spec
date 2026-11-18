# -*- mode: python -*-
# PyInstaller spec of RAFTcorr.app: macOS on Apple silicon, self-contained, PyTorch on the CPU
# (a Mac has no CUDA). Built by packaging/build_macos.py in a clean virtual environment.

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

SPEC_DIR = Path(SPECPATH)
ROOT = SPEC_DIR.parents[1]
_version = {}
exec((ROOT / "src" / "raftcorr" / "_version.py").read_text(encoding="utf-8"), _version)
VERSION = _version["__version__"]
ICON = ROOT / "src" / "raftcorr" / "gui" / "assets" / "icon" / "RAFTcorr.icns"

# Translations (.qm), icons and SVG arrows of the GUI; not the translators' sources (.ts).
datas = collect_data_files("raftcorr", excludes=["**/*.ts", "**/__pycache__/**"])
compiled = [d for d in datas if d[0].endswith(".qm")]
if len(compiled) != 7:
    raise SystemExit(f"expected 7 compiled translations, found {len(compiled)}: run scripts/i18n.py compile")

hiddenimports = collect_submodules("raftcorr") + [
    "PySide6.QtSvg",
    "matplotlib.backends.backend_agg",
    "matplotlib.backends.backend_qtagg",
    "h5py",
    "imageio_ffmpeg",
    "tifffile",
]

a = Analysis(
    [str(SPEC_DIR / "raftcorr_launcher.py")],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "PyQt5", "PyQt6", "IPython", "jupyter", "pytest", "tensorboard",
              "torch.utils.tensorboard", "triton"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RAFTcorr",
    console=False,               # run from Terminal it still has a stdout (CI reads the self-test)
    upx=False,
    argv_emulation=False,
    target_arch="arm64",
    codesign_identity=None,          # ad-hoc signature, as pyALDIC's
    entitlements_file=None,
    icon=str(ICON),
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="RAFTcorr")
app = BUNDLE(
    coll,
    name="RAFTcorr.app",
    icon=str(ICON),
    bundle_identifier="edu.utexas.raftcorr",
    version=VERSION,
    info_plist={
        "CFBundleName": "RAFTcorr",
        "CFBundleDisplayName": "RAFTcorr",
        "CFBundleShortVersionString": VERSION,
        "CFBundleVersion": VERSION,
        "LSMinimumSystemVersion": "14.0",
        "NSHighResolutionCapable": True,
        "NSRequiresAquaSystemAppearance": False,
    },
)
