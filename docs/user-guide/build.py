"""Build the user guide and the quick guide (pdflatex) into docs/RAFTcorr_User_Guide.pdf and
docs/RAFTcorr_Quick_Guide.pdf.

    python docs/user-guide/build.py

Needs a LaTeX distribution (MiKTeX, TeX Live or MacTeX). The version comes from
src/raftcorr/_version.py; the figures in figures/ come from scripts/media/screenshots.py.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AUX = (".aux", ".log", ".out", ".toc", ".lof", ".lot")
OUTPUT = {"user-guide": "RAFTcorr_User_Guide.pdf", "quick-guide": "RAFTcorr_Quick_Guide.pdf"}


def version() -> str:
    text = (ROOT / "src" / "raftcorr" / "_version.py").read_text(encoding="utf-8")
    return re.search(r'__version__\s*=\s*"([^"]+)"', text).group(1)


def build(stem: str, passes: int, ver: str) -> Path:
    for _ in range(passes):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{stem}.tex"],
                           cwd=HERE, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            sys.stdout.write(r.stdout[-4000:])
            raise SystemExit(f"pdflatex failed on {stem}.tex")
    log = (HERE / f"{stem}.log").read_text(encoding="utf-8", errors="replace")
    for line in log.splitlines():
        if line.startswith(("Overfull", "LaTeX Warning: Reference", "LaTeX Warning: Citation")) or "undefined" in line:
            print(f"  {stem}: {line}")
    # Stable names, so that links from the README and the website keep working; the version is on
    # the title line of each document.
    out = ROOT / "docs" / OUTPUT[stem]
    shutil.copyfile(HERE / f"{stem}.pdf", out)
    for ext in AUX + (".pdf",):
        (HERE / f"{stem}{ext}").unlink(missing_ok=True)
    print(f"wrote {out.relative_to(ROOT)}")
    return out


def main() -> None:
    ver = version()
    (HERE / "version.tex").write_text(f"\\newcommand{{\\raftversion}}{{{ver}}}\n", encoding="utf-8")
    build("user-guide", 3, ver)
    build("quick-guide", 2, ver)


if __name__ == "__main__":
    main()
