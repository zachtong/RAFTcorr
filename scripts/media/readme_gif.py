"""The README preview: docs/media/RAFTcorr_quickstart.gif from the English quick-start video.

    python scripts/media/readme_gif.py [--video docs/media/RAFTcorr_quickstart_en.mp4] [--width 800] [--fps 12]

One palette for the whole clip (ffmpeg palettegen in "diff" mode, rectangle updates), which keeps the
file small: GitHub shows a GIF inline, a video only as a link.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MEDIA = ROOT / "docs" / "media"


def find_ffmpeg() -> str:
    import os

    if os.environ.get("FFMPEG"):
        return os.environ["FFMPEG"]
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", type=Path, default=MEDIA / "RAFTcorr_quickstart_en.mp4")
    ap.add_argument("--out", type=Path, default=MEDIA / "RAFTcorr_quickstart.gif")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--fps", type=int, default=12)
    a = ap.parse_args()
    graph = (f"fps={a.fps},scale={a.width}:-1:flags=lanczos,split[s0][s1];"
             "[s0]palettegen=max_colors=200:stats_mode=diff[p];"
             "[s1][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle")
    cmd = [find_ffmpeg(), "-y", "-loglevel", "error", "-i", str(a.video), "-vf", graph, "-loop", "0", str(a.out)]
    if subprocess.run(cmd).returncode != 0:
        sys.exit("ffmpeg failed")
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
