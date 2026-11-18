"""Pull a handful of frames from the diagnostic GIFs and save as PNGs so they
can be inspected visually."""

import os
from PIL import Image

OUT = os.path.join(os.path.dirname(__file__), "diagnose_output")

for gif_name in ("anim_reference.gif", "anim_deformed.gif"):
    path = os.path.join(OUT, gif_name)
    gif = Image.open(path)
    stem = os.path.splitext(gif_name)[0]
    for idx in (0, 2, 4, 6, 7):     # span expand + shrink + below-ref
        if idx < gif.n_frames:
            gif.seek(idx)
            out = os.path.join(OUT, f"{stem}_frame{idx}.png")
            gif.convert("RGB").save(out)
            print("Wrote", out)
