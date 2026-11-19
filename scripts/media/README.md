# Documentation pictures and the quick-start video

Every picture in the user guide, the README banner and the quick-start videos are made by driving the
real application: no mock-ups, no screen recorder. The scripts run RAFTcorr on Qt's offscreen platform
at a device-pixel ratio of 2 (nothing appears on screen, the user's own settings are not touched),
perform each step with the same widgets and mouse events a user would, and take pictures of what the
application draws. Rerun them whenever the interface changes.

| Script | Output |
|---|---|
| `screenshots.py` | `docs/user-guide/figures/*.png`, `docs/media/banner.png` |
| `quickstart_video.py` | `docs/media/RAFTcorr_quickstart_<lang>.mp4` (1920 x 1080, 60 fps, about one minute) |
| `readme_gif.py` | `docs/media/RAFTcorr_quickstart.gif` (the README preview, from the English video) |

The demo data is the aluminium plate with two holes in tension (`Al_with_holes_1`, 44 images of
552 x 3008 px), which is not in the repository. Requirements: RAFTcorr installed or this checkout with
its model files, `opencv-python-headless`, and an ffmpeg with libx264 (`imageio-ffmpeg`, a RAFTcorr
dependency, bundles one; or set `FFMPEG`). A GPU makes each run take a couple of minutes.

```bash
python scripts/media/screenshots.py --images path/to/Al_with_holes_1
python scripts/media/quickstart_video.py --images path/to/Al_with_holes_1 --lang en --out docs/media/RAFTcorr_quickstart_en.mp4
python scripts/media/quickstart_video.py --images path/to/Al_with_holes_1 --lang zh_CN --out docs/media/RAFTcorr_quickstart_zh.mp4
python scripts/media/readme_gif.py
python docs/user-guide/build.py          # the PDF guides, with the new figures
```

Files: `app_setup.py` (the off-screen application and the specimen geometry), `compositor.py` (frames:
camera, pointer, captions, cards; 4K master reduced to 1080p), `director.py` (pointer moves, clicks,
menus, drags and time-lapses on the real widgets), `storyboard.py` (caption text, English and
Chinese), `banner.py`. The compositor and director follow the pyALDIC tutorial-video generator.
