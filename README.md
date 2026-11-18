# RAFTcorr

**Learned digital image correlation: a desktop application, a command line and a Python library.**

RAFTcorr measures full-field displacement and strain from image sequences with the RAFTcorr
networks: learned correlation models trained on speckle images with exact Lagrangian labels.
Four released models share one architecture and differ in the encoder downsampling factor `s`:

| Model | `s` | Nominal reach (`32 s` px) | Use it for |
|---|---|---|---|
| RAFTcorr-s1 | 1 | 32 | small displacements, finest detail (strain concentrations) |
| RAFTcorr-s2 | 2 | 64 | fine detail, several times faster than s1 |
| RAFTcorr-s4 | 4 | 128 | a balance of detail and reach (the default) |
| RAFTcorr-s8 | 8 | 256 | large motion and rotation, or no prior knowledge |

A smaller `s` resolves shorter wavelengths; a larger `s` reaches further. The **Suggest a model**
button measures the motion of your sequence with s8 and picks the finest model that reaches it.

## Install

Windows: run `install.bat` (creates the conda environment `raftcorr-app`, installs the PyTorch
build that matches your NVIDIA driver, then RAFTcorr), put the four model files
`RAFTcorr-s1.pth` ... `RAFTcorr-s8.pth` in `models/`, and start with `run.bat`.

> TODO: where to download the model files (to be published with the release).

Any platform:

```bash
conda env create -f environment.yml
conda activate raftcorr-app
pip install torch --index-url https://download.pytorch.org/whl/cu126   # match your GPU, or /cpu
pip install -e ".[all]"
raftcorr models      # where the weights are looked for
raftcorr selftest    # every installed model on a synthetic pair
raftcorr             # start the application
```

The weights are found in, in order: a path you give, `$RAFTCORR_WEIGHTS`, `models/` of a source
checkout, and the per-user folder (`%LOCALAPPDATA%\RAFTcorr\models` on Windows,
`~/.raftcorr/models` elsewhere). Every file is checked against its SHA-256.

## The application

1. **Images**: open a folder; the first image in natural order is the reference (frame 1 in the
   application, index 0 in the arrays of exported data files). 8- to 16-bit grey or colour images
   (PNG, TIFF, BMP, JPEG).
2. **Region of interest**: rectangle, circle, polygon and brush, each adding or cutting; undo and
   redo; import mask images. A mask on a later frame excludes the regions it leaves out (a crack,
   a hole) and is the region of interest whenever that frame is a reference.
3. **Model**: s1, s2, s4 or s8, or let RAFTcorr suggest one.
4. **Tracking**: accumulative (every frame against the reference image; the default, and the most
   accurate whenever the motion stays within the model's reach), incremental, every N frames, or
   custom reference frames. 20 iterations and no smoothing by default.
5. **Run**: large regions are tiled automatically to fit the GPU memory.

Results are shown on the reference image (Lagrangian) or on the current image (deformed
configuration), with any colour map, fixed or automatic ranges, physical units and playback.
The **residual** view shows how well each frame was tracked (the grey-level mismatch after
warping), and RAFTcorr warns when it jumps. **Strain** (strain window) comes from a virtual strain
gauge (default: Green-Lagrange, unweighted plane fit, 41 px gauge), never fitted across a crack.
**Analysis**: points, extensometers, crack-opening gauges, rectangles, circles and polygons, plotted
over time, along a line, as a kymograph, or against the load of a testing machine (stress-strain).
**Export**: HDF5, MATLAB v7.3 or NumPy files with every field and the full provenance; per-frame
CSV tables; PNG / TIFF / PDF / SVG figures; GIF / MP4 animations; an HTML report; probe tables; and
project files (`.raftcorr`) that reopen the whole session.

The interface is available in English, Simplified and Traditional Chinese, Japanese, Korean,
German, French and Spanish (**Settings → Language**).

## Command line

```bash
raftcorr run images/ --roi roi.png --model s4 --strain --out results.h5 --project test.raftcorr
raftcorr run images/ --model s8 --mode incremental --out results.mat
raftcorr suggest images/ --roi roi.png
```

## Python

```python
import raftcorr

series = raftcorr.track("images/", roi="roi.png", model="s4")     # DisplacementSeries
strain = raftcorr.compute_strain(series)                          # default strain settings
u, v = series.u[5], series.v[5]                                   # image 5 (0-based), px, NaN outside the ROI
exx = strain.components["exx"][5]
```

## Conventions

`(u, v)` is the displacement from the reference to the deformed image: `+u` right, `+v` down, in
pixels. Every field is stored in the coordinates of the reference image (Lagrangian); in data
files and in Python, index `k` of a result is image `k` of the sequence, counted from 0. Shear
strain is the tensor component (half the engineering shear).

## License

The University of Texas at Austin Research License (see `LICENSE.md`). The network code is derived
from RAFT (Teed and Deng, ECCV 2020), BSD 3-Clause License: see `src/raftcorr/nn/LICENSE-RAFT`.

## Citing

TODO: the RAFTcorr paper (in preparation).
