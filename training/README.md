# Training code and data generator

The code that trained the four released models, RAFTcorr-s1, -s2, -s4 and -s8, and the data
generator that draws their training pairs. It is not part of the `raftcorr` package: run it from a
clone of this repository.

| File | Content |
|---|---|
| `train_v2.py` | Trains one model from scratch with fresh generator samples at every step |
| `fields_v2.py`, `gpu_synth.py` | The data generator: procedural speckle, deformation, occlusion and noise, rendered on the GPU, with the exact displacement as the label |
| `generate_samples.py` | Writes a few generator pairs and their exact displacement, without training |
| `check_parity.py` | Checks that this network gives the same displacement as the application |
| `raft_core/` | The network: one RAFT for every downsampling factor `s` (derived from RAFT, see `raft_core/LICENSE-RAFT`) |
| `corr_otf.py` | The on-the-fly correlation, identical to the stored all-pairs volume, with memory linear in the image size |
| `raft_ckpt.py`, `losses.py`, `common.py` | Optional gradient checkpointing, the sequence loss, model loading |
| `slurm/train_v2.slurm` | An example job for a cluster |

## Requirements

A CUDA GPU, Python 3.10 or newer, [PyTorch](https://pytorch.org/get-started/locally/) with CUDA,
NumPy and Pillow (`pip install numpy pillow`; add `matplotlib` for `generate_samples.py --figure`).
`check_parity.py` also needs the application: `pip install raftcorr`.

## The DIV2K images

The smooth part of the deformation fields is made from natural images of the DIV2K data set
(E. Agustsson and R. Timofte, "NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and
Study", CVPR Workshops 2017). DIV2K is not redistributed here. Download `DIV2K_train_HR.zip` (800 PNG
images) from <https://data.vision.ee.ethz.ch/cvl/DIV2K/>, unzip it, and pass the folder with
`--field-dir`.

## Training the four models

The released models were trained from scratch with these settings, one GPU each:

| Model | `--s` | `--crop` (px) | Batch | Steps | Learning rate | GRU iterations | Seed |
|---|---|---|---|---|---|---|---|
| RAFTcorr-s1 | 1 | 128 | 8 | 60,000 | 2e-4 | 12 | 0 |
| RAFTcorr-s2 | 2 | 256 | 8 | 60,000 | 2e-4 | 12 | 0 |
| RAFTcorr-s4 | 4 | 512 | 8 | 60,000 | 2e-4 | 12 | 0 |
| RAFTcorr-s8 | 8 | 512 | 8 | 60,000 | 2e-4 | 12 | 0 |

All four also used `--min-edge 0.125 --p-detail 1.0 --otf auto` and the defaults of every other
option (AdamW with weight decay 1e-5, one-cycle schedule with 5 % warm-up, gradient clipping at 1,
mixed precision, sequence-loss weight 0.8). With `--otf auto`, all four used the stored all-pairs
correlation volume at these crops. From the repository root:

```bash
python training/train_v2.py --name v2_s1c128_seed0 --s 1 --crop 128 --batch 8 --total-iters 60000 --lr 2e-4 --gru-iters 12 --seed 0 --otf auto --min-edge 0.125 --p-detail 1.0 --field-dir path/to/DIV2K_train_HR
python training/train_v2.py --name v2_s2c256_seed0 --s 2 --crop 256 --batch 8 --total-iters 60000 --lr 2e-4 --gru-iters 12 --seed 0 --otf auto --min-edge 0.125 --p-detail 1.0 --field-dir path/to/DIV2K_train_HR
python training/train_v2.py --name v2_s4c512_seed0 --s 4 --crop 512 --batch 8 --total-iters 60000 --lr 2e-4 --gru-iters 12 --seed 0 --otf auto --min-edge 0.125 --p-detail 1.0 --field-dir path/to/DIV2K_train_HR
python training/train_v2.py --name v2_s8c512_seed0 --s 8 --crop 512 --batch 8 --total-iters 60000 --lr 2e-4 --gru-iters 12 --seed 0 --otf auto --min-edge 0.125 --p-detail 1.0 --field-dir path/to/DIV2K_train_HR
```

The models were trained on an NVIDIA GH200 (96 GB). On a smaller GPU:

- `--otf yes` computes the correlation on the fly instead of storing the all-pairs volume. The
  values are the same up to rounding and the memory grows with the number of pixels instead of with its square; this
  matters most for s1 and s2, whose volume is the largest part of the memory.
- `--batch 4 --accum 2` (or `--batch 2 --accum 4`, ...) keeps 8 samples per step with less memory.
- `--grad-ckpt` recomputes each GRU iteration in the backward pass: less memory, slower.
- On Windows, `--max-mem-gb` caps the memory PyTorch may use (a little below the GPU's), so that a
  run that does not fit stops with an out-of-memory error instead of spilling into system memory
  and slowing down to a crawl.

These options change the order of the floating-point operations, so the weights will not be
bit-identical to the released ones.

Each run writes `training/runs/<name>/`: `args.json` (the options), `log.csv` (training loss and the
validation error every 1000 steps, overall and by displacement octave, sample type and occlusion),
and `ckpt/` with `best.pth` (lowest validation error), `latest.pth` (everything needed to resume) and
a copy every 5000 steps. Running the same command again resumes from `latest.pth` (keep
`--total-iters`: the learning-rate schedule is fixed when a run starts); `--fresh` starts over. The validation set is generated once, at a fixed seed, and cached in `training/cache/`.

A checkpoint can be used directly by the application's command line and Python library:
`raftcorr run --model training/runs/<name>/ckpt/best.pth ...` or
`raftcorr.track(..., model="training/runs/<name>/ckpt/best.pth")`.

## Generating samples without training

```bash
python training/generate_samples.py --field-dir path/to/DIV2K_train_HR --s 4 --n 4 --seed 0 --out samples --figure
```

writes `samples/sample_<k>.npz` with the reference and deformed images (`ref`, `dfm`, uint8), the
exact displacement `gt` (float32, 2 x H x W: channel 0 = u, +x to the right; channel 1 = v, +y down;
in px at every reference pixel), the mask `valid` of the pixels the loss uses, and the sample `type`;
the two images as PNG; and, with `--figure`, `samples.png`. A given seed always gives the same
samples on the same GPU and PyTorch version. `--s` sets the displacement range of the variant (up
to 32 s px) and the default size of the samples (the training crop).

The generator draws each sample as a translation, a rotation, a smooth deformation made from a
blurred natural image, or a combination of two or three of them, with a peak displacement spread
evenly over octaves up to the range of the variant. The deformed image is rendered through the
exact inverse of the deformation, so the label is the true displacement of every reference pixel.

## Checking the network

```bash
python training/check_parity.py
```

loads each released model with this folder's network and with the application, runs both on the
same synthetic pairs with 20 iterations, and prints the largest difference of (u, v). The two perform
the same operations, and the difference is exactly 0.

## Conventions

Image 1 is the reference, image 2 the deformed image. The displacement is reference to deformed, u
positive to the right and v positive downward, in px, given at every reference pixel. The network
takes grey levels from 0 to 255 repeated over three channels; images whose sides are not multiples
of 8 are padded by repeating the edge.

## License

The University of Texas at Austin Research License (`LICENSE.md` at the repository root). The
network in `raft_core/` is derived from RAFT (Z. Teed and J. Deng, ECCV 2020), BSD 3-Clause License,
`raft_core/LICENSE-RAFT`.
