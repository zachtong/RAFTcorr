"""On-the-fly GPU synthesis of (reference, deformed, flow, valid) speckle training pairs.

A batched GPU port of the earlier CPU generator of the research code (speckle, mixed displacement
fields, occlusion field, warp), so every training iteration can draw FRESH samples
(effectively infinite data, no disk I/O). Convention: image1 = reference, image2 =
deformed, flow = reference->deformed (u, v), +u right, +v down, def[p] = ref[p - d(p)].

Differences from the CPU generator (deliberate, documented):
  * speckle ellipses are rasterized exactly (same jittered-grid / axis-aligned ellipse model) but
    vectorized over a 3x3 neighbour-cell search instead of a Python loop;
  * the inverse warp uses bicubic convolution (grid_sample) instead of a cubic spline;
  * the smooth deformation component is the self-contained ``smooth_random`` field (blurred noise),
    because the natural-image folder of the earlier CPU generator is not available on every machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class SpeckleCfg:
    diameter_range: tuple = (2.0, 8.0)
    density_range: tuple = (0.4, 0.8)
    variation: float = 0.3
    eccentricity_max: float = 0.4
    invert_prob: float = 0.2
    white_speckle: tuple = (0.0, 80.0)
    white_background: tuple = (200.0, 255.0)
    black_speckle: tuple = (200.0, 255.0)
    black_background: tuple = (0.0, 80.0)
    blur_center_range: tuple = (0.1, 1.1)
    blur_band: float = 0.05


@dataclass
class MixCfg:
    """Sample-type probabilities (normalized internally) and per-type parameters."""
    p_translation: float = 0.52
    p_rotation: float = 0.30
    p_smooth: float = 0.18
    p_occlusion: float = 0.0          # extra occlusion fraction (applied on top, as in the earlier CPU generator)
    trans_deform_frac: float = 0.4    # translation samples that also carry a smooth deformation
    smooth_amp: float = 8.0
    smooth_sigma_range: tuple = (15.0, 45.0)
    rot_range_deg: float = 45.0
    rot_translate_frac: float = 0.3
    occ_rot_frac: float = 0.5
    occ_disp_frac_range: tuple = (0.32, 0.96)  # x max_disp (80-240 px at max_disp 250 in the earlier CPU generator)
    occ_patch_frac_range: tuple = (0.5, 0.72)  # x full canvas


@dataclass
class SynthCfg:
    crop: int = 384
    max_disp: float = 192.0
    noise_std_range: tuple = (2.0, 10.0)
    # "exact": the label is the true reference->deformed displacement d(X) on the reference grid and
    #          the image is warped with its inverse map, so image motion and label agree.
    # "legacy": reproduces raftcorr/data_generation: def[p] = ref[p - d(p)] (d read at the DEFORMED
    #          pixel) while d(X) is stored as the label -> label != image motion wherever grad d != 0.
    label_mode: str = "exact"
    # "ellipse": the original regular jittered-ellipse speckle; "diverse": render_speckle_diverse
    speckle_mode: str = "ellipse"
    speckle: SpeckleCfg = field(default_factory=SpeckleCfg)
    mix: MixCfg = field(default_factory=MixCfg)

    @property
    def full(self) -> int:
        # margin >= max_disp so translated sampling never hits the replicate border inside the crop
        f = self.crop + 2 * int(math.ceil(self.max_disp)) + 16
        return int(math.ceil(f / 16) * 16)


def _u(shape, gen, lo, hi, device):
    return lo + (hi - lo) * torch.rand(shape, generator=gen, device=device)


def render_speckle(B, S, scfg: SpeckleCfg, gen, device):
    """Batch of SxS speckle images (float, 0-255, no noise). Same model as speckle.py."""
    D = _u((B,), gen, *scfg.diameter_range, device)
    dens = _u((B,), gen, *scfg.density_range, device)
    sp = D / dens                                            # grid spacing
    ecc = (torch.rand(B, generator=gen, device=device) ** 2) * scfg.eccentricity_max
    r0 = 0.5 * D
    q = torch.sqrt(torch.sqrt(1.0 - ecc * ecc))
    maj, minr = r0 / q, r0 * q

    n_cells = int(math.floor(S / float(sp.min()))) + 3
    lim = 0.5 * scfg.variation * sp
    jit = (2.0 * torch.rand((B, n_cells, n_cells, 2), generator=gen, device=device) - 1.0) \
        * lim[:, None, None, None]
    cols = torch.floor(S / sp)
    xmin = 0.5 * (S - cols * sp)                             # same offset in x and y (square)

    ar = torch.arange(S, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(ar, ar, indexing="ij")           # (S, S)
    spv = sp[:, None, None]
    c0x = torch.round((gx[None] - xmin[:, None, None]) / spv)
    c0y = torch.round((gy[None] - xmin[:, None, None]) / spv)
    bidx = torch.arange(B, device=device)[:, None, None].expand(B, S, S)
    inside = torch.zeros((B, S, S), dtype=torch.bool, device=device)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cx_i = (c0x + dx).clamp(0, n_cells - 1).long()
            cy_i = (c0y + dy).clamp(0, n_cells - 1).long()
            j = jit[bidx, cy_i, cx_i]                         # (B, S, S, 2)
            cx = xmin[:, None, None] + cx_i.float() * spv + j[..., 0]
            cy = xmin[:, None, None] + cy_i.float() * spv + j[..., 1]
            inside |= ((gx[None] - cx) ** 2 / maj[:, None, None] ** 2
                       + (gy[None] - cy) ** 2 / minr[:, None, None] ** 2) <= 1.0

    invert = torch.rand(B, generator=gen, device=device) < scfg.invert_prob
    si = torch.where(invert, _u((B,), gen, *scfg.black_speckle, device),
                     _u((B,), gen, *scfg.white_speckle, device))
    bi = torch.where(invert, _u((B,), gen, *scfg.black_background, device),
                     _u((B,), gen, *scfg.white_background, device))
    img = torch.where(inside, si[:, None, None], bi[:, None, None])

    center = _u((B,), gen, *scfg.blur_center_range, device)
    sigma = (center - scfg.blur_band).clamp(min=0) \
        + 2 * scfg.blur_band * torch.rand(B, generator=gen, device=device)
    img = gaussian_blur_batch(img, sigma, truncate=2.0)
    info = {"diameter": D, "density": dens, "blur_sigma": sigma, "invert": invert}
    return img, info


def _fft_freqs(S, device):
    f = torch.fft.fftfreq(S, device=device)
    return torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2)            # radial frequency (cycles/px)


def _boolean_blobs(B, S, gen, device):
    """Polydisperse random (Poisson) soft disks: 2-4 radius classes, random per-dot amplitude,
    overlaps combined by saturation 1 - exp(-c * sum). Disk kernel applied in Fourier space (jinc)."""
    rho = _fft_freqs(S, device)
    total = torch.zeros((B, S, S), device=device)
    n_cls = int(torch.randint(2, 5, (1,), generator=gen, device=device))
    for _ in range(n_cls):
        r = torch.exp(_u((B,), gen, math.log(0.8), math.log(6.0), device))       # radius (px)
        cover = _u((B,), gen, 0.15, 0.6, device)                                   # area coverage
        p = (cover / (math.pi * r ** 2)).clamp(max=0.5)
        imp = (torch.rand((B, S, S), generator=gen, device=device) < p[:, None, None]).float()
        imp = imp * _u((B, S, S), gen, 0.4, 1.0, device)
        x = 2 * math.pi * r[:, None, None] * rho[None].clamp(min=1e-6)
        jinc = 2 * torch.special.bessel_j1(x) / x
        soft = torch.exp(-2 * math.pi ** 2 * (0.35 * r[:, None, None]) ** 2 * rho[None] ** 2)
        total = total + torch.fft.ifft2(torch.fft.fft2(imp) * jinc * soft).real
    c = _u((B,), gen, 1.0, 3.0, device)[:, None, None]
    return 1.0 - torch.exp(-c * total.clamp(min=0))


def _bandpass_texture(B, S, gen, device):
    """Band-limited noise (difference of Gaussians in Fourier space), then a soft threshold whose
    gain spans continuous gray texture (gain ~1) to nearly binary speckle (gain ~20)."""
    rho = _fft_freqs(S, device)
    d = torch.exp(_u((B,), gen, math.log(2.5), math.log(12.0), device))             # grain size (px)
    s1 = (0.25 * d)[:, None, None]
    s2 = (1.2 * d)[:, None, None]
    H = torch.exp(-2 * math.pi ** 2 * s1 ** 2 * rho[None] ** 2) - \
        torch.exp(-2 * math.pi ** 2 * s2 ** 2 * rho[None] ** 2)
    x = torch.fft.ifft2(torch.fft.fft2(torch.randn((B, S, S), generator=gen, device=device)) * H).real
    x = (x - x.mean(dim=(1, 2), keepdim=True)) / x.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
    gain = torch.exp(_u((B,), gen, math.log(0.7), math.log(20.0), device))[:, None, None]
    bias = _u((B,), gen, -0.8, 0.8, device)[:, None, None]
    return torch.sigmoid(gain * x + bias)


def render_speckle_diverse(B, S, gen, device, stretch=False, aa=False):
    """Broad speckle-appearance distribution (procedural only, no real images):
    family 0 regular jittered ellipses (wider params), 1 polydisperse Poisson blobs, 2 band-pass
    texture (gray to binary), 3 blobs x texture; then random contrast window, inversion, smooth
    illumination non-uniformity, gamma and optical blur. Output float 0-255, noise added later."""
    fam = torch.multinomial(torch.tensor([0.2, 0.35, 0.3, 0.15], device=device), B, replacement=True,
                            generator=gen)
    k = 2 if aa else 1                                                  # aa: render 2x, average down
    wide = SpeckleCfg(diameter_range=(2.0 * k, 12.0 * k), density_range=(0.3, 0.9), variation=0.7,
                      invert_prob=0.0, white_speckle=(0.0, 0.0), white_background=(255.0, 255.0),
                      blur_center_range=(0.0, 0.0), blur_band=0.0)
    m0 = 1.0 - render_speckle(B, S * k, wide, gen, device)[0] / 255.0   # 1 = dot
    if aa:
        m0 = F.avg_pool2d(m0[:, None], 2)[:, 0]
    m1 = _boolean_blobs(B, S, gen, device)
    m2 = _bandpass_texture(B, S, gen, device)
    m3 = (m1 + m2 + _bandpass_texture(B, S, gen, device)) / 3.0     # multi-scale mixture
    sel = fam[:, None, None]
    m = torch.where(sel == 0, m0, torch.where(sel == 1, m1, torch.where(sel == 2, m2, m3)))
    if stretch:                       # texture-contrast floor: map the 1-99 % range of m to [0, 1]
        sub = m[:, ::4, ::4].reshape(B, -1)
        lo_q, hi_q = torch.quantile(sub, 0.01, dim=1), torch.quantile(sub, 0.99, dim=1)
        m = ((m - lo_q[:, None, None]) / (hi_q - lo_q).clamp(min=1e-3)[:, None, None]).clamp(0, 1)

    invert = torch.rand(B, generator=gen, device=device) < 0.3
    m = torch.where(invert[:, None, None], 1.0 - m, m)
    lo = _u((B,), gen, 0.0, 120.0, device)
    span = _u((B,), gen, 40.0, 255.0, device)
    hi = torch.minimum(lo + span, torch.full_like(lo, 255.0))
    img = hi[:, None, None] - (hi - lo)[:, None, None] * m               # dots dark on bright bg
    illum = 1.0 + _u((B,), gen, 0.0, 0.35, device)[:, None, None] * \
        smooth_field(B, S, 1.0, (60.0, 200.0), gen, device)[:, 0]
    img = img * illum
    gamma = torch.exp(_u((B,), gen, math.log(0.7), math.log(1.4), device))[:, None, None]
    img = 255.0 * (img.clamp(0, 255) / 255.0) ** gamma
    sigma = _u((B,), gen, 0.0, 1.5, device)
    img = gaussian_blur_batch(img, sigma.clamp(min=0.05), truncate=2.0)
    return img.clamp(0, 255), {"family": fam, "diameter": torch.zeros(B, device=device),
                               "blur_sigma": sigma, "invert": invert}


def gaussian_blur_batch(img, sigma, truncate=2.0):
    """Separable per-sample Gaussian blur of (B, H, W) with per-sample sigma (reflect padding)."""
    B = img.shape[0]
    rad = max(1, int(math.ceil(truncate * float(sigma.max()))))
    t = torch.arange(-rad, rad + 1, device=img.device, dtype=img.dtype)
    s = sigma.clamp(min=1e-3)[:, None]
    k = torch.exp(-0.5 * (t[None] / s) ** 2)
    k = k * (t[None].abs() <= torch.ceil(truncate * s))     # per-sample truncation, like scipy
    k = k / k.sum(1, keepdim=True)
    x = img[None]                                            # (1, B, H, W): batch as channels
    x = F.pad(x, (rad, rad, 0, 0), mode="reflect")
    x = F.conv2d(x, k[:, None, None, :], groups=B)
    x = F.pad(x, (0, 0, rad, rad), mode="reflect")
    x = F.conv2d(x, k[:, None, :, None], groups=B)
    return x[0]


def smooth_field(B, S, amp, sigma_range, gen, device):
    """Two independent blurred-noise scalar fields per sample, min-max mapped to [-amp, amp]."""
    noise = torch.rand((B, 2, S, S), generator=gen, device=device)
    sig = _u((B,), gen, *sigma_range, device)
    f = torch.fft.fftfreq(S, device=device)
    f2 = f[:, None] ** 2 + f[None, :] ** 2
    H = torch.exp(-2.0 * math.pi ** 2 * sig[:, None, None, None] ** 2 * f2[None, None])
    out = torch.fft.ifft2(torch.fft.fft2(noise) * H).real
    lo = out.amin(dim=(2, 3), keepdim=True)
    hi = out.amax(dim=(2, 3), keepdim=True)
    norm = (out - lo) / (hi - lo).clamp(min=1e-12)
    a = amp if torch.is_tensor(amp) else torch.full((B,), float(amp), device=device)
    return (-1.0 + 2.0 * norm) * a[:, None, None, None]


def sample_field(field, sx, sy, mode="bilinear"):
    """Sample field (B,C,H,W) at pixel coords (sx, sy) (B,H,W), border clamp (bilinear or bicubic)."""
    B, _, H, W = field.shape
    grid = torch.stack([2.0 * sx.clamp(0, W - 1) / (W - 1) - 1.0,
                        2.0 * sy.clamp(0, H - 1) / (H - 1) - 1.0], dim=-1)
    return F.grid_sample(field, grid, mode=mode, padding_mode="border", align_corners=True)


def invert_displacement(d, iters=12):
    """Eulerian inverse map w on the deformed grid such that p - w(p) = X with p = X + d(X),
    i.e. w(p) = d(p - w(p)), by fixed-point iteration (converges for |grad d| < 1)."""
    B, _, H, W = d.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=d.device, dtype=torch.float32),
                            torch.arange(W, device=d.device, dtype=torch.float32), indexing="ij")
    w = d.clone()
    for _ in range(iters):
        w = sample_field(d, xs[None] - w[:, 0], ys[None] - w[:, 1])
    return w


def rotation_inverse_map(xc, yc, th, tx, ty):
    """Exact w(p) for the rigid motion X -> R(th) X + t about the canvas centre:
    X = R^-1 (p - t)  =>  w = p - R^-1 (p - t)."""
    c, s = torch.cos(th)[:, None, None], torch.sin(th)[:, None, None]
    qx, qy = xc[None] - tx[:, None, None], yc[None] - ty[:, None, None]
    return torch.stack([xc[None] - (c * qx + s * qy), yc[None] - (-s * qx + c * qy)], 1)


def warp_inverse(ref, flow):
    """def[p] = ref[p - flow(p)], bicubic, replicate border. ref (B,H,W), flow (B,2,H,W)."""
    B, H, W = ref.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=ref.device, dtype=torch.float32),
                            torch.arange(W, device=ref.device, dtype=torch.float32), indexing="ij")
    sx = (xs[None] - flow[:, 0]).clamp(0, W - 1)
    sy = (ys[None] - flow[:, 1]).clamp(0, H - 1)
    grid = torch.stack([2.0 * sx / (W - 1) - 1.0, 2.0 * sy / (H - 1) - 1.0], dim=-1)
    out = F.grid_sample(ref[:, None], grid, mode="bicubic", padding_mode="border",
                        align_corners=True)
    return out[:, 0]


class SpeckleSynth:
    """Draws batches of training pairs on the GPU. Call ``sample(B)``."""

    TYPES = ("translation", "rotation", "smooth", "occlusion")

    def __init__(self, cfg: SynthCfg, device="cuda", seed=0):
        self.cfg = cfg
        self.device = device
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

    @torch.no_grad()
    def sample(self, B):
        cfg, m, dev, g = self.cfg, self.cfg.mix, self.device, self.gen
        S, C, R = cfg.full, cfg.crop, float(cfg.max_disp)
        if cfg.speckle_mode == "diverse":
            ref, sinfo = render_speckle_diverse(B, S, g, dev)
        else:
            ref, sinfo = render_speckle(B, S, cfg.speckle, g, dev)

        ar = torch.arange(S, device=dev, dtype=torch.float32) - (S - 1) / 2.0
        yc, xc = torch.meshgrid(ar, ar, indexing="ij")

        # --- choose type per sample ---
        p = torch.tensor([m.p_translation, m.p_rotation, m.p_smooth], device=dev)
        p = p / p.sum()
        r = torch.rand(B, generator=g, device=dev)
        t_type = torch.where(r < p[0], 0, torch.where(r < p[0] + p[1], 1, 2))
        is_occ = torch.rand(B, generator=g, device=dev) < m.p_occlusion
        t_type = torch.where(is_occ, torch.full_like(t_type, 3), t_type)

        flow = torch.zeros((B, 2, S, S), device=dev)
        # translation (radial magnitude), optional superimposed smooth deformation
        add_def = torch.rand(B, generator=g, device=dev) < m.trans_deform_frac
        Rt = torch.where(add_def, torch.full((B,), max(0.0, R - m.smooth_amp), device=dev),
                         torch.full((B,), R, device=dev))
        mag = Rt * torch.rand(B, generator=g, device=dev)
        ang = 2 * math.pi * torch.rand(B, generator=g, device=dev)
        trans = torch.stack([mag * torch.cos(ang), mag * torch.sin(ang)], 1)[:, :, None, None]
        sm = smooth_field(B, S, m.smooth_amp, m.smooth_sigma_range, g, dev)
        f_trans = trans + sm * add_def[:, None, None, None].float()

        # rotation about the canvas centre (== crop centre), optional capped translation
        th = torch.deg2rad(_u((B,), g, -m.rot_range_deg, m.rot_range_deg, dev))
        c, s = torch.cos(th)[:, None, None], torch.sin(th)[:, None, None]
        f_rot = torch.stack([(c - 1) * xc - s * yc, s * xc + (c - 1) * yc], 1)
        half_diag = 0.5 * math.sqrt(2.0) * C
        rot_corner = 2 * half_diag * torch.abs(torch.sin(th / 2))
        tmag = torch.clamp(R - rot_corner, min=0, max=40.0) * torch.rand(B, generator=g, device=dev)
        tmag = tmag * (torch.rand(B, generator=g, device=dev) < m.rot_translate_frac).float()
        tang = 2 * math.pi * torch.rand(B, generator=g, device=dev)
        f_rot = f_rot + torch.stack([tmag * torch.cos(tang), tmag * torch.sin(tang)], 1)[:, :, None, None]

        # pure small smooth deformation
        f_sm = sm

        # occlusion: finite patch on black + large rigid motion (rotation or translation)
        occ_rot = torch.rand(B, generator=g, device=dev) < m.occ_rot_frac
        omag = R * _u((B,), g, *m.occ_disp_frac_range, dev)
        oang = 2 * math.pi * torch.rand(B, generator=g, device=dev)
        f_otr = torch.stack([omag * torch.cos(oang), omag * torch.sin(oang)], 1)[:, :, None, None] \
            .expand(B, 2, S, S)
        th_o = torch.deg2rad(_u((B,), g, -45.0, 45.0, dev))
        co, so = torch.cos(th_o)[:, None, None], torch.sin(th_o)[:, None, None]
        f_orot = torch.stack([(co - 1) * xc - so * yc, so * xc + (co - 1) * yc], 1)
        f_occ = torch.where(occ_rot[:, None, None, None], f_orot, f_otr)

        sel = t_type[:, None, None, None]
        flow = torch.where(sel == 0, f_trans, flow)
        flow = torch.where(sel == 1, f_rot, flow)
        flow = torch.where(sel == 2, f_sm, flow)
        flow = torch.where(sel == 3, f_occ, flow)

        # warp field: legacy uses the label itself (read at the deformed pixel); exact uses the
        # inverse map, analytic for rigid rotation, fixed-point for translation(+smooth)/smooth.
        if cfg.label_mode == "legacy":
            wfield = flow
        elif cfg.label_mode == "exact":
            wfield = invert_displacement(flow)
            zero = torch.zeros(B, device=dev)
            w_rot = rotation_inverse_map(xc, yc, th, tmag * torch.cos(tang), tmag * torch.sin(tang))
            w_orot = rotation_inverse_map(xc, yc, th_o, zero, zero)
            wfield = torch.where(sel == 1, w_rot, wfield)
            wfield = torch.where((sel == 3) & occ_rot[:, None, None, None], w_orot, wfield)
        else:
            raise ValueError(cfg.label_mode)

        pf = _u((B,), g, *m.occ_patch_frac_range, dev)
        P = (S * pf).floor()
        o = ((S - P) / 2).floor()
        ii = torch.arange(S, device=dev, dtype=torch.float32)
        in_p = ((ii[None] >= o[:, None]) & (ii[None] < (o + P)[:, None])).float()   # (B, S)
        patch = in_p[:, :, None] * in_p[:, None, :]
        occ_f = (t_type == 3).float()[:, None, None]
        patch = patch * occ_f + (1 - occ_f)                   # non-occ samples: whole canvas
        ref = torch.floor(ref * patch)                        # uint8 cast semantics (black outside)

        dfm = warp_inverse(ref, wfield)
        ref = torch.round(ref).clamp(0, 255)
        dfm = torch.round(dfm).clamp(0, 255)
        nstd = _u((B,), g, *cfg.noise_std_range, dev)[:, None, None]
        ref = torch.floor((ref + nstd * torch.randn(ref.shape, generator=g, device=dev)).clamp(0, 255))
        dfm = torch.floor((dfm + nstd * torch.randn(dfm.shape, generator=g, device=dev)).clamp(0, 255))

        y0 = (S - C) // 2
        sl = slice(y0, y0 + C)
        ref_c, def_c = ref[:, sl, sl], dfm[:, sl, sl]
        flow_c = flow[:, :, sl, sl].contiguous()
        valid = torch.ones((B, C, C), device=dev)
        if bool((t_type == 3).any()):
            yy, xx = torch.meshgrid(torch.arange(C, device=dev, dtype=torch.float32),
                                    torch.arange(C, device=dev, dtype=torch.float32), indexing="ij")
            dx, dy = xx[None] + flow_c[:, 0], yy[None] + flow_c[:, 1]
            in_crop = (dx >= 0) & (dx < C) & (dy >= 0) & (dy < C)
            v_occ = ((patch[:, sl, sl] > 0.5) & in_crop).float()
            valid = torch.where((t_type == 3)[:, None, None], v_occ, valid)

        img1 = ref_c[:, None].expand(B, 3, C, C).contiguous()
        img2 = def_c[:, None].expand(B, 3, C, C).contiguous()
        info = {"type": t_type, **sinfo, "noise_std": nstd.view(-1)}
        return img1, img2, flow_c, valid, info
