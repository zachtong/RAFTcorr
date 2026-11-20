"""Generator v2 for the s1/s2/s4/s8 study: composite deformation with exact labels.

Forward map (reference X -> deformed p), about the crop centre c:
    phi(X) = c + R(theta) (X + w(X) - c) + t
  w     : natural-image finite deformation (blurred natural image, any scale; amplitude capped by a
          maximum strain so the map never folds),
  theta : rigid rotation, t : rigid translation.
Label   : exact Lagrangian displacement d(X) = phi(X) - X on the reference grid.
Image   : def(p) = ref(X(p)) with the exact inverse X(p): rigid part analytic, then the fixed point
          X = q - w(X), q = c + R^T (p - c - t)  (contraction since max|grad w| <= 0.4).
Sample types: pure translation / pure rotation / pure deformation / composite (2-3 components).
Peak displacement per sample is drawn octave-balanced up to the variant's theoretical maximum
D_max = r * s * 2**(L-1). A fraction of samples carries an occlusion mask (finite patch, holes, edge)
over a static uniform background; the loss is then restricted to visible material pixels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from gpu_synth import render_speckle_diverse, sample_field, warp_inverse

TYPES = ("translation", "rotation", "deformation", "composite")
OCC_KINDS = ("none", "patch", "holes", "edge")


@dataclass
class FieldCfgV2:
    crop: int = 512
    s: int = 8                       # encoder downsampling factor of the variant
    radius: int = 4
    levels: int = 4
    p_type: tuple = (0.20, 0.15, 0.15, 0.50)          # translation, rotation, deformation, composite
    p_occ: float = 0.25
    p_occ_kind: tuple = (0.4, 0.3, 0.3)                # patch, holes, edge
    sigma_range: tuple = (2.0, 600.0)                  # natural-field blur (px), log-uniform
    max_strain_range: tuple = (0.02, 0.4)
    rot_max_deg: float = 45.0
    noise_std_range: tuple = (2.0, 10.0)
    min_edge: float = 4.0            # upper edge of the first magnitude bin [0, min_edge] (px); octaves above
    max_target: float = 0.0          # cap on the target peak (px); 0 = up to d_max
    # detail layer: band-pass Gaussian random field added to w of a fraction of samples, wavelength and strain
    # log-uniform and independent of the peak target (covers small-amplitude / short-wavelength content)
    p_detail: float = 0.0
    detail_lambda: tuple = (6.0, 512.0)              # centre wavelength range (px)
    detail_strain: tuple = (0.002, 0.2)              # max |grad| of the detail field
    detail_bw: float = 0.35                          # std of ln(k / k_c) of the radial band-pass

    @property
    def d_max(self) -> float:
        return float(self.radius * self.s * 2 ** (self.levels - 1))

    @property
    def bins(self) -> list:
        top = self.d_max if self.max_target <= 0 else min(self.max_target, self.d_max)
        edges = [0.0, min(self.min_edge, top)]
        while edges[-1] < top:
            edges.append(min(edges[-1] * 2, top))
        return edges

    @property
    def full(self) -> int:
        f = self.crop + 2 * int(math.ceil(self.d_max)) + 32
        return int(math.ceil(f / 16) * 16)


class NaturalFieldBank:
    """Grayscale natural images kept on the GPU; draws random crops (scale, flip, rot90) as fields."""

    def __init__(self, image_dir, device="cuda", side=768, max_images=None):
        files = sorted(Path(image_dir).glob("*.png"))[:max_images]
        if not files:
            raise FileNotFoundError(f"no png images in {image_dir}")
        imgs = []
        for f in files:
            im = Image.open(f).convert("L")
            k = side / min(im.size)
            im = im.resize((max(side, int(im.size[0] * k)), max(side, int(im.size[1] * k))), Image.BICUBIC)
            imgs.append(torch.tensor(np.asarray(im), dtype=torch.uint8, device=device))
        self.imgs = imgs
        self.device = device

    def sample(self, n, S, gen):
        out = torch.empty((n, S, S), device=self.device)
        for i in range(n):
            im = self.imgs[int(torch.randint(len(self.imgs), (1,), generator=gen, device=self.device))]
            H, W = im.shape
            side = int(min(H, W) * float(0.35 + 0.65 * torch.rand(1, generator=gen, device=self.device)))
            y = int(torch.randint(H - side + 1, (1,), generator=gen, device=self.device))
            x = int(torch.randint(W - side + 1, (1,), generator=gen, device=self.device))
            c = im[y:y + side, x:x + side].float()[None, None] / 255.0
            c = torch.rot90(c, int(torch.randint(4, (1,), generator=gen, device=self.device)), dims=(2, 3))
            out[i] = F.interpolate(c, size=(S, S), mode="bilinear", align_corners=False)[0, 0]
        return out


def _u(shape, gen, lo, hi, dev):
    return lo + (hi - lo) * torch.rand(shape, generator=gen, device=dev)


def blur_fft(x, sigma):
    """Per-sample Gaussian low-pass of (B, C, S, S) in Fourier space (periodic, fine for fields)."""
    S = x.shape[-1]
    f = torch.fft.fftfreq(S, device=x.device)
    f2 = f[:, None] ** 2 + f[None, :] ** 2
    H = torch.exp(-2 * math.pi ** 2 * sigma[:, None, None, None] ** 2 * f2[None, None])
    return torch.fft.ifft2(torch.fft.fft2(x) * H).real


def unit_field(bank, B, S, sigma, gen, interior):
    """Two-component natural field normalised to max|n| = 1 over the interior; returns (n, max|grad n|)."""
    raw = torch.stack([bank.sample(B, S, gen), bank.sample(B, S, gen)], 1)
    return _normalise_unit(blur_fft(raw - raw.mean(dim=(2, 3), keepdim=True), sigma), interior)


def band_field(B, S, lam_c, gen, interior, dev, bw):
    """Two-component Gaussian random field band-passed around wavelength lam_c (B,) with a log-Gaussian radial
    filter exp(-ln(k lam_c)^2 / (2 bw^2)); normalised like unit_field."""
    z = torch.randn((B, 2, S, S), generator=gen, device=dev)
    f = torch.fft.fftfreq(S, device=dev)
    k = torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2).clamp(min=1e-9)
    filt = torch.exp(-torch.log(k[None] * lam_c[:, None, None]) ** 2 / (2 * bw ** 2))
    filt[:, 0, 0] = 0.0
    n = torch.fft.ifft2(torch.fft.fft2(z) * filt[:, None]).real
    return _normalise_unit(n, interior)


def _normalise_unit(n, interior):
    sl = slice(interior[0], interior[1])
    peak = n[:, :, sl, sl].abs().amax(dim=(1, 2, 3)).clamp(min=1e-8)
    n = n / peak[:, None, None, None]
    gy = (n[:, :, 2:, 1:-1] - n[:, :, :-2, 1:-1]) / 2
    gx = (n[:, :, 1:-1, 2:] - n[:, :, 1:-1, :-2]) / 2
    sl2 = slice(interior[0] - 1, interior[1] - 1)
    g = torch.maximum(gx[:, :, sl2, sl2].abs().amax(dim=(1, 2, 3)), gy[:, :, sl2, sl2].abs().amax(dim=(1, 2, 3)))
    return n, g


def draw_params(cfg, B, gen, dev):
    """Random sample types, octave-balanced target peak displacement and its split into components."""
    ptype = torch.tensor(cfg.p_type, device=dev)
    typ = torch.multinomial(ptype, B, replacement=True, generator=gen)
    edges = torch.tensor(cfg.bins, device=dev)
    k = torch.randint(len(edges) - 1, (B,), generator=gen, device=dev)
    m = edges[k] + (edges[k + 1] - edges[k]) * torch.rand(B, generator=gen, device=dev)
    use = torch.zeros((B, 3), device=dev)                       # columns: translation, rotation, deform
    use[typ == 0, 0] = 1
    use[typ == 1, 1] = 1
    use[typ == 2, 2] = 1
    comp = typ == 3
    if comp.any():
        r = torch.rand((int(comp.sum()), 3), generator=gen, device=dev)
        subset = r < 0.67
        none_or_one = subset.sum(1) < 2                           # composites need >= 2 components
        subset[none_or_one] = torch.tensor([True, True, True], device=dev)
        use[comp] = subset.float()
    frac = -torch.log(torch.rand((B, 3), generator=gen, device=dev).clamp(min=1e-9)) * use   # Dirichlet(1)
    frac = frac / frac.sum(1, keepdim=True)
    return {"type": typ, "target": m, "share": frac * m[:, None],
            "t_ang": 2 * math.pi * torch.rand(B, generator=gen, device=dev),
            "rot_sign": torch.where(torch.rand(B, generator=gen, device=dev) < 0.5, -1.0, 1.0),
            "sigma": torch.exp(_u((B,), gen, math.log(cfg.sigma_range[0]), math.log(cfg.sigma_range[1]), dev)),
            "eps_max": _u((B,), gen, *cfg.max_strain_range, dev)}


def build_fields(cfg, prm, bank, gen, dev, escalate=6):
    """Turn component shares into (theta, t, w) and the exact forward displacement d on the canvas."""
    S, C = cfg.full, cfg.crop
    B = prm["target"].shape[0]
    o = (S - C) // 2
    interior = (o, o + C)
    half_diag = C / math.sqrt(2.0)
    m_t, m_r, m_d = prm["share"].unbind(1)

    sigma = prm["sigma"].clone()
    w = torch.zeros((B, 2, S, S), device=dev)
    need = m_d > 0
    real_d = torch.zeros(B, device=dev)
    for _ in range(escalate):                     # strain-limited fields too small -> larger sigma
        if not need.any():
            break
        idx = need.nonzero().view(-1)
        n, g = unit_field(bank, len(idx), S, sigma[idx], gen, interior)
        a = torch.minimum(m_d[idx], prm["eps_max"][idx] / g.clamp(min=1e-8))
        w[idx] = n * a[:, None, None, None]
        real_d[idx] = a
        short = a < 0.9 * m_d[idx]
        sigma[idx[short]] = sigma[idx[short]] * 2.0
        need = torch.zeros_like(need)
        need[idx[short]] = True
    deficit = (m_d - real_d).clamp(min=0)          # give what deformation could not carry to rigid parts
    has_t = m_t > 0
    m_t = torch.where(has_t, m_t + deficit, m_t)
    m_r = torch.where(~has_t & (m_r > 0), m_r + deficit, m_r)

    # unit deformation field and its strain-limited maximum amplitude (so w can be rescaled below)
    a = real_d.clone()
    n_unit = w / a.clamp(min=1e-8)[:, None, None, None]
    a_cap = torch.where(a > 0, a * prm["eps_max"] / _max_strain(w, interior).clamp(min=1e-8), a)

    rot_cap = 2 * half_diag * math.sin(math.radians(cfg.rot_max_deg) / 2)
    ar = torch.arange(S, device=dev, dtype=torch.float32) - (S - 1) / 2.0
    yc, xc = torch.meshgrid(ar, ar, indexing="ij")
    target = prm["target"]
    for _ in range(4):
        # the peak of a composite is below the sum of its components' peaks (different directions):
        # rescale all components so the realised peak over the crop matches the octave-balanced target
        theta = 2 * torch.asin((m_r.clamp(max=rot_cap) / (2 * half_diag)).clamp(max=1.0)) * prm["rot_sign"]
        t = torch.stack([m_t * torch.cos(prm["t_ang"]), m_t * torch.sin(prm["t_ang"])], 1)
        a_used = a.clone()
        w = n_unit * a_used[:, None, None, None]
        c, s_ = torch.cos(theta)[:, None, None], torch.sin(theta)[:, None, None]
        zx, zy = xc[None] + w[:, 0], yc[None] + w[:, 1]                 # X + w(X) - c
        px = c * zx - s_ * zy + t[:, 0, None, None]
        py = s_ * zx + c * zy + t[:, 1, None, None]
        d = torch.stack([px - xc[None], py - yc[None]], 1)              # phi(X) - X
        peak = d[:, :, interior[0]:interior[1], interior[0]:interior[1]].norm(dim=1).amax(dim=(1, 2))
        ratio = (target / peak.clamp(min=1e-6)).clamp(0.5, 2.0)
        m_t, m_r = m_t * ratio, m_r * ratio
        a_new = a * ratio
        over = (a_new - a_cap).clamp(min=0)                            # strain cap: move excess to t
        a = torch.minimum(a_new, a_cap)
        m_t = torch.where(m_t > 0, m_t + over, m_t)
    det_lam = torch.zeros(B, device=dev)
    det_amp = torch.zeros(B, device=dev)
    if cfg.p_detail > 0:
        sel = (torch.rand(B, generator=gen, device=dev) < cfg.p_detail).nonzero().view(-1)
        if len(sel):
            lam = torch.exp(_u((len(sel),), gen, *[math.log(v) for v in cfg.detail_lambda], dev))
            eps = torch.exp(_u((len(sel),), gen, *[math.log(v) for v in cfg.detail_strain], dev))
            nb, gb = band_field(len(sel), S, lam, gen, interior, dev, cfg.detail_bw)
            amp = eps / gb.clamp(min=1e-8)
            w = w.clone()
            w[sel] = w[sel] + nb * amp[:, None, None, None]
            det_lam[sel], det_amp[sel] = lam, amp
            c, s_ = torch.cos(theta)[:, None, None], torch.sin(theta)[:, None, None]
            zx, zy = xc[None] + w[:, 0], yc[None] + w[:, 1]
            px = c * zx - s_ * zy + t[:, 0, None, None]
            py = s_ * zx + c * zy + t[:, 1, None, None]
            d = torch.stack([px - xc[None], py - yc[None]], 1)
            peak = d[:, :, interior[0]:interior[1], interior[0]:interior[1]].norm(dim=1).amax(dim=(1, 2))
    return {"d": d, "w": w, "theta": theta, "t": t, "sigma": sigma, "real_def": a_used, "peak": peak,
            "detail_lambda": det_lam, "detail_amp": det_amp}


def _max_strain(w, interior):
    sl = slice(interior[0] - 1, interior[1] - 1)
    gy = (w[:, :, 2:, 1:-1] - w[:, :, :-2, 1:-1]) / 2
    gx = (w[:, :, 1:-1, 2:] - w[:, :, 1:-1, :-2]) / 2
    return torch.maximum(gx[:, :, sl, sl].abs().amax(dim=(1, 2, 3)), gy[:, :, sl, sl].abs().amax(dim=(1, 2, 3)))


def inverse_map(fl, S, dev, iters=20):
    """Eulerian warp field W(p) = p - X(p) for the forward map in `fl` (exact up to the fixed point)."""
    ar = torch.arange(S, device=dev, dtype=torch.float32)
    yy, xx = torch.meshgrid(ar, ar, indexing="ij")
    cen = (S - 1) / 2.0
    th, t, w = fl["theta"], fl["t"], fl["w"]
    c, s_ = torch.cos(th)[:, None, None], torch.sin(th)[:, None, None]
    ux, uy = xx[None] - cen - t[:, 0, None, None], yy[None] - cen - t[:, 1, None, None]
    qx = c * ux + s_ * uy + cen                                         # q = c + R^T (p - c - t)
    qy = -s_ * ux + c * uy + cen
    X, Y = qx.clone(), qy.clone()
    for _ in range(iters):
        wv = sample_field(w, X, Y, mode="bicubic")
        X, Y = qx - wv[:, 0], qy - wv[:, 1]
    return torch.stack([xx[None] - X, yy[None] - Y], 1), (X, Y)


def occlusion_masks(cfg, B, gen, dev):
    """Material masks in reference coordinates (1 = speckled material, 0 = static background)."""
    S, C = cfg.full, cfg.crop
    kind = torch.zeros(B, dtype=torch.long, device=dev)
    occ = torch.rand(B, generator=gen, device=dev) < cfg.p_occ
    if occ.any():
        kind[occ] = 1 + torch.multinomial(torch.tensor(cfg.p_occ_kind, device=dev), int(occ.sum()),
                                          replacement=True, generator=gen)
    ar = torch.arange(S, device=dev, dtype=torch.float32) - (S - 1) / 2.0
    yc, xc = torch.meshgrid(ar, ar, indexing="ij")
    M = torch.ones((B, S, S), device=dev)
    for i in range(B):
        k = int(kind[i])
        if k == 1:      # finite patch (rectangle) on background
            hw = C * _u((2,), gen, 0.25, 0.45, dev)
            M[i] = ((xc.abs() <= hw[0]) & (yc.abs() <= hw[1])).float()
        elif k == 2:    # holes: random elliptical voids (metamaterial-like)
            nh = int(torch.randint(3, 12, (1,), generator=gen, device=dev))
            for _ in range(nh):
                cx, cy = _u((2,), gen, -0.6 * C, 0.6 * C, dev)
                a, b = _u((2,), gen, 0.03 * C, 0.12 * C, dev)
                ang = float(_u((1,), gen, 0, math.pi, dev))
                xr = (xc - cx) * math.cos(ang) + (yc - cy) * math.sin(ang)
                yr = -(xc - cx) * math.sin(ang) + (yc - cy) * math.cos(ang)
                M[i] = M[i] * (((xr / a) ** 2 + (yr / b) ** 2) > 1).float()
        elif k == 3:    # edge: specimen boundary, background on one side of a (slightly curved) line
            ang = float(_u((1,), gen, 0, 2 * math.pi, dev))
            off = float(_u((1,), gen, -0.3 * C, 0.3 * C, dev))
            curv = float(_u((1,), gen, -1.5e-3, 1.5e-3, dev))
            nrm = xc * math.cos(ang) + yc * math.sin(ang)
            tan = -xc * math.sin(ang) + yc * math.cos(ang)
            M[i] = ((nrm - off - curv * tan ** 2) < 0).float()
    return M, kind


class SpeckleSynthV2:
    """Training pairs for one variant: diverse speckle + composite exact deformation + occlusion."""

    def __init__(self, cfg: FieldCfgV2, bank: NaturalFieldBank, device="cuda", seed=0):
        self.cfg, self.bank, self.dev = cfg, bank, device
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

    @torch.no_grad()
    def sample(self, B, prm=None):
        cfg, g, dev = self.cfg, self.gen, self.dev
        S, C = cfg.full, cfg.crop
        tex, _ = render_speckle_diverse(B, S, g, dev, stretch=True, aa=True)
        prm = prm or draw_params(cfg, B, g, dev)
        fl = build_fields(cfg, prm, self.bank, g, dev)
        Wp, (Xp, Yp) = inverse_map(fl, S, dev)
        M, kind = occlusion_masks(cfg, B, g, dev)
        bg = _u((B,), g, 0, 255, dev)[:, None, None]
        ref = M * tex + (1 - M) * bg
        tex_def = warp_inverse(tex, Wp)
        M_def = (sample_field(M[:, None], Xp, Yp)[:, 0] > 0.5).float()
        dfm = M_def * tex_def + (1 - M_def) * bg
        nstd = _u((B,), g, *cfg.noise_std_range, dev)[:, None, None]
        ref = (ref.round() + nstd * torch.randn(ref.shape, generator=g, device=dev)).clamp(0, 255).floor()
        dfm = (dfm.round() + nstd * torch.randn(dfm.shape, generator=g, device=dev)).clamp(0, 255).floor()
        o = (S - C) // 2
        sl = slice(o, o + C)
        d = fl["d"][:, :, sl, sl].contiguous()
        yy, xx = torch.meshgrid(torch.arange(C, device=dev, dtype=torch.float32),
                                torch.arange(C, device=dev, dtype=torch.float32), indexing="ij")
        inside = ((xx[None] + d[:, 0] >= 0) & (xx[None] + d[:, 0] <= C - 1) &
                  (yy[None] + d[:, 1] >= 0) & (yy[None] + d[:, 1] <= C - 1)).float()
        occ = (kind > 0)[:, None, None].float()
        valid = occ * M[:, sl, sl] * inside + (1 - occ)
        peak = d.norm(dim=1).amax(dim=(1, 2))
        info = {**prm, **{k: v for k, v in fl.items() if k not in ("d", "w")}, "occ_kind": kind,
                "peak": peak, "mask_ref": M[:, sl, sl], "w": fl["w"][:, :, sl, sl]}
        img1 = ref[:, sl, sl][:, None].expand(B, 3, C, C).contiguous()
        img2 = dfm[:, sl, sl][:, None].expand(B, 3, C, C).contiguous()
        return img1, img2, d, valid, info
