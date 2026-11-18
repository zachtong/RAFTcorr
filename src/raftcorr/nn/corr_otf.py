# Identical to the research implementation of the RAFTcorr paper (commit 06eb238).
"""On-the-fly (memory-efficient) RAFT correlation lookup, mathematically identical to CorrBlock.

CorrBlock stores the all-pairs volume corr(f1, f2) / sqrt(C) and its avg-pooled pyramid, then
bilinearly samples a (2r+1)^2 window around coords / 2^l at every level (zero padding outside).
Both pooling and bilinear sampling are linear, so
    level l lookup at c + delta  =  < f1(x), bilinear_sample(avgpool^l(f2), c + delta) > / sqrt(C).
For integer offsets delta all window samples share the fractional part of c, hence every bilinear
sample is a weighted sum of 4 integer-position samples, and the whole (2r+1)^2 window follows from
the (2r+2)^2 integer dot products D[j] = < f1(x), f2_l(floor(c) + j) >, j in [-r, r+1]^2.
D is computed by a custom autograd function that never materialises the gathered features for the
backward pass (it recomputes them), so memory is O(C * H * W) instead of O((H * W)^2).
Ordering of the output channels reproduces CorrBlock exactly (including RAFT's x/y offset quirk).
Three implementations of D: impl="loop" gathers channel-first features one window offset at a time and
reduces with mul + sum; impl="rows" keeps the features channels-last, gathers whole C-vectors with
index_select (contiguous rows) and contracts them with bmm, several offsets per call; impl="tile"
(inference only, 2026-11-03) splits the query grid into TILE x TILE tiles, gathers for each tile the union
of its windows as one f2 patch, computes all tile-patch dot products with one batched GEMM and picks every
pixel's (2r+2)^2 window out of the product. Same fp32 dot products as "rows" (only the summation order
differs), but compute-bound instead of gather-bound. Tiles whose patch is huge (scattered matches) fall back
to a per-pixel gather.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class _IntegerWindowCorr(torch.autograd.Function):
    """D[b, j, p] = sum_c f1[b, c, p] * f2[b, c, idx[b, j, p]] (0 where idx invalid)."""

    @staticmethod
    def forward(ctx, f1, f2, idx, valid):
        B, C, N1 = f1.shape
        J = idx.shape[1]
        out = f1.new_zeros((B, J, N1))
        for j in range(J):
            g = torch.gather(f2, 2, idx[:, j:j + 1, :].expand(B, C, N1))
            out[:, j] = (f1 * g).sum(1) * valid[:, j]
        ctx.save_for_backward(f1, f2, idx, valid)
        return out

    @staticmethod
    def backward(ctx, gD):
        f1, f2, idx, valid = ctx.saved_tensors
        B, C, N1 = f1.shape
        gD = gD * valid
        g1 = torch.zeros_like(f1)
        g2 = torch.zeros_like(f2)
        for j in range(idx.shape[1]):
            ij = idx[:, j:j + 1, :].expand(B, C, N1)
            w = gD[:, j:j + 1, :]
            g1 += torch.gather(f2, 2, ij) * w
            g2.scatter_add_(2, ij, f1 * w)
        return g1, g2, None, None


ROWS_BUDGET_BYTES = 1 << 30                                        # gathered-rows buffer per bmm call


class _WindowRowsCorr(torch.autograd.Function):
    """D[m, j] = <f1t[m], f2t[gidx[m, j]]> * valid[m, j] on channels-last features.

    f1t (M, C) with M = B * N1, f2t (B * N2, C), gidx (M, J) global row index into f2t. Gathered rows are
    recomputed in backward (memory O(jc * M * C) per call instead of O(J * M * C)).
    """

    @staticmethod
    def forward(ctx, f1t, f2t, gidx, valid, jc):
        M, C = f1t.shape
        out = f1t.new_empty((M, gidx.shape[1]))
        for j0 in range(0, gidx.shape[1], jc):
            ix = gidx[:, j0:j0 + jc]
            rows = f2t.index_select(0, ix.reshape(-1)).view(M, ix.shape[1], C)
            out[:, j0:j0 + jc] = torch.bmm(rows, f1t[:, :, None])[..., 0]
        out *= valid
        ctx.save_for_backward(f1t, f2t, gidx, valid)
        ctx.jc = jc
        return out

    @staticmethod
    def backward(ctx, gD):
        f1t, f2t, gidx, valid = ctx.saved_tensors
        M, C = f1t.shape
        need1, need2 = ctx.needs_input_grad[:2]
        gD = gD * valid
        g1 = torch.zeros_like(f1t) if need1 else None
        g2 = torch.zeros_like(f2t) if need2 else None
        for j0 in range(0, gidx.shape[1], ctx.jc):
            ix = gidx[:, j0:j0 + ctx.jc]
            w = gD[:, j0:j0 + ctx.jc]
            if need1:
                rows = f2t.index_select(0, ix.reshape(-1)).view(M, ix.shape[1], C)
                g1 += torch.bmm(w[:, None, :], rows)[:, 0]
            if need2:
                g2.index_add_(0, ix.reshape(-1), (w[:, :, None] * f1t[:, None, :]).reshape(-1, C))
        return g1, g2, None, None, None


def _to_rows(f):
    """(B, C, H, W) -> channels-last rows (B * H * W, C)."""
    B, C, H, W = f.shape
    return f.permute(0, 2, 3, 1).reshape(B * H * W, C)


TILE = 16                                                          # query tile side (feature pixels), impl="tile"
TILE_BIN = 4                                                       # patch sides rounded up to multiples of this
TILE_MAX_PATCH = 64 * 64                                           # larger patches use the per-pixel gather
TILE_BUDGET_BYTES = 1 << 28                                        # working memory per batched GEMM


class CorrBlockOTF:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4, impl="loop"):
        if impl not in ("loop", "rows", "tile"):
            raise ValueError(f"impl must be 'loop', 'rows' or 'tile', got {impl!r}")
        if impl == "tile" and torch.is_grad_enabled() and (fmap1.requires_grad or fmap2.requires_grad):
            raise RuntimeError("impl='tile' has no backward; use impl='rows' for training")
        self.num_levels, self.radius, self.impl = num_levels, radius, impl
        B, C, H, W = fmap1.shape
        self.scale = 1.0 / float(C) ** 0.5
        self.shape = (B, C, H, W)
        self.pyr = []
        f2 = fmap2.float()
        for i in range(num_levels):
            if i > 0:
                f2 = F.avg_pool2d(f2, 2, stride=2)
            self.pyr.append(f2)
        self.level_shapes = [tuple(f.shape[-2:]) for f in self.pyr]
        if impl == "loop":
            self.f1 = fmap1.float().reshape(B, C, H * W)
        elif impl == "tile":
            T = TILE
            Hp, Wp = -(-H // T) * T, -(-W // T) * T
            f1 = F.pad(fmap1.float(), (0, Wp - W, 0, Hp - H))
            nty, ntx = Hp // T, Wp // T
            self.f1_tiles = (f1.reshape(B, C, nty, T, ntx, T).permute(0, 2, 4, 3, 5, 1)
                             .reshape(B * nty * ntx, T * T, C).contiguous())   # (nT, T*T, C)
            self.tile_grid = (Hp, Wp, nty, ntx)
            self.pyr_rows = []                                       # channels-last + one zero row at the end,
            for f in self.pyr:                                       # the target of out-of-frame positions
                rows = f.new_zeros((f.shape[0] * f.shape[2] * f.shape[3] + 1, C))
                rows[:-1] = _to_rows(f)
                self.pyr_rows.append(rows)
            self.pyr = None                                          # channels-first copy not needed
        else:
            self.f1 = _to_rows(fmap1.float())
            self.pyr_rows = [_to_rows(f) for f in self.pyr]
            n_win = (2 * radius + 2) ** 2
            self.jc = int(max(1, min(n_win, ROWS_BUDGET_BYTES // (B * H * W * C * 4))))

    def _window_dots(self, i, idx, valid):
        """Integer-window dot products D (B, n*n, H*W) at pyramid level i."""
        B, C, H, W = self.shape
        if self.impl == "loop":
            f2 = self.pyr[i]
            return _IntegerWindowCorr.apply(self.f1, f2.reshape(B, C, -1), idx, valid)
        Hl, Wl = self.pyr[i].shape[-2:]
        off = (torch.arange(B, device=idx.device) * (Hl * Wl))[:, None, None]
        gidx = (idx + off).permute(0, 2, 1).reshape(B * H * W, -1)
        vt = valid.permute(0, 2, 1).reshape(B * H * W, -1)
        D = _WindowRowsCorr.apply(self.f1, self.pyr_rows[i], gidx, vt, self.jc)
        return D.reshape(B, H * W, -1).permute(0, 2, 1)

    def _tile_dots(self, i, x0, y0):
        """Integer-window dot products D (B, n*n, H*W) at pyramid level i, tiled-GEMM version.

        x0, y0 (B, H, W) long: floor of the lookup centre at this level. Entry k = jy * n + jx is the dot
        product with f2_l at (y0 - r + jy, x0 - r + jx); positions outside the level map give 0, as in "rows".
        """
        B, C, H, W = self.shape
        r, T = self.radius, TILE
        n = 2 * r + 2
        Hp, Wp, nty, ntx = self.tile_grid
        Hl, Wl = self.level_shapes[i]
        dev = x0.device
        if (Hp, Wp) != (H, W):                                       # padded queries copy the edge centres
            iy = torch.arange(Hp, device=dev).clamp(max=H - 1)
            ix = torch.arange(Wp, device=dev).clamp(max=W - 1)
            x0, y0 = x0[:, iy][:, :, ix], y0[:, iy][:, :, ix]
        tiles = lambda a: a.reshape(B, nty, T, ntx, T).permute(0, 1, 3, 2, 4).reshape(-1, T * T)
        xo, yo = tiles(x0 - r), tiles(y0 - r)                        # window origins, (nT, T*T)
        xmin, ymin = xo.amin(1), yo.amin(1)
        lx, ly = xo - xmin[:, None], yo - ymin[:, None]              # origins inside the tile patch
        ux = (lx.amax(1) + n + TILE_BIN - 1) // TILE_BIN * TILE_BIN  # patch sides, binned so that tiles
        uy = (ly.amax(1) + n + TILE_BIN - 1) // TILE_BIN * TILE_BIN  # of similar size share one GEMM
        f2t = self.pyr_rows[i]
        zero_row = f2t.shape[0] - 1
        row0 = torch.arange(B, device=dev).repeat_interleave(nty * ntx) * (Hl * Wl)   # batch offset per tile
        out = torch.empty((xo.shape[0], T * T, n * n), device=dev, dtype=f2t.dtype)
        key = torch.where(uy * ux > TILE_MAX_PATCH, -1, uy * (1 << 20) + ux)
        for kb in torch.unique(key).tolist():                        # one host sync per level
            sel = torch.nonzero(key == kb)[:, 0]
            if kb < 0:
                self._gather_tiles(out, sel, xo, yo, row0, f2t, Hl, Wl)
                continue
            Uy, Ux = divmod(kb, 1 << 20)
            U = Uy * Ux
            m = max(1, TILE_BUDGET_BYTES // (U * (4 * C + 16 + 4 * T * T) + 12 * T * T * n * n))
            ay, ax = torch.arange(Uy, device=dev), torch.arange(Ux, device=dev)
            jj = torch.arange(n, device=dev)
            woff = (jj[:, None] * Ux + jj[None, :]).reshape(1, 1, -1)   # window entry k = jy * n + jx
            for c0 in range(0, sel.numel(), m):
                tt = sel[c0:c0 + m]
                py, px = ymin[tt, None] + ay, xmin[tt, None] + ax    # patch rows / columns, (m, Uy), (m, Ux)
                inside = ((py >= 0) & (py < Hl))[:, :, None] & ((px >= 0) & (px < Wl))[:, None, :]
                flat = torch.where(inside, py[:, :, None] * Wl + px[:, None, :] + row0[tt, None, None], zero_row)
                patch = f2t.index_select(0, flat.reshape(-1)).view(-1, U, C)
                prod = torch.bmm(self.f1_tiles[tt], patch.transpose(1, 2))           # (m, T*T, U)
                out[tt] = torch.gather(prod, 2, (ly[tt] * Ux + lx[tt])[:, :, None] + woff)
        D = out.view(B, nty, ntx, T, T, n * n).permute(0, 5, 1, 3, 2, 4).reshape(B, n * n, Hp, Wp)
        return D[:, :, :H, :W].reshape(B, n * n, H * W)

    def _gather_tiles(self, out, sel, xo, yo, row0, f2t, Hl, Wl):
        """Per-pixel fallback of _tile_dots for tiles whose windows are too scattered for one patch."""
        C, T, n = self.shape[1], TILE, 2 * self.radius + 2
        jj = torch.arange(n, device=xo.device)
        jy, jx = torch.meshgrid(jj, jj, indexing="ij")
        jy, jx = jy.reshape(1, 1, -1), jx.reshape(1, 1, -1)
        m = max(1, TILE_BUDGET_BYTES // (T * T * n * n * (4 * C + 32)))
        for c0 in range(0, sel.numel(), m):
            tt = sel[c0:c0 + m]
            X, Y = xo[tt, :, None] + jx, yo[tt, :, None] + jy                     # (m, T*T, n*n)
            ok = (X >= 0) & (X < Wl) & (Y >= 0) & (Y < Hl)
            gid = torch.where(ok, Y * Wl + X + row0[tt, None, None], f2t.shape[0] - 1)
            rows = f2t.index_select(0, gid.reshape(-1)).view(-1, n * n, C)       # (m*T*T, n*n, C)
            out[tt] = torch.bmm(rows, self.f1_tiles[tt].reshape(-1, C, 1)).view(-1, T * T, n * n)

    def __call__(self, coords):
        B, C, H, W = self.shape
        r = self.radius
        n = 2 * r + 2
        dev = coords.device
        jj = torch.arange(-r, r + 2, device=dev)
        jy, jx = torch.meshgrid(jj, jj, indexing="ij")             # integer window, (n, n)
        out = []
        for i, (Hl, Wl) in enumerate(self.level_shapes):
            c = coords / 2 ** i                                     # (B, 2, H, W): x, y
            x0, y0 = torch.floor(c[:, 0]), torch.floor(c[:, 1])
            fx, fy = c[:, 0] - x0, c[:, 1] - y0
            if self.impl == "tile":
                D = self._tile_dots(i, x0.long(), y0.long())
            else:
                X = x0.long()[:, None] + jx.reshape(1, -1, 1, 1)     # (B, n*n, H, W)
                Y = y0.long()[:, None] + jy.reshape(1, -1, 1, 1)
                valid = ((X >= 0) & (X < Wl) & (Y >= 0) & (Y < Hl)).float().reshape(B, n * n, H * W)
                idx = (Y.clamp(0, Hl - 1) * Wl + X.clamp(0, Wl - 1)).reshape(B, n * n, H * W)
                D = self._window_dots(i, idx, valid)
            D = D.reshape(B, n, n, H, W) * self.scale                # D[b, jy, jx]
            # bilinear combination -> window value at offsets (oy, ox) in [-r, r]^2
            w00 = ((1 - fx) * (1 - fy))[:, None, None]
            w01 = (fx * (1 - fy))[:, None, None]
            w10 = ((1 - fx) * fy)[:, None, None]
            w11 = (fx * fy)[:, None, None]
            win = (w00 * D[:, :-1, :-1] + w01 * D[:, :-1, 1:] + w10 * D[:, 1:, :-1] + w11 * D[:, 1:, 1:])
            # win[b, oy, ox]; CorrBlock flattens with the x-offset as the OUTER index (RAFT quirk)
            out.append(win.permute(0, 2, 1, 3, 4).reshape(B, (2 * r + 1) ** 2, H, W))
        return torch.cat(out, 1).contiguous().float()
