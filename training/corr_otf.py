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
Two implementations of D: impl="loop" gathers channel-first features one window offset at a time and
reduces with mul + sum; impl="rows" keeps the features channels-last, gathers whole C-vectors with
index_select (contiguous rows) and contracts them with bmm, several offsets per call.
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


class CorrBlockOTF:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4, impl="loop"):
        if impl not in ("loop", "rows"):
            raise ValueError(f"impl must be 'loop' or 'rows', got {impl!r}")
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
        if impl == "loop":
            self.f1 = fmap1.float().reshape(B, C, H * W)
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

    def __call__(self, coords):
        B, C, H, W = self.shape
        r = self.radius
        n = 2 * r + 2
        dev = coords.device
        jj = torch.arange(-r, r + 2, device=dev)
        jy, jx = torch.meshgrid(jj, jj, indexing="ij")             # integer window, (n, n)
        out = []
        for i, f2 in enumerate(self.pyr):
            Hl, Wl = f2.shape[-2:]
            c = coords / 2 ** i                                      # (B, 2, H, W): x, y
            x0, y0 = torch.floor(c[:, 0]), torch.floor(c[:, 1])
            fx, fy = c[:, 0] - x0, c[:, 1] - y0
            X = x0.long()[:, None] + jx.reshape(1, -1, 1, 1)         # (B, n*n, H, W)
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
