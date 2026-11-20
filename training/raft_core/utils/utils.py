# Derived from RAFT (Teed and Deng, ECCV 2020), BSD 3-Clause License: see LICENSE-RAFT in raft_core/.
import torch
import torch.nn.functional as F


class InputPadder:
    """Pad images so spatial dims are divisible by ``factor`` (the model downsample factor)."""

    def __init__(self, dims, factor=8, mode="sintel"):
        self.ht, self.wd = dims[-2:]
        self.factor = factor
        if factor <= 1:
            self._pad = [0, 0, 0, 0]
        else:
            pad_ht = (((self.ht // factor) + 1) * factor - self.ht) % factor
            pad_wd = (((self.wd // factor) + 1) * factor - self.wd) % factor
            if mode == "sintel":
                self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
            else:
                self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


CUDNN_GRID_SAMPLE_MAX = 2 ** 31  # cuDNN's grid sampler fails above this many input elements


def bilinear_sampler(img, coords, mode="bilinear", mask=False):
    """Wrapper for grid_sample that uses pixel coordinates."""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    if img.numel() > CUDNN_GRID_SAMPLE_MAX:
        # e.g. a large all-pairs correlation volume: PyTorch's own kernel supports 64-bit indexing
        with torch.backends.cudnn.flags(enabled=False):
            img = F.grid_sample(img, grid, align_corners=True)
    else:
        img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(
        torch.arange(ht, device=device), torch.arange(wd, device=device), indexing="ij"
    )
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow(flow, factor, mode="bilinear"):
    """Bilinearly upsample a flow field by ``factor`` and scale magnitudes accordingly."""
    new_size = (factor * flow.shape[2], factor * flow.shape[3])
    return factor * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)
