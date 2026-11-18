# Identical to the research implementation of the RAFTcorr paper (commit 06eb238).
# Derived from RAFT (Teed and Deng, ECCV 2020), BSD 3-Clause License: see LICENSE-RAFT in this folder.
"""Feature/context encoders, unified across downsample factors.

A single encoder supports any ``downsample_factor`` in {1, 2, 4, 8} by distributing the
stride-2 stages. This merges the legacy ``extractor.py`` (factor 8) and
``extractor_without_downsampling.py`` (factor 1) into one implementation and exposes the
intermediate factors (2, 4) needed for study #1 (speckle x downsample coupling).

The residual/bottleneck projection condition is ``stride == 1 and in_planes == planes`` so
that channel changes at stride 1 (which occur at full resolution) are handled correctly.
"""

import torch
import torch.nn as nn

# downsample_factor -> (conv1_stride, layer2_stride, layer3_stride); layer1 is always stride 1.
_FACTOR_STRIDES = {
    1: (1, 1, 1),
    2: (2, 1, 1),
    4: (2, 2, 1),
    8: (2, 2, 2),
}


def strides_for_factor(factor: int):
    if factor not in _FACTOR_STRIDES:
        raise ValueError(
            f"Unsupported downsample_factor={factor}; supported: {sorted(_FACTOR_STRIDES)}"
        )
    return _FACTOR_STRIDES[factor]


def _make_norm(norm_fn, num_channels, num_groups=None):
    if norm_fn == "group":
        return nn.GroupNorm(num_groups=num_groups or num_channels // 8, num_channels=num_channels)
    if norm_fn == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm_fn == "instance":
        return nn.InstanceNorm2d(num_channels)
    return nn.Sequential()


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        needs_proj = not (stride == 1 and in_planes == planes)
        self.norm1 = _make_norm(norm_fn, planes)
        self.norm2 = _make_norm(norm_fn, planes)
        norm3 = _make_norm(norm_fn, planes) if needs_proj else None

        if needs_proj:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), norm3
            )
        else:
            self.downsample = None

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes // 4, planes // 4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes // 4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        needs_proj = not (stride == 1 and in_planes == planes)
        self.norm1 = _make_norm(norm_fn, planes // 4)
        self.norm2 = _make_norm(norm_fn, planes // 4)
        self.norm3 = _make_norm(norm_fn, planes)
        norm4 = _make_norm(norm_fn, planes) if needs_proj else None

        if needs_proj:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), norm4
            )
        else:
            self.downsample = None

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn="batch", dropout=0.0, downsample_factor=8):
        super().__init__()
        self.norm_fn = norm_fn
        s0, s1, s2 = strides_for_factor(downsample_factor)

        self.norm1 = _make_norm(norm_fn, 64, num_groups=8)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=s0, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=s1)
        self.layer3 = self._make_layer(128, stride=s2)
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None
        _init_weights(self)

    def _make_layer(self, dim, stride=1):
        layers = (
            ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride),
            ResidualBlock(dim, dim, self.norm_fn, stride=1),
        )
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        return _encoder_forward(self, x)


class SmallEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn="batch", dropout=0.0, downsample_factor=8):
        super().__init__()
        self.norm_fn = norm_fn
        s0, s1, s2 = strides_for_factor(downsample_factor)

        self.norm1 = _make_norm(norm_fn, 32, num_groups=8)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=s0, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(32, stride=1)
        self.layer2 = self._make_layer(64, stride=s1)
        self.layer3 = self._make_layer(96, stride=s2)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None
        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)
        _init_weights(self)

    def _make_layer(self, dim, stride=1):
        layers = (
            BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride),
            BottleneckBlock(dim, dim, self.norm_fn, stride=1),
        )
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        return _encoder_forward(self, x)


def _init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def _encoder_forward(self, x):
    is_list = isinstance(x, (tuple, list))
    if is_list:
        batch_dim = x[0].shape[0]
        x = torch.cat(x, dim=0)

    x = self.relu1(self.norm1(self.conv1(x)))
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.conv2(x)

    if self.training and self.dropout is not None:
        x = self.dropout(x)

    if is_list:
        x = torch.split(x, [batch_dim, batch_dim], dim=0)
    return x
