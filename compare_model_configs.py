import math
import os
from collections import OrderedDict
from typing import Dict, Tuple

import torch


def load_state(path: str) -> OrderedDict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def infer_corr_settings(in_channels: int) -> Tuple[int, int]:
    for radius in range(7, 0, -1):
        base = (2 * radius + 1) ** 2
        if in_channels % base == 0:
            levels = in_channels // base
            return levels, radius
    return -1, -1


def summarize(state_a: OrderedDict, state_b: OrderedDict) -> Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    shared_keys = state_a.keys() & state_b.keys()
    diff = {}
    for key in sorted(shared_keys):
        shape_a = tuple(state_a[key].shape)
        shape_b = tuple(state_b[key].shape)
        if shape_a != shape_b:
            diff[key] = (shape_a, shape_b)
    return diff


def main() -> None:
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    ckpt_v1 = os.path.join(models_dir, "raft-dic_v1.pth")
    ckpt_v2 = os.path.join(models_dir, "raft-dic_v2.pth")

    state_v1 = load_state(ckpt_v1)
    state_v2 = load_state(ckpt_v2)

    print(f"Loaded checkpoints: v1 ({len(state_v1)} tensors), v2 ({len(state_v2)} tensors)")

    diffs = summarize(state_v1, state_v2)
    print(f"Parameters with differing shapes: {len(diffs)}")
    for name, (shape_v1, shape_v2) in list(diffs.items())[:10]:
        print(f"  {name}: v1 {shape_v1} vs v2 {shape_v2}")
    if len(diffs) > 10:
        print(f"  ... {len(diffs) - 10} more entries differ")

    conv_key = "module.update_block.encoder.convc1.weight"
    if conv_key in state_v1 and conv_key in state_v2:
        in_v1 = state_v1[conv_key].shape[1]
        in_v2 = state_v2[conv_key].shape[1]
        levels_v1, radius_v1 = infer_corr_settings(in_v1)
        levels_v2, radius_v2 = infer_corr_settings(in_v2)
        print("\nInferred correlation settings from motion encoder:")
        print(f"  v1: in_channels={in_v1}, corr_levels={levels_v1}, corr_radius={radius_v1}")
        print(f"  v2: in_channels={in_v2}, corr_levels={levels_v2}, corr_radius={radius_v2}")
    else:
        print(f"\nKey not found in both checkpoints: {conv_key}")

    only_v1 = sorted(state_v1.keys() - state_v2.keys())
    only_v2 = sorted(state_v2.keys() - state_v1.keys())
    if only_v1:
        print(f"\nKeys only in v1 ({len(only_v1)}): {only_v1[:5]}" + (" ..." if len(only_v1) > 5 else ""))
    if only_v2:
        print(f"Keys only in v2 ({len(only_v2)}): {only_v2[:5]}" + (" ..." if len(only_v2) > 5 else ""))


if __name__ == "__main__":
    main()
