"""
Model helpers for RAFT-DIC
- RAFT args shim
- Model loading
- Inference wrapper
"""

import os
import sys
import torch
import numpy as np

# Ensure the repository's core/ directory is importable for RAFT submodules
_here = os.path.dirname(__file__)
_repo_root = os.path.abspath(os.path.join(_here, os.pardir))
_core_path = os.path.join(_repo_root, 'core')
if _core_path not in sys.path:
    sys.path.insert(0, _core_path)

from core.raft import RAFT
from core.utils.utils import InputPadder


class Args:
    def __init__(self, model: str = '', path: str = '', small: bool = False,
                 mixed_precision: bool = True, alternate_corr: bool = False):
        self.model = model
        self.path = path
        self.small = small
        self.mixed_precision = mixed_precision
        self.alternate_corr = alternate_corr

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration


def process_img(img: np.ndarray, device: str):
    """Convert numpy HxWx3 RGB uint8 to torch [1,3,H,W] float on device."""
    return torch.from_numpy(img).permute(2, 0, 1).float()[None].to(device)


def load_model(weights_path: str, args: Args, weights_only: bool = True):
    """Load RAFT with weights and place on CUDA (required by app)."""
    model = RAFT(args)
    pretrained_weights = torch.load(weights_path, map_location=torch.device("cpu"))
    model = torch.nn.DataParallel(model)
    model.load_state_dict(pretrained_weights)
    model.to("cuda")
    return model


def inference(model, frame1, frame2, device: str, pad_mode: str = 'sintel',
              iters: int = 12, flow_init=None, upsample: bool = True, test_mode: bool = True):
    """Run RAFT inference for a pair and crop to original size."""
    model.eval()
    with torch.no_grad():
        frame1 = process_img(frame1, device)
        frame2 = process_img(frame2, device)

        original_size = (frame1.shape[2], frame1.shape[3])

        padder = InputPadder(frame1.shape, mode=pad_mode)
        frame1, frame2 = padder.pad(frame1, frame2)

        with torch.amp.autocast('cuda', enabled=True):
            if test_mode:
                flow_low, flow_up = model(frame1, frame2, iters=iters,
                                          flow_init=flow_init,
                                          upsample=upsample,
                                          test_mode=test_mode)

                flow_up = flow_up[:, :, :original_size[0], :original_size[1]]
                flow_low = flow_low[:, :, :original_size[0]//8, :original_size[1]//8]
                return flow_low, flow_up
            else:
                flow_iters = model(frame1, frame2, iters=iters,
                                   flow_init=flow_init,
                                   upsample=upsample,
                                   test_mode=test_mode)
                return flow_iters
