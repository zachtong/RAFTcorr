"""The RAFTcorr network: one architecture for the four released models (s = 1, 2, 4, 8).

The network files are copies of the research implementation so that the application and the
paper share one source of truth. Load trained weights through :mod:`raftcorr.models`.
"""

from .corr import CORR_MODES, CorrBlock, default_corr_mode
from .corr_otf import CorrBlockOTF
from .raft import RAFT, SUPPORTED_FACTORS, RAFTConfig


def build_raft(model_cfg: dict, corr_mode: str | None = None) -> RAFT:
    """Construct an (untrained) network from a ``model_cfg`` dict."""
    return RAFT(RAFTConfig.from_dict(model_cfg), corr_mode=corr_mode)


__all__ = [
    "CORR_MODES",
    "CorrBlock",
    "CorrBlockOTF",
    "RAFT",
    "RAFTConfig",
    "SUPPORTED_FACTORS",
    "build_raft",
    "default_corr_mode",
]
