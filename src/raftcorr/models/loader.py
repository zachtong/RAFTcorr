"""Strict checkpoint loading: the only way a trained network enters the application.

A checkpoint must be a dict with the weights under ``"model"`` and the architecture under
``"model_cfg"``. The architecture is never guessed from tensor shapes: stride leaves no trace in
the convolution weights, and guessing once shipped a randomly initialised network. Every key of
the architecture must be present in the file and every key of the file must belong to the
architecture. The file is read with ``weights_only=True`` so a checkpoint cannot execute code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..callbacks import Callbacks, coerce
from .zoo import ModelCard, ModelSpec, load_card, sha256_of, spec_for_sha256

LEGACY_HINT = (
    "RAFTcorr 2 loads only checkpoints that store their architecture (model_cfg), such as the "
    "released RAFTcorr-s1, -s2, -s4 and -s8 models. The 1.x models (RAFTcorr-L, RAFTcorr-Fine, "
    "RAFTcorr_large_*, RAFTcorr_fine_*) are not supported."
)


class CheckpointError(ValueError):
    """The file is not a usable RAFTcorr checkpoint."""


@dataclass(frozen=True)
class LoadedModel:
    """A network on its device, in eval mode, with its provenance."""

    network: Any            # raftcorr.nn.RAFT
    path: str
    sha256: str
    device: str
    spec: ModelSpec | None  # the released model this file is, if any
    card: ModelCard
    step: int | None = None

    @property
    def downsample_factor(self) -> int:
        return int(self.network.cfg.downsample_factor)

    @property
    def lookup_radius_px(self) -> int:
        return int(self.network.cfg.lookup_radius_px)

    @property
    def corr_mode(self) -> str:
        return str(self.network.corr_mode)

    @property
    def name(self) -> str:
        if self.spec is not None:
            return self.spec.name
        return f"custom (s={self.downsample_factor}, {Path(self.path).name})"

    @property
    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.network.parameters()))

    def model_cfg(self) -> dict[str, Any]:
        return self.network.cfg.to_dict()


def resolve_device(device: str = "auto") -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' was requested but CUDA is not available; use 'cpu'")
    return device


def read_checkpoint(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return ``(model_cfg, state_dict, extras)`` of a checkpoint file, or raise CheckpointError."""
    import torch

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    try:
        ck = torch.load(p, map_location="cpu", weights_only=True)
    except Exception as exc:  # unpickling refused or corrupt file
        raise CheckpointError(f"{p.name}: cannot be read as a weights-only PyTorch checkpoint ({exc}). {LEGACY_HINT}") from exc
    if not isinstance(ck, dict) or "model" not in ck or "model_cfg" not in ck:
        raise CheckpointError(f"{p.name}: no model_cfg in the checkpoint. {LEGACY_HINT}")
    cfg = ck["model_cfg"]
    sd = ck["model"]
    if not isinstance(cfg, dict) or not isinstance(sd, dict):
        raise CheckpointError(f"{p.name}: malformed checkpoint (model_cfg or model is not a dict)")
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    extras = {k: v for k, v in ck.items() if k not in ("model", "model_cfg")}
    return dict(cfg), sd, extras


def load_model(
    path: str | Path,
    device: str = "auto",
    corr_mode: str | None = None,
    callbacks: Callbacks | None = None,
) -> LoadedModel:
    """Load a checkpoint, verify it against its declared architecture, move it to ``device``."""
    from ..nn import build_raft

    cb = coerce(callbacks)
    p = Path(path)
    cfg, sd, extras = read_checkpoint(p)
    digest = sha256_of(p)
    try:
        network = build_raft(cfg, corr_mode=corr_mode)
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"{p.name}: unsupported model_cfg {cfg!r} ({exc})") from exc

    expected = set(network.state_dict().keys())
    got = set(sd.keys())
    missing, unexpected = sorted(expected - got), sorted(got - expected)
    if missing or unexpected:
        raise CheckpointError(
            f"{p.name}: weights do not match the declared architecture "
            f"(missing {len(missing)} e.g. {missing[:3]}, unexpected {len(unexpected)} e.g. {unexpected[:3]})"
        )
    try:
        network.load_state_dict(sd, strict=True)
    except RuntimeError as exc:  # shape mismatch
        raise CheckpointError(f"{p.name}: weights do not match the declared architecture ({exc})") from exc

    dev = resolve_device(device)
    network = network.to(dev).eval()
    spec = spec_for_sha256(digest)
    step = extras.get("step")
    loaded = LoadedModel(
        network=network,
        path=str(p),
        sha256=digest,
        device=dev,
        spec=spec,
        card=load_card(p),
        step=int(step) if isinstance(step, int) else None,
    )
    cb.log(
        f"loaded {loaded.name} (s={loaded.downsample_factor}, {loaded.n_parameters / 1e6:.2f} M parameters, "
        f"correlation '{loaded.corr_mode}', {dev})",
        "info",
    )
    return loaded
