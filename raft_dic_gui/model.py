"""
Model helpers for RAFT-DIC
- RAFT args shim
- Model loading
- Inference wrapper
"""

import json
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure the repository's core/ directory is importable for RAFT submodules
_here = os.path.dirname(__file__)
_repo_root = os.path.abspath(os.path.join(_here, os.pardir))
_core_path = os.path.join(_repo_root, 'core')
if _core_path not in sys.path:
    sys.path.insert(0, _core_path)

_models_path = os.path.join(_repo_root, 'models')

from core.raft import RAFT
from core.utils.utils import InputPadder


VARIANT_PYRAMID = "pyramid"
VARIANT_FULL_RES = "full_res"
_KNOWN_VARIANTS = {VARIANT_PYRAMID, VARIANT_FULL_RES}


@dataclass(frozen=True)
class ModelEntry:
    path: str
    label: str


@dataclass(frozen=True)
class ModelMetadata:
    path: str
    label: str
    small: bool
    mixed_precision: bool
    alternate_corr: bool
    corr_levels: Optional[int]
    corr_radius: Optional[int]
    dropout: float = 0.0
    strict: bool = False
    data_parallel: Optional[bool] = None
    variant: str = VARIANT_PYRAMID
    full_resolution: bool = False

    def summary(self) -> str:
        arch = "small" if self.small else "large"
        mp = "on" if self.mixed_precision else "off"
        alt = "on" if self.alternate_corr else "off"
        if self.corr_levels is None or self.corr_radius is None:
            corr = "corr unknown"
        else:
            corr = f"corr {self.corr_levels}A-r{self.corr_radius}"
        variant_label = "full-res" if self.full_resolution else "pyramid"
        return f"Arch {arch} | {corr} | {variant_label} | mixed precision {mp} | alternate corr {alt}"


DEFAULT_DEVICE = "cuda"
DEFAULT_CORR_LEVELS: Optional[int] = None
DEFAULT_CORR_RADIUS: Optional[int] = None
CONFIG_FILE_NAME = "model_config.json"

_MODEL_CACHE: Dict[Tuple[str, str], torch.nn.Module] = {}
_USER_OVERRIDE_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


class Args:
    def __init__(self, model: str = '', path: str = '', small: bool = False,
                 mixed_precision: bool = True, alternate_corr: bool = False,
                 corr_levels: Optional[int] = DEFAULT_CORR_LEVELS, corr_radius: Optional[int] = DEFAULT_CORR_RADIUS,
                 dropout: float = 0.0, strict: bool = False, variant: str = VARIANT_PYRAMID,
                 full_resolution: Optional[bool] = None):
        self.model = model
        self.path = path
        self.small = small
        self.mixed_precision = mixed_precision
        self.alternate_corr = alternate_corr
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.dropout = dropout
        self.strict = strict
        normalized_variant = variant.lower().replace('-', '_') if isinstance(variant, str) else VARIANT_PYRAMID
        if normalized_variant not in _KNOWN_VARIANTS:
            normalized_variant = VARIANT_PYRAMID
        if full_resolution is None:
            full_resolution = (normalized_variant == VARIANT_FULL_RES)
        else:
            full_resolution = bool(full_resolution)
            if full_resolution:
                normalized_variant = VARIANT_FULL_RES
        self.variant = normalized_variant
        self.full_resolution = bool(full_resolution)

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration


def _tokens_from_part(part: str) -> List[str]:
    sanitized = part.replace('-', '_').replace('.', '_')
    return [token for token in sanitized.split('_') if token]


def _normalize_variant(value: Optional[Any]) -> Optional[str]:
    if value is None:
        return None
    token = str(value).strip().lower().replace('-', '_').replace(' ', '_')
    token = token.replace('resolution', 'res')
    if token in {"fullres", "full_res", "fullresstride1", "full"}:
        return VARIANT_FULL_RES
    if token in {"pyramid", "py", "pyr"}:
        return VARIANT_PYRAMID
    return None


def _guess_hints_from_path(path: str) -> Dict[str, Any]:
    hints: Dict[str, Any] = {}
    parts = [p.lower() for p in Path(path).parts[-3:]]
    for part in parts:
        tokens = _tokens_from_part(part)
        if 'small' in tokens:
            hints['small'] = True
        elif 'large' in tokens:
            hints['small'] = False

        if 'fullres' in tokens or {'full', 'res'}.issubset(set(tokens)):
            hints['variant'] = VARIANT_FULL_RES
        if 'pyramid' in tokens:
            hints['variant'] = VARIANT_PYRAMID

        if 'fp32' in tokens:
            hints['mixed_precision'] = False
        if 'fp16' in tokens or {'mixed', 'precision'}.issubset(tokens) or 'mixedprecision' in tokens:
            hints['mixed_precision'] = True

        if 'no' in tokens and 'alt' in tokens:
            hints['alternate_corr'] = False
        elif 'alt' in tokens:
            hints.setdefault('alternate_corr', True)

        for token in tokens:
            if token.startswith('dropout'):
                try:
                    value = token.replace('dropout', '')
                    if value:
                        hints['dropout'] = float(value)
                except ValueError:
                    pass

        if 'strict' in tokens:
            hints['strict'] = True
    return hints


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _coerce_int(value: Any, default: Optional[int]) -> Optional[int]:
    if value is None:
        return default
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            pass
    return default


def _coerce_float(value: Any, default: Optional[float]) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            pass
    return default


def infer_corr_settings(in_channels: int) -> Tuple[int, int]:
    for radius in range(7, 0, -1):
        base = (2 * radius + 1) ** 2
        if in_channels % base == 0:
            levels = in_channels // base
            return levels, radius
    return -1, -1


def _load_user_overrides() -> Dict[str, Dict[str, Any]]:
    global _USER_OVERRIDE_CACHE
    if _USER_OVERRIDE_CACHE is not None:
        return _USER_OVERRIDE_CACHE

    overrides: Dict[str, Dict[str, Any]] = {}
    config_path = os.path.join(_models_path, CONFIG_FILE_NAME)
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                overrides = {str(k): v for k, v in data.items() if isinstance(v, dict)}
        except Exception as exc:
            print(f"[model] Warning: failed to read {CONFIG_FILE_NAME}: {exc}")
    _USER_OVERRIDE_CACHE = overrides
    return overrides


@lru_cache(maxsize=128)
def _load_sidecar_overrides(path: str) -> Dict[str, Any]:
    candidate = Path(path).with_suffix('.json')
    if not candidate.is_file():
        return {}
    try:
        with open(candidate, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except Exception as exc:
        print(f"[model] Warning: failed to read sidecar config {candidate.name}: {exc}")
    return {}


def _find_override_for_path(path: str) -> Dict[str, Any]:
    overrides = _load_user_overrides()
    if not overrides:
        return {}

    abs_path = os.path.abspath(path)
    candidates: List[str] = [Path(abs_path).name]
    try:
        rel = Path(abs_path).relative_to(_models_path).as_posix()
        candidates.insert(0, rel)
    except ValueError:
        pass

    for candidate in candidates:
        if candidate in overrides:
            return overrides[candidate]
    return {}


@lru_cache(maxsize=32)
def _inspect_checkpoint(abs_path: str) -> Dict[str, Any]:
    state_dict = torch.load(abs_path, map_location=torch.device("cpu"))
    info: Dict[str, Any] = {}
    try:
        keys = list(state_dict.keys())
        info['data_parallel'] = bool(keys and keys[0].startswith('module.'))

        def lookup(*names: str):
            for name in names:
                if name in state_dict:
                    return state_dict[name]
            return None

        conv1 = lookup('module.fnet.conv1.weight', 'fnet.conv1.weight',
                       'module.fnet.layer0.0.conv1.weight', 'fnet.layer0.0.conv1.weight')
        if conv1 is not None and conv1.ndim >= 1:
            out_channels = conv1.shape[0]
            if out_channels in (32, 48):
                info['small'] = True
            elif out_channels in (64, 96):
                info['small'] = False

        convc1 = lookup('module.update_block.encoder.convc1.weight',
                        'update_block.encoder.convc1.weight')
        if convc1 is not None and convc1.ndim >= 2:
            in_channels = int(convc1.shape[1])
            levels, radius = infer_corr_settings(in_channels)
            if levels > 0:
                info['corr_levels'] = levels
                info['corr_radius'] = radius
                if 'variant' not in info:
                    info['variant'] = VARIANT_FULL_RES if levels <= 2 else VARIANT_PYRAMID
                if 'full_resolution' not in info:
                    info['full_resolution'] = bool(levels <= 2)
    finally:
        # Free references eagerly
        del state_dict
    return info


def _resolve_bool(info_value: Any, hint_value: Any, default: bool) -> bool:
    if info_value is not None:
        return _coerce_bool(info_value, default)
    if hint_value is not None:
        return _coerce_bool(hint_value, default)
    return default


def _resolve_int(info_value: Any, hint_value: Any, default: Optional[int]) -> Optional[int]:
    if info_value is not None:
        return _coerce_int(info_value, default)
    if hint_value is not None:
        return _coerce_int(hint_value, default)
    return default


def _resolve_float(info_value: Any, hint_value: Any, default: Optional[float]) -> Optional[float]:
    if info_value is not None:
        return _coerce_float(info_value, default)
    if hint_value is not None:
        return _coerce_float(hint_value, default)
    return default


def describe_checkpoint(weights_path: str,
                        overrides: Optional[Dict[str, Any]] = None) -> ModelMetadata:
    abs_path = os.path.abspath(weights_path)
    info = _inspect_checkpoint(abs_path)
    hints = _guess_hints_from_path(abs_path)
    if overrides:
        hints.update(overrides)
    hints.update(_find_override_for_path(abs_path))
    hints.update(_load_sidecar_overrides(abs_path))

    small = _resolve_bool(info.get('small'), hints.get('small'), default=False)
    mixed_precision = _resolve_bool(info.get('mixed_precision'), hints.get('mixed_precision'), default=True)
    alternate_corr = _resolve_bool(info.get('alternate_corr'), hints.get('alternate_corr'), default=False)
    corr_levels = _resolve_int(info.get('corr_levels'), hints.get('corr_levels'), default=None)
    corr_radius = _resolve_int(info.get('corr_radius'), hints.get('corr_radius'), default=None)
    dropout = _resolve_float(info.get('dropout'), hints.get('dropout'), 0.0)
    strict = _resolve_bool(info.get('strict'), hints.get('strict'), default=False)

    variant = _normalize_variant(hints.get('variant')) or _normalize_variant(info.get('variant'))
    if variant not in _KNOWN_VARIANTS:
        variant = None

    full_resolution_default = (variant == VARIANT_FULL_RES) if variant else False
    full_resolution = _resolve_bool(info.get('full_resolution'), hints.get('full_resolution'),
                                    default=full_resolution_default)
    if full_resolution:
        variant = VARIANT_FULL_RES
    if variant is None:
        variant = VARIANT_FULL_RES if full_resolution else VARIANT_PYRAMID
    if variant not in _KNOWN_VARIANTS:
        variant = VARIANT_PYRAMID
        full_resolution = False
    else:
        full_resolution = (variant == VARIANT_FULL_RES)

    data_parallel = info.get('data_parallel')
    if data_parallel is not None:
        data_parallel = bool(data_parallel)

    try:
        label = Path(abs_path).relative_to(_models_path).as_posix()
    except ValueError:
        label = Path(abs_path).name

    return ModelMetadata(
        path=abs_path,
        label=label,
        small=small,
        mixed_precision=mixed_precision,
        alternate_corr=alternate_corr,
        corr_levels=corr_levels,
        corr_radius=corr_radius,
        dropout=dropout,
        strict=strict,
        data_parallel=data_parallel,
        variant=variant,
        full_resolution=full_resolution
    )


def metadata_summary(metadata: ModelMetadata) -> str:
    return metadata.summary()


def discover_models(models_dir: Optional[str] = None,
                    recursive: bool = True) -> List[ModelEntry]:
    root = Path(models_dir or _models_path)
    if not root.exists():
        return []

    pattern = "**/*.pth" if recursive else "*.pth"
    candidates: List[ModelEntry] = []
    for path in root.glob(pattern):
        if not path.is_file():
            continue
        try:
            label = path.relative_to(root).as_posix()
        except ValueError:
            label = path.name
        candidates.append(ModelEntry(path=str(path), label=label))

    candidates.sort(key=lambda entry: entry.label.lower())
    return candidates


def build_args_from_metadata(metadata: ModelMetadata, base_args: Optional[Args] = None) -> Args:
    args = base_args or Args()
    args.model = metadata.path
    args.path = os.path.dirname(metadata.path)
    args.small = metadata.small
    args.mixed_precision = metadata.mixed_precision
    args.alternate_corr = metadata.alternate_corr
    args.corr_levels = metadata.corr_levels if metadata.corr_levels is not None else None
    args.corr_radius = metadata.corr_radius if metadata.corr_radius is not None else None
    args.dropout = metadata.dropout
    args.strict = metadata.strict
    args.variant = metadata.variant
    args.full_resolution = metadata.full_resolution
    return args


def process_img(img: np.ndarray, device: str):
    """Convert numpy HxWx3 RGB uint8 to torch [1,3,H,W] float on device."""
    return torch.from_numpy(img).permute(2, 0, 1).float()[None].to(device)


def load_model(weights_path: str,
               args: Optional[Args] = None,
               weights_only: bool = True,
               device: str = DEFAULT_DEVICE,
               metadata: Optional[ModelMetadata] = None,
               strict: Optional[bool] = None,
               overrides: Optional[Dict[str, Any]] = None,
               reuse_cached: bool = True):
    """Load RAFT with weights and place on the requested device."""
    if not weights_path:
        raise ValueError("Model checkpoint path is empty. Please select a model.")
    if os.path.isdir(weights_path):
        raise ValueError(f"Model path is a directory, expected a file: {weights_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model checkpoint not found: {weights_path}")
        
    metadata = metadata or describe_checkpoint(weights_path, overrides=overrides)
    device = device or DEFAULT_DEVICE
    if strict is None:
        strict = metadata.strict

    cache_key = (metadata.path, device)
    if reuse_cached and cache_key in _MODEL_CACHE:
        model = _MODEL_CACHE[cache_key]
        return model

    args = build_args_from_metadata(metadata, base_args=args)
    model = RAFT(args)

    pretrained_weights = torch.load(metadata.path, map_location=torch.device("cpu"))

    keys = list(pretrained_weights.keys())
    is_data_parallel = metadata.data_parallel
    if is_data_parallel is None:
        is_data_parallel = bool(keys and keys[0].startswith('module.'))

    if is_data_parallel:
        model = torch.nn.DataParallel(model)
        load_result = model.load_state_dict(pretrained_weights, strict=strict)
    else:
        new_state = {}
        for k, v in pretrained_weights.items():
            prefixed = k if k.startswith('module.') else f"module.{k}"
            new_state[prefixed] = v
        model = torch.nn.DataParallel(model)
        load_result = model.load_state_dict(new_state, strict=strict)

    del pretrained_weights

    if strict:
        missing_keys = []
        unexpected_keys = []
    else:
        missing_keys = list(getattr(load_result, "missing_keys", []))
        unexpected_keys = list(getattr(load_result, "unexpected_keys", []))

    if not strict and (missing_keys or unexpected_keys):
        print(f"[model] Loaded checkpoint with strict={strict}")
        if unexpected_keys:
            print(f"[model]   Ignored {len(unexpected_keys)} unexpected keys")
        if missing_keys:
            print(f"[model]   Warning: {len(missing_keys)} keys missing (using default init)")

    model.to(device)
    model.eval()
    setattr(model, "_raft_dic_metadata", metadata)

    _MODEL_CACHE[cache_key] = model
    return model


def inference(model, frame1, frame2, device: str, pad_mode: str = 'sintel',
              iters: int = 12, flow_init=None, upsample: bool = True, test_mode: bool = True):
    """Run RAFT inference for a pair and crop to original size."""
    model.eval()
    mp_enabled = True
    try:
        raft_obj = getattr(model, 'module', model)
        mp_enabled = bool(getattr(raft_obj.args, "mixed_precision", True))
    except Exception:
        mp_enabled = True
    is_cuda = device.startswith("cuda")
    autocast_context = nullcontext()
    if is_cuda and mp_enabled:
        try:
            autocast_context = torch.amp.autocast(device_type='cuda', enabled=True)
        except Exception:
            autocast_context = nullcontext()

    with torch.no_grad():
        frame1 = process_img(frame1, device)
        frame2 = process_img(frame2, device)

        original_size = (frame1.shape[2], frame1.shape[3])

        padder = InputPadder(frame1.shape, mode=pad_mode)
        frame1, frame2 = padder.pad(frame1, frame2)

        raft_args = getattr(getattr(model, 'module', model), 'args', None)
        is_full_res = bool(getattr(raft_args, 'full_resolution', False))

        with autocast_context:
            if test_mode:
                flow_low, flow_up = model(frame1, frame2, iters=iters,
                                          flow_init=flow_init,
                                          upsample=upsample,
                                          test_mode=test_mode)

                flow_up = flow_up[:, :, :original_size[0], :original_size[1]]
                if is_full_res:
                    flow_low = flow_low[:, :, :original_size[0], :original_size[1]]
                else:
                    flow_low = flow_low[:, :, :original_size[0]//8, :original_size[1]//8]
                return flow_low, flow_up
            else:
                flow_iters = model(frame1, frame2, iters=iters,
                                   flow_init=flow_init,
                                   upsample=upsample,
                                   test_mode=test_mode)
                return flow_iters



