"""The released models and where their weights are found.

The four models share one architecture and differ only in the encoder downsampling factor
``s``. A smaller ``s`` resolves shorter wavelengths; a larger ``s`` reaches further: the
coarsest correlation level looks ``32 s`` px away (the nominal lookup radius).

Weight resolution order for a model:

1. an explicit file path;
2. ``$RAFTCORR_WEIGHTS/<filename>``;
3. ``<checkout>/models/<filename>`` (a development checkout);
4. the per-user weights folder (:func:`user_weights_dir`);
5. a download from :data:`RELEASE_BASE_URL` when one is configured.

A file whose SHA-256 differs from the recorded release hash is reported and skipped, never
used silently. Optional *model cards* (``<filename stem>.json`` next to the weights) carry
measured performance hints shown in the GUI.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..callbacks import Callbacks, coerce

ENV_WEIGHTS = "RAFTCORR_WEIGHTS"

#: The weights are assets of this GitHub release of the public repository (published with the
#: RAFTcorr paper). Until then a download reports that they are not published yet.
RELEASE_TAG = "weights-v1"
RELEASE_BASE_URL: str | None = f"https://github.com/zachtong/RAFTcorr/releases/download/{RELEASE_TAG}"


class WeightsUnavailableError(RuntimeError):
    """The weights of a model could not be downloaded (not published yet, or no connection)."""


@dataclass(frozen=True)
class ModelSpec:
    """One released model."""

    key: str                 # "s4"
    downsample_factor: int
    filename: str
    sha256: str | None
    summary: str

    @property
    def name(self) -> str:
        return f"RAFTcorr-{self.key}"

    @property
    def lookup_radius_px(self) -> int:
        """Nominal reach of the coarsest correlation level (4 levels, radius 4): ``32 s`` px."""
        return 32 * self.downsample_factor

    @property
    def url(self) -> str | None:
        return None if RELEASE_BASE_URL is None else RELEASE_BASE_URL.rstrip("/") + "/" + self.filename


@dataclass(frozen=True)
class ModelCard:
    """Measured hints for one model (all optional; read from a JSON sidecar)."""

    spatial_resolution_px: float | None = None
    translation_capture_px: float | None = None
    rotation_capture_deg: float | None = None
    notes: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelCard":
        known = {"spatial_resolution_px", "translation_capture_px", "rotation_capture_deg", "notes"}
        return cls(
            spatial_resolution_px=_opt_float(d.get("spatial_resolution_px")),
            translation_capture_px=_opt_float(d.get("translation_capture_px")),
            rotation_capture_deg=_opt_float(d.get("rotation_capture_deg")),
            notes=tuple(str(n) for n in d.get("notes", ())),
            extra={k: v for k, v in d.items() if k not in known},
        )


def _opt_float(x: Any) -> float | None:
    return None if x is None else float(x)


MODELS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        ModelSpec("s1", 1, "RAFTcorr-s1.pth",
                  "e1745ace1238889e3625d46d2003d5d7e29881b11db409b7d017ee1f1e0acbca",
                  "finest spatial resolution; small displacements"),
        ModelSpec("s2", 2, "RAFTcorr-s2.pth",
                  "5e67f70f6465cf469bdae0a21a97e6b2f98191c7f64de7fcbdf201fbd1161c23",
                  "fine resolution, several times faster than s1; small to moderate displacements"),
        ModelSpec("s4", 4, "RAFTcorr-s4.pth",
                  "373cecf78a5c815c396eb1a61e4fc69a006520c6fc5017295c76e92393793b41",
                  "balanced resolution and capture range"),
        ModelSpec("s8", 8, "RAFTcorr-s8.pth",
                  "238ec2ae1d53c8a2aa1fd0ac7e7f97a3710350ecc7ddf47fb37d0809432cde03",
                  "largest capture range, coarsest resolution; large motion and rotation"),
    )
}

MODEL_KEYS: tuple[str, ...] = tuple(MODELS)


def get_spec(key: str) -> ModelSpec:
    """Spec of a released model by key (``"s4"``) or name (``"RAFTcorr-s4"``)."""
    k = key.removeprefix("RAFTcorr-")
    try:
        return MODELS[k]
    except KeyError:
        raise KeyError(f"unknown model {key!r}; available: {', '.join(MODEL_KEYS)}") from None


def spec_for_factor(downsample_factor: int) -> ModelSpec | None:
    for spec in MODELS.values():
        if spec.downsample_factor == downsample_factor:
            return spec
    return None


def spec_for_sha256(digest: str) -> ModelSpec | None:
    for spec in MODELS.values():
        if spec.sha256 == digest:
            return spec
    return None


# -- locations ------------------------------------------------------------------------------------
def user_weights_dir() -> Path:
    """Per-user folder for downloaded or installed weights."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "RAFTcorr" / "models"
    return Path.home() / ".raftcorr" / "models"


def checkout_models_dir() -> Path | None:
    """``<checkout>/models`` when running from a source checkout (``src/raftcorr/models/zoo.py``)."""
    here = Path(__file__).resolve()
    if len(here.parents) < 4:
        return None
    root = here.parents[3]
    if (root / "pyproject.toml").is_file() and (root / "models").is_dir():
        return root / "models"
    return None


def search_dirs(weights_dir: str | Path | None = None) -> list[Path]:
    out: list[Path] = []
    if weights_dir:
        out.append(Path(weights_dir))
    env = os.environ.get(ENV_WEIGHTS)
    if env:
        out.append(Path(env))
    checkout = checkout_models_dir()
    if checkout is not None:
        out.append(checkout)
    out.append(user_weights_dir())
    uniq: list[Path] = []
    for p in out:
        if p not in uniq:
            uniq.append(p)
    return uniq


def sha256_of(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def find_weights(spec: ModelSpec, weights_dir: str | Path | None = None, callbacks: Callbacks | None = None) -> Path | None:
    """First verified local copy of a model's weights, or None."""
    cb = coerce(callbacks)
    for d in search_dirs(weights_dir):
        p = d / spec.filename
        if not p.is_file():
            continue
        if spec.sha256 is None or sha256_of(p) == spec.sha256:
            return p
        cb.log(f"{p} exists but its SHA-256 does not match the {spec.name} release; skipping it", "warning")
    return None


def download(spec: ModelSpec, dest_dir: Path | None = None, callbacks: Callbacks | None = None,
             timeout: float = 60.0) -> Path:
    """Download a model's weights (atomic rename) and verify the hash.

    Raises :class:`WeightsUnavailableError` when the release has no such file (not published yet)
    or the server cannot be reached, with a message that says where to put the file instead.
    """
    cb = coerce(callbacks)
    dest = (dest_dir or user_weights_dir()) / spec.filename
    if spec.url is None:
        raise WeightsUnavailableError(
            f"no download location is configured for {spec.filename}; place it in {dest.parent}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    cb.log(f"downloading {spec.url}", "info")
    req = urllib.request.Request(spec.url, headers={"User-Agent": "raftcorr"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
            total = int(resp.headers.get("Content-Length") or 0)
            got = 0
            while chunk := resp.read(1 << 18):
                fh.write(chunk)
                got += len(chunk)
                if total:
                    cb.progress(min(1.0, got / total), f"downloading {spec.filename}: {got / 1e6:.1f} MB")
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        if exc.code == 404:
            raise WeightsUnavailableError(
                f"{spec.filename} is not published yet (the model files are released with the RAFTcorr "
                f"paper); if you have the file, place it in {dest.parent}") from exc
        raise WeightsUnavailableError(f"downloading {spec.filename} failed (HTTP {exc.code}); "
                                      f"place the file in {dest.parent}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise WeightsUnavailableError(f"cannot download {spec.filename} ({exc}); check the internet "
                                      f"connection, or place the file in {dest.parent}") from exc
    if spec.sha256 is not None and sha256_of(tmp) != spec.sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded {spec.filename} failed the SHA-256 check; refusing to use it")
    shutil.move(str(tmp), str(dest))
    cb.log(f"saved {dest}", "info")
    return dest


def resolve_weights(key_or_path: str | Path, weights_dir: str | Path | None = None,
                    allow_download: bool = False, callbacks: Callbacks | None = None) -> Path:
    """Path of the weights for a model key (``"s4"``) or an explicit checkpoint path.

    With *allow_download* a model without a verified local copy is downloaded from the release
    into the per-user folder; a FileNotFoundError says why when that is not possible either.
    """
    p = Path(key_or_path)
    if p.suffix in (".pth", ".pt") or p.is_file():
        if not p.is_file():
            raise FileNotFoundError(f"checkpoint not found: {p}")
        return p
    spec = get_spec(str(key_or_path))
    found = find_weights(spec, weights_dir, callbacks)
    if found is not None:
        return found
    looked = ", ".join(str(d) for d in search_dirs(weights_dir))
    if allow_download and spec.url is not None:
        try:
            return download(spec, callbacks=callbacks)
        except WeightsUnavailableError as exc:
            raise FileNotFoundError(f"weights for {spec.name} not found locally (looked in: {looked}) "
                                    f"and not downloaded: {exc}") from exc
    raise FileNotFoundError(
        f"weights for {spec.name} ({spec.filename}) not found; looked in: {looked}. "
        f"Place the file in one of these folders or set {ENV_WEIGHTS}."
    )


def download_missing(keys: list[str] | None = None, callbacks: Callbacks | None = None) -> dict[str, str]:
    """Download every model (or *keys*) without a verified local copy into the per-user folder.

    Returns ``key -> outcome`` ("installed: <path>", "downloaded: <path>" or the reason it failed);
    one failure does not stop the others.
    """
    cb = coerce(callbacks)
    out: dict[str, str] = {}
    for key in keys or list(MODELS):
        spec = get_spec(key)
        found = find_weights(spec, callbacks=cb)
        if found is not None:
            out[spec.key] = f"installed: {found}"
            continue
        try:
            out[spec.key] = f"downloaded: {download(spec, callbacks=cb)}"
        except (WeightsUnavailableError, RuntimeError) as exc:
            out[spec.key] = str(exc)
    return out


def available_models(weights_dir: str | Path | None = None) -> dict[str, Path | None]:
    """Model key -> local weights path (None when missing). Does not hash (fast)."""
    out: dict[str, Path | None] = {}
    for key, spec in MODELS.items():
        out[key] = next((d / spec.filename for d in search_dirs(weights_dir) if (d / spec.filename).is_file()), None)
    return out


def load_card(weights_path: str | Path) -> ModelCard:
    """Model card next to a weights file (``RAFTcorr-s4.json``), or an empty card."""
    card = Path(weights_path).with_suffix(".json")
    if not card.is_file():
        return ModelCard()
    try:
        return ModelCard.from_dict(json.loads(card.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return ModelCard()
