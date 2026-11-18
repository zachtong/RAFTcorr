"""Model weights from the GitHub release: download, verification and the messages when it fails.

The network is never touched: ``urllib.request.urlopen`` is replaced by a fake server.
"""

from __future__ import annotations

import hashlib
import io
import urllib.error
from dataclasses import replace
from pathlib import Path

import pytest

from raftcorr.models import zoo

PAYLOAD = b"not really a checkpoint" * 1000


class _Response(io.BytesIO):
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


@pytest.fixture()
def spec() -> zoo.ModelSpec:
    return replace(zoo.MODELS["s4"], sha256=hashlib.sha256(PAYLOAD).hexdigest())


@pytest.fixture()
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No local weights anywhere: an empty per-user folder and no checkout folder."""
    user = tmp_path / "user"
    monkeypatch.setattr(zoo, "user_weights_dir", lambda: user)
    monkeypatch.setattr(zoo, "checkout_models_dir", lambda: None)
    monkeypatch.delenv(zoo.ENV_WEIGHTS, raising=False)
    return user


def _serve(monkeypatch: pytest.MonkeyPatch, data: bytes | None = None, error: Exception | None = None) -> list[str]:
    asked: list[str] = []

    def fake_urlopen(req, timeout=None):
        asked.append(req.full_url)
        if error is not None:
            raise error
        return _Response(data)

    monkeypatch.setattr(zoo.urllib.request, "urlopen", fake_urlopen)
    return asked


def test_the_release_url_names_the_tag_and_file():
    assert zoo.MODELS["s4"].url == "https://github.com/zachtong/RAFTcorr/releases/download/weights-v1/RAFTcorr-s4.pth"


def test_download_verifies_and_moves_the_file(spec, isolated, monkeypatch):
    asked = _serve(monkeypatch, PAYLOAD)
    path = zoo.download(spec)
    assert asked == [spec.url] and path == isolated / spec.filename
    assert path.read_bytes() == PAYLOAD and not list(isolated.glob("*.part"))


def test_a_corrupted_download_is_refused(spec, isolated, monkeypatch):
    _serve(monkeypatch, PAYLOAD + b"x")
    with pytest.raises(RuntimeError, match="SHA-256"):
        zoo.download(spec)
    assert not any(isolated.iterdir())


def test_a_missing_release_file_says_so(spec, isolated, monkeypatch):
    _serve(monkeypatch, error=urllib.error.HTTPError(spec.url, 404, "Not Found", {}, None))
    with pytest.raises(zoo.WeightsUnavailableError, match="not on the release"):
        zoo.download(spec)
    assert not any(isolated.iterdir())


def test_no_connection_says_so(spec, isolated, monkeypatch):
    _serve(monkeypatch, error=urllib.error.URLError("no route to host"))
    with pytest.raises(zoo.WeightsUnavailableError, match="internet connection"):
        zoo.download(spec)


def test_first_use_downloads_and_failures_name_the_folders(spec, isolated, monkeypatch):
    monkeypatch.setitem(zoo.MODELS, "s4", spec)
    _serve(monkeypatch, PAYLOAD)
    assert zoo.resolve_weights("s4", allow_download=True) == isolated / spec.filename
    (isolated / spec.filename).unlink()
    _serve(monkeypatch, error=urllib.error.HTTPError(spec.url, 404, "Not Found", {}, None))
    with pytest.raises(FileNotFoundError, match="not on the release") as info:
        zoo.resolve_weights("s4", allow_download=True)
    assert str(isolated) in str(info.value)
    with pytest.raises(FileNotFoundError, match="Place the file"):
        zoo.resolve_weights("s4")                     # no download asked: no request either


def test_download_missing_reports_each_model(spec, isolated, monkeypatch):
    monkeypatch.setitem(zoo.MODELS, "s4", spec)
    _serve(monkeypatch, PAYLOAD)
    first = zoo.download_missing(["s4"])
    assert first["s4"].startswith("downloaded: ")
    assert zoo.download_missing(["s4"])["s4"].startswith("installed: ")
    _serve(monkeypatch, error=urllib.error.HTTPError("u", 404, "Not Found", {}, None))
    assert "not on the release" in zoo.download_missing(["s8"])["s8"]


def test_cli_models_download_exit_code(spec, isolated, monkeypatch, capsys):
    from raftcorr.cli.main import main

    monkeypatch.setitem(zoo.MODELS, "s4", spec)
    _serve(monkeypatch, PAYLOAD)
    assert main(["models", "--download", "s4"]) == 0
    _serve(monkeypatch, error=urllib.error.HTTPError("u", 404, "Not Found", {}, None))
    assert main(["models", "--download", "s8"]) == 3
    assert "not on the release" in capsys.readouterr().out
