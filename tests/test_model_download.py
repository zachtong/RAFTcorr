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


# -- a new user without the weights: cancel, the error the GUI acts on, and import ----------------
def test_every_model_knows_its_download_size():
    for spec in zoo.MODELS.values():
        assert 15e6 < spec.size_bytes < 30e6, spec.name     # the release assets are about 20 MB
    assert zoo.RELEASE_PAGE_URL == "https://github.com/zachtong/RAFTcorr/releases/tag/weights-v1"


def test_stop_cancels_a_download_and_leaves_nothing_behind(spec, isolated, monkeypatch):
    from raftcorr.callbacks import FunctionCallbacks, StopRequested

    _serve(monkeypatch, PAYLOAD * 20)                 # several chunks
    seen: list[float] = []
    cb = FunctionCallbacks(progress=lambda f, m="": seen.append(f), should_stop=lambda: bool(seen))
    with pytest.raises(StopRequested):
        zoo.download(spec, callbacks=cb)
    assert seen and not any(isolated.iterdir())       # no file and no .part


def test_download_missing_does_not_swallow_a_stop(spec, isolated, monkeypatch):
    from raftcorr.callbacks import FunctionCallbacks, StopRequested

    monkeypatch.setitem(zoo.MODELS, "s4", spec)
    _serve(monkeypatch, PAYLOAD * 20)
    with pytest.raises(StopRequested):
        zoo.download_missing(["s4"], FunctionCallbacks(should_stop=lambda: True))


def test_a_corrupted_download_is_a_weights_unavailable_error(spec, isolated, monkeypatch):
    _serve(monkeypatch, PAYLOAD + b"x")
    with pytest.raises(zoo.WeightsUnavailableError, match="SHA-256"):
        zoo.download(spec)


def test_a_missing_model_raises_an_error_that_names_the_model(spec, isolated, monkeypatch):
    monkeypatch.setitem(zoo.MODELS, "s4", spec)
    _serve(monkeypatch, error=urllib.error.URLError("getaddrinfo failed"))
    with pytest.raises(zoo.ModelUnavailableError) as info:
        zoo.resolve_weights("s4", allow_download=True)
    err = info.value
    assert isinstance(err, FileNotFoundError)         # callers catching FileNotFoundError still do
    assert err.spec.key == "s4" and "internet connection" in err.reason
    # the cause alone, for the Models window, which shows the folder and the import itself
    assert err.short == "no connection to the download server (getaddrinfo failed); check the internet connection"
    with pytest.raises(zoo.ModelUnavailableError) as info:
        zoo.resolve_weights("s4")                     # no download asked
    assert info.value.spec.key == "s4" and str(isolated) in info.value.reason


def test_import_identifies_the_model_by_its_hash(spec, isolated, monkeypatch, tmp_path):
    monkeypatch.setitem(zoo.MODELS, "s4", spec)
    src = tmp_path / "downloads" / "RAFTcorr-s4 (1).pth"     # renamed by the browser
    src.parent.mkdir()
    src.write_bytes(PAYLOAD)
    got, path = zoo.import_weights(src)
    assert got.key == "s4" and path == isolated / "RAFTcorr-s4.pth"
    assert path.read_bytes() == PAYLOAD and src.is_file()      # copied, the original stays
    assert zoo.resolve_weights("s4") == path
    assert zoo.import_weights(path) == (got, path)             # importing the installed file: no-op


def test_import_refuses_a_file_that_is_not_a_released_model(isolated, tmp_path):
    src = tmp_path / "RAFTcorr-s4.pth"
    src.write_bytes(b"something else")
    with pytest.raises(ValueError, match="not one of the released RAFTcorr models"):
        zoo.import_weights(src)
    assert not isolated.exists() or not any(isolated.iterdir())


def test_cli_models_import(spec, isolated, monkeypatch, tmp_path, capsys):
    from raftcorr.cli.main import main

    monkeypatch.setitem(zoo.MODELS, "s4", spec)
    good, bad = tmp_path / "a.pth", tmp_path / "b.pth"
    good.write_bytes(PAYLOAD)
    bad.write_bytes(b"other")
    assert main(["models", "--import", str(good)]) == 0
    assert (isolated / "RAFTcorr-s4.pth").is_file()
    assert main(["models", "--import", str(bad)]) == 3
    out = capsys.readouterr().out
    assert "RAFTcorr-s4: imported" in out and "not one of the released RAFTcorr models" in out


# -- what can go wrong around the download itself ---------------------------------------------------
def test_a_folder_that_cannot_be_written_is_not_called_a_network_problem(spec, tmp_path, monkeypatch):
    blocker = tmp_path / "a file"
    blocker.write_text("x")
    monkeypatch.setattr(zoo, "user_weights_dir", lambda: blocker / "models")    # mkdir under a file fails
    _serve(monkeypatch, PAYLOAD)
    with pytest.raises(zoo.WeightsUnavailableError) as info:
        zoo.download(spec)
    assert info.value.short.startswith("cannot write to the model folder") and "internet" not in info.value.short


def test_a_dropped_connection_is_a_network_problem(spec, isolated, monkeypatch):
    import http.client

    class _Dropping(_Response):
        def read(self, *_a):
            raise http.client.IncompleteRead(b"")

    monkeypatch.setattr(zoo.urllib.request, "urlopen", lambda req, timeout=None: _Dropping(PAYLOAD))
    with pytest.raises(zoo.WeightsUnavailableError) as info:
        zoo.download(spec)
    assert info.value.short.endswith("check the internet connection") and not any(isolated.iterdir())


def test_another_writers_temporary_file_is_left_alone(spec, isolated, monkeypatch):
    """A run downloading a model while the Models window imports it: each writes its own file."""
    isolated.mkdir(parents=True)
    other = isolated / "RAFTcorr-s4.pth.part"
    other.write_bytes(b"someone else is writing this")
    with open(other, "rb"):                           # held open, as by the other writer
        _serve(monkeypatch, PAYLOAD)
        path = zoo.download(spec)
    assert path.read_bytes() == PAYLOAD and other.read_bytes() == b"someone else is writing this"


def test_a_checked_file_is_not_hashed_again_until_it_changes(spec, isolated, monkeypatch):
    import os

    monkeypatch.setitem(zoo.MODELS, "s4", spec)
    isolated.mkdir(parents=True)
    f = isolated / spec.filename
    f.write_bytes(PAYLOAD)
    calls: list = []
    real = zoo.sha256_of
    monkeypatch.setattr(zoo, "sha256_of", lambda p, *a: calls.append(p) or real(p, *a))
    assert zoo.find_weights(spec) == f and zoo.find_weights(spec) == f and len(calls) == 1
    st = f.stat()
    f.write_bytes(PAYLOAD[:-1] + b"?")                # the same size, another content
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert zoo.find_weights(spec) is None and len(calls) == 2
