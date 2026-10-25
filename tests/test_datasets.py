"""External datasets: download from the owner, checksum, cache, local-copy fallback."""
import hashlib
import http.server
import os
import threading
from functools import partial

import pytest

from ai_learning_kit import datasets
from ai_learning_kit.datasets import DatasetUnavailable, Source


@pytest.fixture()
def cache(tmp_path, monkeypatch):
    folder = tmp_path / "cache"
    monkeypatch.setenv("AILK_DATA_DIR", str(folder))
    work = tmp_path / "notebook"
    work.mkdir()
    monkeypatch.chdir(work)
    datasets._verified.clear()
    return folder


@pytest.fixture()
def server(tmp_path):
    """A local web server standing in for a dataset owner's site; counts requests."""
    root = tmp_path / "owner"
    root.mkdir()
    hits = []

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            hits.append(self.path)
            return super().do_GET()

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(root)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield root, f"http://127.0.0.1:{httpd.server_address[1]}", hits
    httpd.shutdown()


def register(monkeypatch, url, data: bytes, sha=None, name="toy"):
    src = Source(name=name, title="Toy dataset", owner="Toy Lab", url=url, filename="toy.bin",
                 sha256=sha or hashlib.sha256(data).hexdigest(), size=len(data), licence="test", landing=url)
    monkeypatch.setitem(datasets.SOURCES, name, src)
    return src


def test_downloads_once_checks_and_caches(cache, server, monkeypatch, capsys):
    root, base, hits = server
    data = os.urandom(5000)
    (root / "toy.bin").write_bytes(data)
    register(monkeypatch, f"{base}/toy.bin", data)
    path = datasets.fetch("toy")
    assert path == cache / "toy.bin" and path.read_bytes() == data
    assert "downloading from Toy Lab" in capsys.readouterr().out
    datasets.fetch("toy")
    assert hits == ["/toy.bin"]                       # the second call used the cache
    assert not list(cache.glob("*.part"))


def test_a_changed_file_is_refused_and_nothing_is_kept(cache, server, monkeypatch):
    root, base, _ = server
    (root / "toy.bin").write_bytes(b"the owner changed this file")
    register(monkeypatch, f"{base}/toy.bin", b"x" * 27, sha="0" * 64)
    with pytest.raises(DatasetUnavailable, match="not the one this kit was built with"):
        datasets.fetch("toy")
    assert not (cache / "toy.bin").exists() and not list(cache.glob("*.part"))


def test_a_corrupted_cache_is_downloaded_again(cache, server, monkeypatch):
    root, base, hits = server
    data = b"abc" * 1000
    (root / "toy.bin").write_bytes(data)
    register(monkeypatch, f"{base}/toy.bin", data)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "toy.bin").write_bytes(b"cba" * 1000)    # same size, wrong content
    assert datasets.fetch("toy").read_bytes() == data and hits == ["/toy.bin"]


def test_owner_offline_falls_back_to_a_copy_beside_the_notebook(cache, monkeypatch, tmp_path):
    data = b"local copy" * 100
    register(monkeypatch, "http://127.0.0.1:9/unreachable.bin", data)
    (tmp_path / "notebook" / "data").mkdir()
    (tmp_path / "notebook" / "data" / "toy.bin").write_bytes(data)
    assert datasets.fetch("toy", quiet=True).read_bytes() == data


def test_owner_offline_without_a_copy_says_what_to_do(cache, monkeypatch):
    register(monkeypatch, "http://127.0.0.1:9/unreachable.bin", b"z" * 10)
    with pytest.raises(DatasetUnavailable, match="put it next to this notebook"):
        datasets.fetch("toy", quiet=True)


def test_a_wrong_local_copy_is_ignored(cache, monkeypatch, tmp_path):
    register(monkeypatch, "http://127.0.0.1:9/unreachable.bin", b"right" * 10)
    (tmp_path / "notebook" / "toy.bin").write_bytes(b"wrong" * 10)
    with pytest.raises(DatasetUnavailable):
        datasets.fetch("toy", quiet=True)


def test_every_registered_dataset_names_its_owner_licence_and_checksum():
    for entry in datasets.catalog():
        assert entry["owner"] and entry["url"].startswith("https://") and len(entry["sha256"]) == 64
        assert entry["licence"] and entry["citation"] and entry["size"] > 0
    with pytest.raises(KeyError, match="Registered datasets"):
        datasets.source("not-a-dataset")


@pytest.mark.skipif(not os.environ.get("AILK_ONLINE"), reason="set AILK_ONLINE=1 to download from the owners")
@pytest.mark.parametrize("name", sorted(datasets.SOURCES))
def test_owner_still_serves_the_exact_file(name, cache):
    """CI: every registered dataset still downloads from its owner and matches its checksum."""
    path = datasets.fetch(name, quiet=True)
    assert datasets.sha256_of(path) == datasets.source(name).sha256
