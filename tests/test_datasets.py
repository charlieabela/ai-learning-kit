"""External datasets: owner first, then copies on this computer or in the cloud; small datasets in the kit."""
import hashlib
import http.server
import os
import threading
from functools import partial
from pathlib import Path

import pytest

from ai_learning_kit import datasets
from ai_learning_kit.datasets import DatasetUnavailable, Source

UNREACHABLE = "http://127.0.0.1:9/owner.bin"       # nothing listens on port 9: the owner's site is "down"


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """An empty data folder, a notebook folder as the current folder, and no settings from the machine."""
    folder = tmp_path / "data"
    monkeypatch.setenv("AILK_DATA_DIR", str(folder))
    for var in ("AILK_DATA_PATH", "AILK_DATA_SOURCES", "NOTEBOOK_TUTOR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AILK_COPIES_URL", "")          # the repository's list of course copies is not read
    monkeypatch.setattr(datasets, "checkout_data_dir", lambda: None)
    work = tmp_path / "notebook"
    work.mkdir()
    monkeypatch.chdir(work)
    datasets._verified.clear()
    return folder


@pytest.fixture()
def server(tmp_path):
    """A local web server standing in for an owner's site or a cloud share; counts requests."""
    root = tmp_path / "web"
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


def register(monkeypatch, url, data: bytes, sha=None, name="toy", mirrors=()):
    src = Source(name=name, title="Toy dataset", owner="Toy Lab", url=url, filename="toy.bin",
                 sha256=sha or hashlib.sha256(data).hexdigest(), size=len(data), licence="test", landing=url,
                 mirrors=tuple(mirrors))
    monkeypatch.setitem(datasets.SOURCES, name, src)
    return src


# --------------------------------------------------------------------------- the owner's site
def test_downloads_from_the_owner_once_and_keeps_it(store, server, monkeypatch, capsys):
    root, base, hits = server
    data = os.urandom(5000)
    (root / "toy.bin").write_bytes(data)
    register(monkeypatch, f"{base}/toy.bin", data)
    path = datasets.fetch("toy")
    assert path == store / "toy.bin" and path.read_bytes() == data
    assert "downloading from Toy Lab" in capsys.readouterr().out
    datasets.fetch("toy")
    assert hits == ["/toy.bin"] and not list(store.glob("*.part"))


def test_a_changed_file_is_refused_and_nothing_is_kept(store, server, monkeypatch):
    root, base, _ = server
    (root / "toy.bin").write_bytes(b"the owner changed this file")
    register(monkeypatch, f"{base}/toy.bin", b"x" * 27, sha="0" * 64)
    with pytest.raises(DatasetUnavailable, match="not the one this kit was built with"):
        datasets.fetch("toy")
    assert not (store / "toy.bin").exists() and not list(store.glob("*.part"))


def test_a_corrupted_copy_in_the_data_folder_is_replaced(store, server, monkeypatch):
    root, base, hits = server
    data = b"abc" * 1000
    (root / "toy.bin").write_bytes(data)
    register(monkeypatch, f"{base}/toy.bin", data)
    store.mkdir(parents=True, exist_ok=True)
    (store / "toy.bin").write_bytes(b"cba" * 1000)    # same size, wrong content
    assert datasets.fetch("toy").read_bytes() == data and hits == ["/toy.bin"]


def test_a_lecturer_approved_course_copy_is_tried_after_the_owner(store, server, monkeypatch, capsys):
    root, base, hits = server
    data = b"course copy" * 50
    (root / "course.bin").write_bytes(data)
    register(monkeypatch, UNREACHABLE, data, mirrors=[f"{base}/course.bin"])
    assert datasets.fetch("toy").read_bytes() == data and hits == ["/course.bin"]
    out = capsys.readouterr().out
    assert "downloading from the course copy" in out and "Toy Lab did not answer, so the course copy was used" in out


def test_course_copies_listed_later_in_the_repository_are_used(store, server, monkeypatch):
    """A notebook pinned to an earlier kit still reaches a course copy added to data-copies.json later."""
    root, base, hits = server
    data = b"added later" * 80
    (root / "later.bin").write_bytes(data)
    register(monkeypatch, UNREACHABLE, data, mirrors=[f"{base}/missing.bin"])
    # the list itself only accepts https addresses (tested below); the local test server speaks http
    monkeypatch.setattr(datasets, "_remote_copies", lambda: {"toy": [f"{base}/later.bin"]})
    assert datasets.fetch("toy", quiet=True).read_bytes() == data
    assert hits == ["/missing.bin", "/later.bin"]


def test_the_repository_list_is_read_safely(server, monkeypatch):
    root, base, _ = server
    (root / "copies.json").write_text('{"_about": "text", "toy": ["https://example.org/toy.bin", "http://insecure", 3]}')
    monkeypatch.setenv("AILK_COPIES_URL", f"{base}/copies.json")
    assert datasets._remote_copies() == {"toy": ["https://example.org/toy.bin"]}
    monkeypatch.setenv("AILK_COPIES_URL", f"{base}/nothing-here.json")
    assert datasets._remote_copies() == {}
    monkeypatch.setenv("AILK_COPIES_URL", "")
    assert datasets._remote_copies() == {}


def test_in_colab_the_student_is_asked_for_the_file(store, monkeypatch, tmp_path, capsys):
    import sys
    import types
    data = b"the lecturer's copy" * 30
    register(monkeypatch, UNREACHABLE, data)
    chosen = {"bytes": b"something else"}

    def upload():                       # google.colab.files.upload saves the chosen file in the current folder
        Path("toy.bin").write_bytes(chosen["bytes"])
        return {"toy.bin": chosen["bytes"]}

    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=upload)
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    with pytest.raises(DatasetUnavailable, match="Files panel"):            # the wrong file: refused, then the advice
        datasets.fetch("toy", quiet=True)
    assert "checksum differs" in capsys.readouterr().out
    chosen["bytes"] = data
    assert datasets.fetch("toy", quiet=True) == store / "toy.bin"
    assert "right file" in capsys.readouterr().out
    monkeypatch.setenv("AILK_NO_PROMPT", "1")
    assert datasets._colab_upload(datasets.source("toy")) is None


def test_the_missing_data_message_is_shown_without_a_traceback(store, monkeypatch):
    register(monkeypatch, UNREACHABLE, b"q" * 5)
    with pytest.raises(DatasetUnavailable) as e:
        datasets.fetch("toy", quiet=True)
    lines = e.value._render_traceback_()
    assert len(lines) == 1 and "Toy dataset is missing" in lines[0] and "Traceback" not in lines[0]


# --------------------------------------------------------------------------- copies on this computer
def test_a_copy_beside_the_notebook_or_in_data_is_used(store, monkeypatch, tmp_path):
    data = b"local copy" * 100
    register(monkeypatch, UNREACHABLE, data)
    (tmp_path / "notebook" / "data").mkdir()
    (tmp_path / "notebook" / "data" / "toy.bin").write_bytes(data)
    path = datasets.fetch("toy", quiet=True)
    assert path == store / "toy.bin" and path.read_bytes() == data       # kept in the data folder for next time


def test_a_copy_under_another_name_in_a_synced_folder_is_found(store, monkeypatch, tmp_path):
    data = b"synced" * 300
    register(monkeypatch, UNREACHABLE, data)
    synced = tmp_path / "Google Drive" / "My Drive" / "ICS5111"
    synced.mkdir(parents=True)
    (synced / "toy (1).bin").write_bytes(data)        # a browser download renamed by the operating system
    with pytest.raises(DatasetUnavailable):
        datasets.fetch("toy", quiet=True)
    datasets.add_folder(synced)
    assert datasets.fetch("toy", quiet=True).read_bytes() == data
    assert str(synced.resolve()) in [str(f) for f in datasets.folders()]


def test_folders_from_the_environment(store, monkeypatch, tmp_path):
    data = b"env folder" * 30
    register(monkeypatch, UNREACHABLE, data)
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    (b / "toy.bin").write_bytes(data)
    monkeypatch.setenv("AILK_DATA_PATH", f"{a}{os.pathsep}{b}")
    assert datasets.fetch("toy", quiet=True).read_bytes() == data


def test_use_imports_a_file_checks_it_and_remembers_where(store, monkeypatch, tmp_path):
    data = b"downloaded in a browser" * 20
    register(monkeypatch, UNREACHABLE, data)
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    (downloads / "toy.bin").write_bytes(data)
    assert datasets.use("toy", downloads / "toy.bin").read_bytes() == data
    (store / "toy.bin").unlink()                       # the data folder was cleared
    assert datasets.fetch("toy", quiet=True).read_bytes() == data       # found again through the remembered location
    assert datasets.locations("toy") == [str(downloads / "toy.bin")]


def test_use_refuses_a_different_file(store, monkeypatch, tmp_path):
    register(monkeypatch, UNREACHABLE, b"right" * 10)
    (tmp_path / "wrong.bin").write_bytes(b"wrong" * 10)
    with pytest.raises(ValueError, match="checksum differs"):
        datasets.use("toy", tmp_path / "wrong.bin")
    with pytest.raises(ValueError, match="does not exist"):
        datasets.use("toy", tmp_path / "missing.bin")


@pytest.mark.parametrize("where,expected", [("1", "Data files button"), (None, "same folder as this notebook")])
def test_owner_offline_and_nothing_found_says_what_to_do(store, monkeypatch, where, expected):
    register(monkeypatch, UNREACHABLE, b"z" * 10)
    if where:
        monkeypatch.setenv("NOTEBOOK_TUTOR", where)
    else:
        monkeypatch.delenv("NOTEBOOK_TUTOR", raising=False)
    with pytest.raises(DatasetUnavailable) as e:
        datasets.fetch("toy", quiet=True)
    msg = str(e.value)
    assert msg.startswith("Toy dataset is missing") and expected in msg and "toy.bin" in msg
    assert "copy your lecturer provides" in msg and 'kit.datasets.use("toy"' in msg


# --------------------------------------------------------------------------- copies in the cloud (shared links)
def test_use_imports_from_a_shared_link(store, server, monkeypatch):
    root, base, hits = server
    data = b"shared by the lecturer" * 40
    (root / "shared.bin").write_bytes(data)
    register(monkeypatch, UNREACHABLE, data)
    assert datasets.use("toy", f"{base}/shared.bin").read_bytes() == data
    assert hits == ["/shared.bin"]


def test_a_link_from_the_environment_is_tried(store, server, monkeypatch):
    root, base, _ = server
    data = b"ci secret link" * 40
    (root / "s.bin").write_bytes(data)
    register(monkeypatch, UNREACHABLE, data)
    monkeypatch.setenv("AILK_DATA_SOURCES", f"other=x\ntoy={base}/s.bin")
    assert datasets.fetch("toy", quiet=True).read_bytes() == data
    monkeypatch.setenv("AILK_DATA_SOURCES", '{"toy": "%s/s.bin"}' % base)
    assert datasets.locations("toy") == [f"{base}/s.bin"]


def test_a_link_that_opens_a_web_page_is_explained(store, server, monkeypatch):
    root, base, _ = server
    (root / "page.html").write_text("<html>Sign in to continue</html>")
    register(monkeypatch, UNREACHABLE, b"q" * 10)
    with pytest.raises(ValueError, match="web page, not the file"):
        datasets.use("toy", f"{base}/page.html")


@pytest.mark.parametrize("link,expected", [
    ("https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQr/view?usp=sharing",
     "https://drive.usercontent.google.com/download?id=1AbCdEfGhIjKlMnOpQr&export=download&confirm=t"),
    ("https://drive.google.com/open?id=1AbCdEfGhIjKlMnOpQr",
     "https://drive.usercontent.google.com/download?id=1AbCdEfGhIjKlMnOpQr&export=download&confirm=t"),
    ("https://drive.google.com/uc?export=download&id=1AbCdEfGhIjKlMnOpQr",
     "https://drive.usercontent.google.com/download?id=1AbCdEfGhIjKlMnOpQr&export=download&confirm=t"),
    ("https://universitymalta-my.sharepoint.com/:u:/g/personal/x/EAbc?e=Xy1",
     "https://universitymalta-my.sharepoint.com/:u:/g/personal/x/EAbc?e=Xy1&download=1"),
    ("https://www.dropbox.com/s/abc/cora.tgz?dl=0", "https://www.dropbox.com/s/abc/cora.tgz?dl=1"),
    ("https://example.org/data/cora.tgz", "https://example.org/data/cora.tgz"),
])
def test_sharing_links_become_download_links(link, expected):
    assert datasets.direct_url(link) == expected


def test_onedrive_personal_links_use_the_shares_endpoint():
    out = datasets.direct_url("https://1drv.ms/u/s!AbCdEf")
    assert out.startswith("https://api.onedrive.com/v1.0/shares/u!") and out.endswith("/root/content")


# --------------------------------------------------------------------------- where data is kept
def test_data_folder_is_the_clone_s_data_folder_when_running_from_a_clone(tmp_path, monkeypatch):
    monkeypatch.delenv("AILK_DATA_DIR", raising=False)
    monkeypatch.setattr(datasets, "checkout_data_dir", lambda: tmp_path / "clone" / "data")
    assert datasets.data_dir() == tmp_path / "clone" / "data" and (tmp_path / "clone" / "data").is_dir()
    monkeypatch.setenv("AILK_DATA_DIR", str(tmp_path / "tutor-clone" / "data"))        # Notebook Tutor sets this
    assert datasets.data_dir() == tmp_path / "tutor-clone" / "data"


def test_status_reports_where_data_is(store, monkeypatch):
    register(monkeypatch, UNREACHABLE, b"s" * 5)
    entry = next(e for e in datasets.status() if e["name"] == "toy")
    assert entry["available"] is False and entry["data_folder"] == str(store) and entry["owner_url"] == UNREACHABLE


def test_every_registered_dataset_names_its_owner_licence_and_checksum():
    for entry in datasets.catalog():
        assert entry["owner"] and entry["url"].startswith("https://") and len(entry["sha256"]) == 64
        assert entry["licence"] and entry["citation"] and entry["size"] > 0
    with pytest.raises(KeyError, match="Registered datasets"):
        datasets.source("not-a-dataset")


# --------------------------------------------------------------------------- small datasets stored in the kit
def test_small_datasets_are_stored_with_their_citations():
    cat = {d["name"]: d for d in datasets.small_catalog()}
    assert set(cat) >= {"karate", "lesmis", "florentine", "davis"}
    assert all(d["citation"] and d["licence"] for d in cat.values())
    assert (cat["karate"]["nodes"], cat["karate"]["edges"]) == (34, 78)
    g = datasets.small("karate")
    assert {g.nodes[n]["club"] for n in g} == {"Mr. Hi", "Officer"} and all("weight" in d for *_, d in g.edges(data=True))
    assert datasets.small("lesmis").number_of_nodes() == 77
    with pytest.raises(KeyError, match="Available"):
        datasets.small("not-a-graph")


def test_small_datasets_match_the_networkx_copies():
    nx = pytest.importorskip("networkx")
    for name, func in (("karate", nx.karate_club_graph), ("florentine", nx.florentine_families_graph)):
        a, b = datasets.small(name), func()
        assert sorted(map(sorted, a.edges())) == sorted(map(sorted, b.edges()))


# --------------------------------------------------------------------------- the owners themselves (CI)
@pytest.mark.skipif(not os.environ.get("AILK_ONLINE"), reason="set AILK_ONLINE=1 to contact the owners")
@pytest.mark.parametrize("name", sorted(datasets.SOURCES))
def test_owner_still_serves_the_exact_file(name, store):
    """CI: the owner's own address still gives the exact file. A refused connection is reported, not failed:
    the owner's availability is outside our control, and the kit falls back to the other locations."""
    import urllib.error
    s = datasets.source(name)
    try:
        datasets._download(s.url, store / s.filename, s, quiet=True)
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        print(f"::warning title={name}::{s.owner} could not be reached from this machine ({getattr(e, 'reason', e)}); "
              f"the kit falls back to course copies and locations given by students.")
        pytest.skip(f"owner unreachable from here: {e}")
    assert datasets.sha256_of(store / s.filename) == s.sha256


def _course_copies():
    import json
    listed = json.loads((Path(__file__).parents[1] / "data-copies.json").read_text())
    pairs = [(n, u) for n, s in datasets.SOURCES.items() for u in s.mirrors]
    pairs += [(n, u) for n, us in listed.items() if isinstance(us, list) for u in us]
    return sorted(set(pairs))


@pytest.mark.skipif(not os.environ.get("AILK_ONLINE"), reason="set AILK_ONLINE=1 to contact the course copies")
@pytest.mark.parametrize("name,url", _course_copies())
def test_every_course_copy_serves_the_exact_file(name, url, store):
    """CI: a course copy that answers must be the exact file. One that is not there yet (the data release is
    published after the code that refers to it is merged) is reported as a warning, not a failure."""
    import urllib.error
    s = datasets.source(name)
    try:
        datasets._download(url, datasets.data_dir() / s.filename, s, quiet=True)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        print(f"::warning title={name}::the course copy {url} does not exist yet: publish the data release that "
              f"holds {s.filename}.")
        pytest.skip(f"course copy not published yet: {url}")
    assert datasets.sha256_of(store / s.filename) == s.sha256
