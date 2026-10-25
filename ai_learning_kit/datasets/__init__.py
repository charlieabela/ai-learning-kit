"""Datasets: where every dataset the kit uses comes from, and how it reaches a notebook.

There are three kinds, and a problem or session may use any of them:

- External datasets are published by someone else (a research group, an archive).
  The kit does not copy them: it downloads each one from its owner's website the
  first time it is needed, checks it against the checksum recorded here, and keeps
  it in a cache so that later notebooks reuse it. Each has an entry in SOURCES.
- Built-in datasets are small research datasets that an installed library already
  ships, for example Zachary's karate club in networkx. Helpers call that library.
- Packaged datasets are small files that belong to the kit itself (synthetic data,
  fictional case files, reference results), stored inside the package.

    from ai_learning_kit import datasets
    datasets.fetch("cora")          # path to the downloaded and checked file
    datasets.describe("cora")       # owner, address, licence note and citation
    datasets.cache_dir()            # where downloads are kept

Where downloads are kept (the first that applies):
- the folder named by the environment variable AILK_DATA_DIR;
- in Colab with Google Drive connected: My Drive › ai-learning-kit › data, so
  that a new Colab session does not download again;
- in Colab without Drive: /content/ai-learning-kit-data (temporary);
- otherwise the user's cache folder: ~/.cache/ai-learning-kit on Linux,
  ~/Library/Caches/ai-learning-kit on macOS, %LOCALAPPDATA%\\ai-learning-kit\\cache
  on Windows.

If a download is not possible (no internet, the owner's site is down), a copy of
the same file placed next to the notebook, or in a data/ folder beside it, is used
instead, provided its checksum matches.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .._version import __version__


@dataclass(frozen=True)
class Source:
    """An external dataset: who publishes it, where, and how to recognise the exact file."""
    name: str                 # short name used in code: fetch("cora")
    title: str
    owner: str                # who publishes it
    url: str                  # the owner's download address
    filename: str             # name of the file in the cache
    sha256: str               # checksum of the exact file the kit's results were computed with
    size: int                 # bytes, for progress messages and sanity checks
    landing: str = ""         # the owner's page about the dataset
    licence: str = ""         # what the owner says about use and redistribution
    citation: str = ""        # how to cite it
    notes: str = ""           # anything a lecturer or student should know
    mirrors: tuple = field(default_factory=tuple)   # further official addresses, tried in order


SOURCES: dict[str, Source] = {
    "cora": Source(
        name="cora",
        title="Cora citation network (LINQS)",
        owner="LINQS Statistical Relational Learning Group, UC Santa Cruz",
        url="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
        filename="cora.tgz",
        sha256="0d4ed463d1627bb7f3e8420effe8f5545fd492ae8f88dab44ce86cee7b26d7e8",
        size=168052,
        landing="https://linqs.org/datasets/",
        licence=("The archive's README describes provenance and schema; it states no redistribution licence, "
                 "so the kit downloads the archive from LINQS rather than copying it."),
        citation=("Sen, P., Namata, G., Bilgic, M., Getoor, L., Gallagher, B. and Eliassi-Rad, T. (2008). "
                  "Collective classification in network data. AI Magazine, 29(3), 93-106."),
        notes="2708 papers, 5429 citation links (5278 after removing self-loops and duplicates), 1433 binary "
              "word features, 7 topics.",
    ),
}

CHUNK = 1 << 16
_verified: set = set()       # files whose checksum was already checked in this Python session


class DatasetUnavailable(RuntimeError):
    """The dataset could not be downloaded and no valid local copy was found."""


# --------------------------------------------------------------------------- where files are kept
def _in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def cache_dir() -> Path:
    """The folder where downloaded datasets are kept (created if needed)."""
    env = os.environ.get("AILK_DATA_DIR")
    if env:
        base = Path(env).expanduser()
    elif _in_colab():
        drive = Path("/content/drive/MyDrive")
        base = drive / "ai-learning-kit" / "data" if os.path.ismount("/content/drive") else Path(
            "/content/ai-learning-kit-data")
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "ai-learning-kit" / "cache"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches" / "ai-learning-kit"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "ai-learning-kit"
    base.mkdir(parents=True, exist_ok=True)
    return base


# --------------------------------------------------------------------------- looking up sources
def source(name: str) -> Source:
    """The registered external dataset called name."""
    key = str(name).strip().lower()
    if key not in SOURCES:
        raise KeyError(f"Unknown dataset {name!r}. Registered datasets: {', '.join(SOURCES)}")
    return SOURCES[key]


def describe(name: str) -> dict:
    """Owner, address, licence note and citation of a dataset, as plain data."""
    s = source(name)
    return {"name": s.name, "title": s.title, "owner": s.owner, "url": s.url, "landing": s.landing,
            "sha256": s.sha256, "size": s.size, "licence": s.licence, "citation": s.citation, "notes": s.notes}


def catalog() -> list:
    """Every registered external dataset, as plain data."""
    return [describe(n) for n in SOURCES]


# --------------------------------------------------------------------------- checking and fetching
def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _matches(path: Path, s: Source) -> bool:
    if not path.is_file() or path.stat().st_size != s.size:
        return False
    key = (str(path.resolve()), path.stat().st_mtime_ns)
    if key in _verified:
        return True
    ok = sha256_of(path) == s.sha256
    if ok:
        _verified.add(key)
    return ok


def _local_copies(s: Source):
    """Copies a student or lecturer may have placed by hand: beside the notebook or in data/ next to it."""
    here = Path.cwd()
    for folder in (here, here / "data"):
        yield folder / s.filename


def _download(url: str, target: Path, s: Source, quiet: bool) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": f"ai-learning-kit/{__version__} (teaching)"})
    fd, tmp = tempfile.mkstemp(prefix=f".{s.filename}.", suffix=".part", dir=target.parent)
    os.close(fd)
    tmp = Path(tmp)
    try:
        with urllib.request.urlopen(request, timeout=60) as response, open(tmp, "wb") as out:
            total, done, last = int(response.headers.get("Content-Length") or s.size), 0, 0.0
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                out.write(block)
                done += len(block)
                if not quiet and time.time() - last > 2 and total > 5_000_000:
                    print(f"  {done / 1e6:.1f} of {total / 1e6:.1f} MB", flush=True)
                    last = time.time()
        got = sha256_of(tmp)
        if got != s.sha256:
            raise DatasetUnavailable(
                f"The file downloaded from {url} is not the one this kit was built with (checksum {got[:12]}..., "
                f"expected {s.sha256[:12]}...). The owner may have changed it. Nothing was saved. Tell your lecturer.")
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def fetch(name: str, quiet: bool = False) -> Path:
    """Path to a checked copy of an external dataset, downloading it from its owner the first time.

    Looks in the cache, then beside the notebook (and in data/ next to it), then downloads from
    the owner's address (and any official mirrors). Every copy is checked against the recorded
    checksum, so results are the same wherever the notebook runs.
    """
    s = source(name)
    cached = cache_dir() / s.filename
    if _matches(cached, s):
        return cached
    for local in _local_copies(s):
        if _matches(local, s):
            if not quiet:
                print(f"{s.title}: using the copy at {local}.")
            try:
                shutil.copy2(local, cached)
                return cached
            except OSError:
                return local
    errors = []
    for url in (s.url, *s.mirrors):
        if not quiet:
            print(f"{s.title}: downloading from {s.owner} ({url}) ...", flush=True)
        try:
            _download(url, cached, s, quiet)
            if not quiet:
                print(f"{s.title}: saved in {cached.parent} and checked.")
            return cached
        except DatasetUnavailable:
            raise
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            errors.append(f"{url}: {getattr(e, 'reason', e)}")
    raise DatasetUnavailable(
        f"{s.title} could not be downloaded ({'; '.join(errors)}). Check the internet connection and run the cell "
        f"again. If the publisher's site is down, download {s.filename} from {s.landing or s.url} on another computer "
        f"and put it next to this notebook.")


def prefetch(names, quiet: bool = False) -> dict:
    """Fetch several datasets now (kit.start does this for a session's datasets); returns name -> path."""
    return {n: fetch(n, quiet=quiet) for n in names}
