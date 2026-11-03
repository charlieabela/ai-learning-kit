"""Datasets: where every dataset the kit uses comes from, and how it reaches a notebook.

There are three kinds, and a problem or session may use any of them:

- External datasets are published by someone else (a research group, an archive).
  The kit does not copy them: it downloads each one from its owner's website the
  first time it is needed, checks it against the checksum recorded here, and keeps
  it in the data folder so that later notebooks reuse it. Each has an entry in SOURCES.
- Small research datasets (Zachary's karate club, Les Miserables, ...) are stored in
  the kit itself, with their citations: datasets.small("karate").
- Packaged files that belong to the kit (fictional case files, reference results,
  synthetic data) live with their problem or session.

    from ai_learning_kit import datasets
    datasets.fetch("cora")                     # path to the checked file, downloading it the first time
    datasets.use("cora", "~/Downloads/cora.tgz")                     # a copy on this computer
    datasets.use("cora", "https://drive.google.com/file/d/.../view") # a shared copy in the cloud
    datasets.add_folder("~/Library/CloudStorage/GoogleDrive-me/My Drive/ICS5111/data")
    datasets.small("karate")                   # a small dataset stored in the kit
    datasets.status()                          # what is where

Where an external dataset is looked for, in order (every copy is checked against the checksum):
1. the data folder (data_dir()), where earlier downloads and imports are kept;
2. next to the notebook, and in a data/ folder beside it;
3. folders you have added (add_folder(), or the environment variable AILK_DATA_PATH), for
   example a Google Drive or OneDrive folder synced to this computer, or Drive mounted in Colab;
4. the owner's website, then any copies the lecturer has approved for the course (Source.mirrors);
5. locations given for that dataset (use(), or the environment variable AILK_DATA_SOURCES):
   a file or folder on this computer, or a link to a shared copy (Google Drive, OneDrive or
   SharePoint, Dropbox, or any direct link).

The data folder is, in order: the folder named by AILK_DATA_DIR (Notebook Tutor sets it to the
data/ folder of its own clone); the data/ folder of the kit's clone when the kit runs from a git
clone; My Drive > ai-learning-kit > data in Colab with Drive connected (/content/ai-learning-kit-data
without Drive); otherwise the user's cache folder (~/.cache/ai-learning-kit, ~/Library/Caches/
ai-learning-kit, or %LOCALAPPDATA%\\ai-learning-kit\\cache).
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .._version import __version__
from .small import catalog as small_catalog
from .small import small


@dataclass(frozen=True)
class Source:
    """An external dataset: who publishes it, where, and how to recognise the exact file."""
    name: str                 # short name used in code: fetch("cora")
    title: str
    owner: str                # who publishes it
    url: str                  # the owner's download address
    filename: str             # name of the file in the data folder
    sha256: str               # checksum of the exact file the kit's results were computed with
    size: int                 # bytes, for progress messages and sanity checks
    landing: str = ""         # the owner's page about the dataset
    licence: str = ""         # what the owner says about use and redistribution
    citation: str = ""        # how to cite it
    notes: str = ""           # anything a lecturer or student should know
    mirrors: tuple = field(default_factory=tuple)   # copies approved by the lecturer, tried after the owner


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
              "word features, 7 topics. LINQS's server has been seen to refuse connections from some cloud "
              "machines (GitHub's CI runners); a copy on your own computer or a shared course copy then works.",
    ),
}

CHUNK = 1 << 16
SETTINGS = "locations.json"     # in the data folder: folders and per-dataset locations given with add_folder()/use()
_verified: set = set()          # files whose checksum was already checked in this Python session


class DatasetUnavailable(RuntimeError):
    """The dataset could not be found, downloaded or imported."""


# --------------------------------------------------------------------------- the data folder
def _in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def checkout_data_dir() -> Path | None:
    """The data/ folder of the kit's own git clone, when the kit runs from one (pip install -e)."""
    repo = Path(__file__).resolve().parents[2]
    return repo / "data" if (repo / ".git").exists() and (repo / "pyproject.toml").is_file() else None


def data_dir() -> Path:
    """The folder where downloaded and imported datasets are kept (created if needed)."""
    env = os.environ.get("AILK_DATA_DIR")
    clone = checkout_data_dir()
    if env:
        base = Path(env).expanduser()
    elif clone is not None:
        base = clone
    elif _in_colab():
        base = (Path("/content/drive/MyDrive/ai-learning-kit/data") if os.path.ismount("/content/drive")
                else Path("/content/ai-learning-kit-data"))
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "ai-learning-kit" / "cache"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches" / "ai-learning-kit"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "ai-learning-kit"
    base.mkdir(parents=True, exist_ok=True)
    return base


cache_dir = data_dir                # earlier name, kept for compatibility


# --------------------------------------------------------------------------- remembered locations
def _settings_path() -> Path:
    return data_dir() / SETTINGS


def _load_settings() -> dict:
    try:
        data = json.loads(_settings_path().read_text(encoding="utf-8"))
        return {"folders": list(data.get("folders", [])), "sources": dict(data.get("sources", {}))}
    except (OSError, ValueError):
        return {"folders": [], "sources": {}}


def _save_settings(settings: dict) -> None:
    try:
        _settings_path().write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except OSError:
        pass                         # a read-only data folder only means the setting is not remembered


def _env_sources() -> dict:
    """AILK_DATA_SOURCES: a JSON object {name: location}, or lines "name=location"."""
    raw = os.environ.get("AILK_DATA_SOURCES", "").strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        try:
            return {str(k).lower(): str(v) for k, v in json.loads(raw).items()}
        except ValueError:
            return {}
    out = {}
    for line in re.split(r"[\r\n]+", raw):
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip().lower()] = v.strip()
    return out


def folders() -> list:
    """Folders searched for dataset files, besides the data folder and the notebook's folder."""
    found = [p for p in os.environ.get("AILK_DATA_PATH", "").split(os.pathsep) if p.strip()]
    found += _load_settings()["folders"]
    clone = checkout_data_dir()
    if clone is not None:
        found.append(str(clone))
    seen, out = set(), []
    for f in found:
        p = Path(f).expanduser()
        if str(p) not in seen:
            seen.add(str(p))
            out.append(p)
    return out


def add_folder(folder) -> Path:
    """Remember a folder to search for dataset files, e.g. a synced Google Drive or OneDrive folder."""
    p = Path(folder).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"{p} is not a folder on this computer.")
    settings = _load_settings()
    if str(p) not in settings["folders"]:
        settings["folders"].append(str(p))
        _save_settings(settings)
    return p


def locations(name: str) -> list:
    """Locations given for one dataset (use() and AILK_DATA_SOURCES), in the order they are tried."""
    key = str(name).lower()
    out = []
    env = _env_sources().get(key)
    if env:
        out.append(env)
    saved = _load_settings()["sources"].get(key)
    if saved and saved not in out:
        out.append(saved)
    return out


# --------------------------------------------------------------------------- registered datasets
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
            "sha256": s.sha256, "size": s.size, "licence": s.licence, "citation": s.citation, "notes": s.notes,
            "filename": s.filename}


def catalog() -> list:
    """Every registered external dataset, as plain data."""
    return [describe(n) for n in SOURCES]


# --------------------------------------------------------------------------- checking
def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _matches(path: Path, s: Source) -> bool:
    try:
        if not path.is_file() or path.stat().st_size != s.size:
            return False
        key = (str(path.resolve()), path.stat().st_mtime_ns)
    except OSError:
        return False
    if key in _verified:
        return True
    ok = sha256_of(path) == s.sha256
    if ok:
        _verified.add(key)
    return ok


def _keep(found: Path, s: Source) -> Path:
    """Copy a checked file into the data folder (so other notebooks and clones find it) and return its path."""
    target = data_dir() / s.filename
    if found.resolve() == target.resolve():
        return target
    try:
        shutil.copy2(found, target)
        return target
    except OSError:
        return found


def _in_folder(folder: Path, s: Source) -> Path | None:
    """A matching copy inside a folder: under the dataset's file name, or any file of the same size."""
    candidate = folder / s.filename
    if _matches(candidate, s):
        return candidate
    try:
        for p in folder.iterdir():
            if p.is_file() and p.name != s.filename and _matches(p, s):
                return p
    except OSError:
        pass
    return None


# --------------------------------------------------------------------------- links to shared copies
def direct_url(link: str) -> str:
    """Turn a sharing link into a download link (Google Drive, OneDrive/SharePoint, Dropbox); others unchanged."""
    u = urllib.parse.urlparse(link)
    host = u.netloc.lower()
    q = urllib.parse.parse_qs(u.query)
    if host.endswith("drive.google.com") or host.endswith("docs.google.com"):
        m = re.search(r"/d/([A-Za-z0-9_-]{10,})", u.path)
        fid = m.group(1) if m else (q.get("id") or [None])[0]
        if fid:
            return f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"
    if host.endswith("sharepoint.com"):
        if "download" in q:
            return link
        return link + ("&" if u.query else "?") + "download=1"
    if host in ("1drv.ms", "onedrive.live.com"):
        token = base64.urlsafe_b64encode(link.encode()).decode().rstrip("=")
        return f"https://api.onedrive.com/v1.0/shares/u!{token}/root/content"
    if host.endswith("dropbox.com"):
        query = {k: v for k, v in q.items() if k != "dl"}
        query["dl"] = ["1"]
        return urllib.parse.urlunparse(u._replace(query=urllib.parse.urlencode(query, doseq=True)))
    return link


def _download(url: str, target: Path, s: Source, quiet: bool, timeout: int = 30) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": f"ai-learning-kit/{__version__} (teaching)"})
    fd, tmp = tempfile.mkstemp(prefix=f".{s.filename}.", suffix=".part", dir=target.parent)
    os.close(fd)
    tmp = Path(tmp)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, open(tmp, "wb") as out:
            kind = response.headers.get("Content-Type", "")
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
            if "html" in kind.lower():
                raise ValueError(f"{url} returned a web page, not the file. If it is a shared link, share the file "
                                 f"with 'Anyone with the link' (Google Drive) or 'Anyone' (OneDrive), or download it "
                                 f"in a browser and use the downloaded file.")
            raise ValueError(f"the file from {url} is not the one this kit was built with (checksum {got[:12]}..., "
                             f"expected {s.sha256[:12]}...)")
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def _from_location(location: str, s: Source, quiet: bool) -> Path:
    """A checked copy from a file or folder on this computer, or from a link."""
    if re.match(r"^https?://", location, re.I):
        target = data_dir() / s.filename
        _download(direct_url(location), target, s, quiet)
        return target
    p = Path(location).expanduser()
    if p.is_dir():
        found = _in_folder(p, s)
        if found is None:
            raise ValueError(f"no copy of {s.filename} with the right checksum in the folder {p}")
        return _keep(found, s)
    if p.is_file():
        if not _matches(p, s):
            raise ValueError(f"{p} is not the file this kit was built with (its checksum differs)")
        return _keep(p, s)
    raise ValueError(f"{p} does not exist on this computer")


# --------------------------------------------------------------------------- fetching
def fetch(name: str, quiet: bool = False) -> Path:
    """Path to a checked copy of an external dataset; see the module docstring for where it is looked for."""
    s = source(name)
    kept = data_dir() / s.filename
    if _matches(kept, s):
        return kept
    here = Path.cwd()
    extra = [Path("/content")] if _in_colab() else []          # where Colab's Files panel puts uploads
    for folder in (here, here / "data", *extra, *folders()):
        found = _in_folder(folder, s) if folder.is_dir() else None
        if found is not None:
            if not quiet:
                print(f"{s.title}: using the copy in {folder}.")
            return _keep(found, s)
    tried = []
    for i, url in enumerate((s.url, *s.mirrors)):
        who = s.owner if i == 0 else "the course copy approved by your lecturer"
        if not quiet:
            print(f"{s.title}: downloading from {who} ...", flush=True)
        try:
            _download(direct_url(url), kept, s, quiet)
            if not quiet:
                print(f"{s.title}: saved in {kept.parent} and checked.")
            return kept
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as e:
            tried.append(f"{who}: {getattr(e, 'reason', e)}")
    for location in locations(name):
        if not quiet:
            print(f"{s.title}: trying {location} ...", flush=True)
        try:
            path = _from_location(location, s, quiet)
            if not quiet:
                print(f"{s.title}: saved in {path.parent} and checked.")
            return path
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as e:
            tried.append(f"{location}: {getattr(e, 'reason', e)}")
    raise DatasetUnavailable(_missing_message(s, tried))


def _where() -> str:
    if os.environ.get("NOTEBOOK_TUTOR") == "1":
        return "tutor"
    return "colab" if _in_colab() else "jupyter"


def _missing_message(s: Source, tried: list) -> str:
    """What to do when a dataset is missing, in the words of the place the notebook runs in."""
    add = {
        "tutor": f"Add {s.filename} with the Data files button (Data files > Add data file...), then run this cell again.",
        "colab": f"Upload {s.filename} with the Files panel (the folder icon on the left), then run this cell again.",
        "jupyter": f"Put {s.filename} in the same folder as this notebook, then run this cell again.",
    }[_where()]
    return (f"{s.title} is missing: it could not be downloaded from {s.owner}.\n"
            + "".join(f"  - {t}\n" for t in tried)
            + f"{add}\nIf you do not have {s.filename}, use the copy your lecturer provides (for example on the VLE). "
              f"The kit checks that it is the right file and keeps it, so you add it only once.\n"
              f"(A copy elsewhere on this computer or a shared link also works: "
              f"kit.datasets.use(\"{s.name}\", \"<file, folder or link>\").)")


def use(name: str, location: str, remember: bool = True) -> Path:
    """Import a dataset from a file or folder on this computer or from a shared link, check it, and keep it.

    location: a file (".../cora.tgz"), a folder that holds it (a synced Google Drive or OneDrive
    folder, or /content/drive/MyDrive/... in Colab), or a link (Google Drive, OneDrive/SharePoint,
    Dropbox, or a direct download link). With remember=True the location is tried again on
    another computer that shares this data folder, or after the data folder is cleared.
    """
    s = source(name)
    path = _from_location(str(location), s, quiet=False)
    if remember:
        settings = _load_settings()
        settings["sources"][s.name] = str(location)
        _save_settings(settings)
    print(f"{s.title}: imported from {location}, checked, and kept in {path.parent}.")
    return path


def prefetch(names, quiet: bool = False) -> dict:
    """Fetch several datasets now (kit.start does this for a session's datasets); returns name -> path."""
    return {n: fetch(n, quiet=quiet) for n in names}


def status() -> list:
    """For each registered dataset: whether a checked copy is in the data folder, and where it will be looked for."""
    out = []
    for name, s in SOURCES.items():
        kept = data_dir() / s.filename
        out.append({"name": name, "title": s.title, "available": _matches(kept, s), "data_folder": str(data_dir()),
                    "folders": [str(f) for f in folders()], "owner_url": s.url, "mirrors": list(s.mirrors),
                    "locations": locations(name)})
    return out
