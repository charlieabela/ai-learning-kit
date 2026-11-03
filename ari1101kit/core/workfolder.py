"""Where a student's work is kept: the folder of the notebook itself.

kit.start() finds the folder that holds the running notebook and makes it the
current folder, so the answer book, the case files and every file saved with a
plain name, for example df.to_csv("q3_high_value.csv"), sit next to the notebook.

How the notebook's folder is found
- Jupyter (Notebook or Lab) and VS Code start the kernel in the notebook's
  folder; VS Code also tells the kernel the notebook's path (__vsc_ipynb_file__),
  and recent Jupyter servers set JPY_SESSION_NAME.
- Colab runs the kernel on a temporary machine, not in Drive. The kit connects
  Google Drive, asks the Colab runtime which notebook is open (its name and Drive
  file id), and finds that file in the mounted Drive: first by name, and if the
  name is not unique, through the Drive API (which asks for permission once).

kit.start(..., folder=...) chooses another folder. In Colab a relative path is a
folder in My Drive ("ARI1101/S02"); elsewhere it is relative to the notebook.

When there is no such folder (a notebook opened from GitHub or from someone
else's shared link is not in the student's own Drive), the kit says why, keeps
the work in My Drive/ARI1101/<session>, and explains how to save a copy.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import time
import urllib.request
from pathlib import Path

COLAB_DRIVE = Path("/content/drive")      # where Colab mounts Google Drive
COLAB_LOCAL = Path("/content")            # Colab's temporary disk
UNIT = "ARI1101"
START_DIR = Path.cwd()                    # where the kernel started: the notebook's folder in Jupyter
SAVE_A_COPY = ("To keep your work next to the notebook, open it in Colab, choose File › Save a copy in Drive, "
               "open that copy, and run kit.start again.")


# --------------------------------------------------------------------------- environment
def in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def _writable(folder: Path) -> bool:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / ".ari1101kit_write_test"
        probe.write_text("ok")
        probe.unlink()
        return True
    except OSError:
        return False


# --------------------------------------------------------------------------- Jupyter, VS Code
def local_notebook_folder() -> tuple[Path, str]:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        vsc = ip.user_ns.get("__vsc_ipynb_file__") if ip is not None else None
        if vsc and Path(vsc).exists():
            return Path(vsc).parent, "the folder of this notebook"
    except ImportError:
        pass
    jpy = os.environ.get("JPY_SESSION_NAME")
    if jpy and Path(jpy).is_absolute() and Path(jpy).exists():
        return Path(jpy).parent, "the folder of this notebook"
    return START_DIR, "the folder of this notebook (the folder Jupyter started it in)"


# --------------------------------------------------------------------------- Colab
def colab_session() -> dict | None:
    """Ask the Colab runtime's Jupyter server which notebook is open: {'name': ..., 'path': 'fileId=...'}."""
    hosts = []
    try:
        own_ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        own_ip = None
    for h in (os.environ.get("COLAB_JUPYTER_IP"), own_ip, "172.28.0.12", "172.28.0.2", "127.0.0.1"):
        if h and h not in hosts:
            hosts.append(h)
    port = os.environ.get("COLAB_JUPYTER_PORT") or "9000"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for host in hosts:
        try:
            with opener.open(f"http://{host}:{port}/api/sessions", timeout=2) as r:
                sessions = json.load(r)
            if sessions:
                return sessions[0]
        except Exception:
            continue
    return None


def file_id_of(session: dict) -> str | None:
    m = re.search(r"fileId=([\w-]+)", str(session.get("path", "")))
    return m.group(1) if m else None


def find_by_name(drive_root: Path, name: str, file_id: str | None = None,
                 max_depth: int = 5, max_entries: int = 5000) -> Path | None:
    """Look for the notebook file by name under My Drive; return its folder if exactly one match.

    'Colab Notebooks' (where File › Save a copy in Drive puts copies) is searched first.
    Where the mounted Drive exposes file ids as extended attributes, they settle ties.
    """
    if not name or not drive_root.exists():
        return None

    def drive_id(p):
        try:
            return os.getxattr(p, "user.drive.id").decode()
        except (OSError, AttributeError):
            return None

    matches = []
    first = drive_root / "Colab Notebooks" / name
    if first.exists():
        if file_id and drive_id(first) == file_id:
            return first.parent                  # confirmed by the Drive file id
        matches.append(first)
    seen = 0
    for folder, dirs, files in os.walk(drive_root):
        depth = len(Path(folder).relative_to(drive_root).parts)
        dirs[:] = [d for d in dirs if not d.startswith(".")] if depth < max_depth else []
        seen += len(files) + len(dirs)
        if name in files and Path(folder, name) not in matches:
            matches.append(Path(folder, name))
        if seen > max_entries:
            break
    if len(matches) > 1 and file_id:
        same = [p for p in matches if drive_id(p) == file_id]
        if same:
            matches = same
    return matches[0].parent if len(matches) == 1 else None


def path_from_drive_api(file_id: str) -> Path:
    """Follow the file's parent folders with the Drive API (asks the student for permission once)."""
    from google.colab import auth  # type: ignore
    from googleapiclient.discovery import build  # type: ignore
    auth.authenticate_user()
    files = build("drive", "v3", cache_discovery=False).files()
    names, fid = [], file_id
    while True:
        f = files.get(fileId=fid, fields="name,parents,driveId", supportsAllDrives=True).execute()
        parents = f.get("parents")
        if not parents:                      # reached the top: My Drive, or a shared drive
            root = COLAB_DRIVE / "Shareddrives" / f["name"] if f.get("driveId") else COLAB_DRIVE / "MyDrive"
            break
        names.append(f["name"])
        fid = parents[0]
    names.reverse()                          # [folder, ..., notebook name]
    return root.joinpath(*names[:-1])


def drive_mounted() -> bool:
    """True only when Google Drive is really connected (a mount point), not when a folder of that name exists."""
    return os.path.ismount(COLAB_DRIVE)


def _rescue(backup: Path) -> list[Path]:
    """Copy files that were written to a look-alike /content/drive folder into the real Drive.

    A file is copied only when it is missing in Drive or older there, so newer work is never overwritten.
    """
    copied = []
    for src in backup.rglob("*"):
        if not src.is_file():
            continue
        dst = COLAB_DRIVE / src.relative_to(backup)
        try:
            if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied.append(dst)
        except OSError:
            continue
    return copied


def ensure_drive() -> None:
    """Connect Google Drive in Colab, unless it is already connected.

    If an earlier call created a plain folder at /content/drive (so files were written to the temporary
    disk instead of Drive), that folder is moved aside, Drive is connected, and its files are copied in.
    """
    if drive_mounted():
        return
    from google.colab import drive as gdrive  # type: ignore
    backup = None
    if COLAB_DRIVE.exists() and any(COLAB_DRIVE.iterdir()):
        backup = COLAB_LOCAL / f"drive_not_synced_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(COLAB_DRIVE), str(backup))
    print("Connecting your Google Drive, so that your work is kept after Colab disconnects.")
    print("A window will ask you to choose your account and allow access.")
    gdrive.mount(str(COLAB_DRIVE))
    if backup is not None:
        copied = _rescue(backup)
        print(f"Files saved earlier while Drive was not connected were found in {backup}.")
        if copied:
            print(f"{len(copied)} of them were copied into your Drive, for example {copied[0]}.")


def resolve_folder(folder) -> Path:
    """Turn the folder= argument into a path.

    In Colab, a relative path is a folder in My Drive: "ARI1101/S02" means My Drive › ARI1101 › S02, and a
    leading "My Drive/" or "MyDrive/" is accepted too. Elsewhere, a relative path is taken from the
    notebook's folder.
    """
    p = Path(str(folder).strip()).expanduser()
    if in_colab():
        parts = p.parts
        if p.is_absolute():
            if parts[:4] == ("/", "content", "drive", "My Drive"):
                p = COLAB_DRIVE / "MyDrive" / Path(*parts[4:])
            return p
        if parts and parts[0] in ("My Drive", "MyDrive"):
            parts = parts[1:]
        return COLAB_DRIVE / "MyDrive" / Path(*parts) if parts else COLAB_DRIVE / "MyDrive"
    return p if p.is_absolute() else local_notebook_folder()[0] / p


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def colab_notebook_folder(mount=True) -> tuple[Path | None, str]:
    """Return (folder, reason). folder is None when the notebook is not in the student's Drive."""
    if mount:
        ensure_drive()
    session = colab_session()
    if session is None:
        return None, "Colab did not say which notebook is open"
    file_id = file_id_of(session)
    if file_id is None:
        return None, "this notebook is not stored in Google Drive (it was opened from GitHub or a link)"
    print(f"Looking for {session.get('name', 'this notebook')} in your Drive...")
    folder = find_by_name(COLAB_DRIVE / "MyDrive", session.get("name", ""), file_id)
    if folder is None:
        try:
            folder = path_from_drive_api(file_id)
        except Exception:
            return None, ("this notebook is not in your own Drive: it belongs to someone else and was shared "
                          "with you, so you cannot save files next to it")
    if not folder.exists():
        return None, ("this notebook is not in your own Drive: it belongs to someone else and was shared with you, "
                      "so you cannot save files next to it")
    return folder, "the folder of this notebook in your Google Drive"


# --------------------------------------------------------------------------- choice
def choose(session: str, folder=None, drive: bool = True) -> tuple[Path, str]:
    """Return (work folder, a plain description of where it is); the folder becomes the current folder."""
    if folder is not None:
        base, where = resolve_folder(folder), "the folder you chose"
        if in_colab() and _under(base, COLAB_DRIVE):
            if not drive:
                raise ValueError("That folder is in Google Drive, so Drive must be connected: leave out drive=False.")
            ensure_drive()                       # never write a look-alike Drive folder on the temporary disk
            where = "the folder you chose in your Google Drive"
    elif in_colab() and drive:
        try:
            base, reason = colab_notebook_folder()
        except Exception as exc:                 # Drive refused or unavailable
            base, reason = None, f"Google Drive could not be connected ({exc.__class__.__name__})"
        if base is not None and not _writable(base):
            base, reason = None, "you do not have permission to save files in the notebook's folder"
        if base is not None:
            where = reason
        elif drive_mounted():
            base = COLAB_DRIVE / "MyDrive" / UNIT / session
            where = (f"a folder in your Google Drive (My Drive › {UNIT} › {session}), because {reason}. "
                     f"{SAVE_A_COPY}")
        else:
            base = COLAB_LOCAL / UNIT / session
            where = (f"Colab's temporary storage, because {reason}. WARNING: these files are deleted when Colab "
                     "disconnects. Run kit.start again and allow Drive access, or download your answer book with "
                     "kit.submit() before you close the notebook")
    elif in_colab():
        base = COLAB_LOCAL / UNIT / session
        where = "Colab's temporary storage (deleted when Colab disconnects), because you chose drive=False"
    else:
        base, where = local_notebook_folder()
    base.mkdir(parents=True, exist_ok=True)
    base = base.resolve()
    os.chdir(base)
    return base, where
