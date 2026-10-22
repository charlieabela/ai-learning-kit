"""Where a student's work is kept.

In Colab the notebook runs on a temporary machine: anything written there is
deleted when the session disconnects. So, in Colab, the kit connects the
student's Google Drive and keeps the work in My Drive/ARI1101/<session>. On a
student's own computer (Jupyter, VS Code) the work is kept in the folder the
notebook runs in, which is normally the notebook's own folder.

kit.start() makes the work folder the current folder, so every file saved with
a plain name, for example df.to_csv("q3_high_value.csv"), lands there too.
"""
from __future__ import annotations

import os
from pathlib import Path

COLAB_DRIVE = Path("/content/drive")      # where Colab mounts Google Drive
COLAB_LOCAL = Path("/content")            # Colab's temporary disk
UNIT = "ARI1101"


def in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def choose(session: str, folder=None, drive: bool = True) -> tuple[Path, str]:
    """Return (work folder, a plain description of where it is), creating the folder."""
    if folder is not None:
        base = Path(folder).expanduser()
        where = "the folder you chose"
    elif in_colab() and drive:
        try:
            from google.colab import drive as gdrive  # type: ignore
            if not (COLAB_DRIVE / "MyDrive").exists():
                print("Connecting your Google Drive so that your work is kept after Colab disconnects.")
                print("A window will ask you to allow access: choose your account and allow it.")
                gdrive.mount(str(COLAB_DRIVE))
            base = COLAB_DRIVE / "MyDrive" / UNIT / session
            where = f"your Google Drive, in My Drive › {UNIT} › {session}"
        except Exception as exc:  # the student declined, or Drive is unavailable
            print(f"Google Drive could not be connected ({exc.__class__.__name__}).")
            base = COLAB_LOCAL / UNIT / session
            where = ("Colab's temporary storage. WARNING: these files are deleted when Colab disconnects. "
                     "Run kit.start again and allow Drive access, or download your answer book with kit.submit() "
                     "before you close the notebook")
    elif in_colab():
        base = COLAB_LOCAL / UNIT / session
        where = "Colab's temporary storage (deleted when Colab disconnects)"
    else:
        base = Path.cwd()
        where = "the folder this notebook runs in"
    base.mkdir(parents=True, exist_ok=True)
    base = base.resolve()
    os.chdir(base)
    return base, where
