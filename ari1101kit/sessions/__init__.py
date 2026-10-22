"""Session modules. Each module defines SESSION, TITLE and TASKS.

Adding a session: create sNN.py with the same three names and register it in
REGISTRY below. Earlier sessions are never changed, so notebooks pinned to an
older release keep working.
"""
from importlib import import_module

REGISTRY = {
    "S02": "ari1101kit.sessions.s02",
}


def get(session: str):
    key = session.upper().replace("SESSION", "S").replace("-", "").replace(" ", "")
    if key not in REGISTRY:
        raise KeyError(f"Unknown session {session!r}. Available in this version: {', '.join(REGISTRY)}")
    return import_module(REGISTRY[key])
