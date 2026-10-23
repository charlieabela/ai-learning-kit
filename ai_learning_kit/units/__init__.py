"""Study units and their sessions.

Each unit is a package in this folder with:
- UNIT: a dict describing the unit (code, title, programme, level, notes);
- SESSIONS: an ordered dict mapping a session code ("S02") to the module that
  defines it, as a dotted path relative to the unit package (".s02").

A session module binds one problem from the problem library to a sequence of
tasks for that unit's learning outcomes. It defines UNIT, SESSION, TITLE, PROBLEM
and TASKS (see docs/AUTHORING_A_SESSION.md). Earlier sessions are never changed
once released, so notebooks pinned to an older release keep working.
"""
from importlib import import_module

REGISTRY = {
    "ARI1101": "ai_learning_kit.units.ari1101",
    "ICS2205": "ai_learning_kit.units.ics2205",
    "ICS5111": "ai_learning_kit.units.ics5111",
}


def normalise_unit(unit: str) -> str:
    return str(unit).strip().upper().replace(" ", "").replace("-", "")


def normalise_session(session: str) -> str:
    key = str(session).strip().upper().replace("SESSION", "S").replace("-", "").replace(" ", "")
    if key.isdigit():
        key = f"S{int(key):02d}"
    if key.startswith("S") and key[1:].isdigit():
        key = f"S{int(key[1:]):02d}"
    return key


def unit(code: str):
    """The unit package registered under a code, e.g. unit("ARI1101")."""
    key = normalise_unit(code)
    if key not in REGISTRY:
        raise KeyError(f"Unknown unit {code!r}. Available: {', '.join(REGISTRY)}")
    return import_module(REGISTRY[key])


def session(unit_code: str, session_code: str):
    """The session module for a unit and session code, e.g. session("ARI1101", "S02")."""
    pkg = unit(unit_code)
    key = normalise_session(session_code)
    sessions = getattr(pkg, "SESSIONS", {})
    if key not in sessions:
        available = ", ".join(sessions) or "none yet"
        raise KeyError(f"Unit {pkg.UNIT['code']} has no session {session_code!r}. Available: {available}")
    return import_module(sessions[key], package=pkg.__name__)
