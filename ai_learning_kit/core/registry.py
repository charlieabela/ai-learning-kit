"""Discovery: which units, sessions and problems this version of the kit contains.

catalog() returns plain data (dicts and lists), so that other tools, such as a
notebook tutor, can list what is available without importing the kit's internals.
"""
from __future__ import annotations

from .. import problems as _problems
from .. import units as _units
from .._version import __version__

SESSION_FIELDS = ("UNIT", "SESSION", "TITLE", "PROBLEM", "TASKS")


def unit(code: str):
    return _units.unit(code)


def session(unit_code: str, session_code: str):
    mod = _units.session(unit_code, session_code)
    missing = [f for f in SESSION_FIELDS if not hasattr(mod, f)]
    if missing:
        raise AttributeError(f"Session module {mod.__name__} lacks {', '.join(missing)}")
    return mod


def problem(problem_id: str):
    return _problems.get(problem_id)


def task_summary(task_id: str, task: dict) -> dict:
    return {"id": task_id, "title": task["title"], "question": task["question"],
            "needs_value": bool(task.get("needs_value")), "needs_reasoning": bool(task.get("needs_reasoning")),
            "auto_checked": task.get("check") is not None, "written_checklist": task.get("written") is not None}


def session_summary(mod) -> dict:
    return {"unit": mod.UNIT, "session": mod.SESSION, "title": mod.TITLE, "problem": mod.PROBLEM,
            "stages": dict(getattr(mod, "STAGES", {}) or {}),
            "tasks": [task_summary(tid, t) for tid, t in mod.TASKS.items()]}


def tutor_context(unit_code: str, session_code: str) -> dict:
    """What a tutor needs to pitch its help for a session: cohort, problem, scope and tasks."""
    from .session import render_guide, scope_text
    mod = session(unit_code, session_code)
    pkg = unit(unit_code)
    prob = problem(mod.PROBLEM)
    return {
        "unit": dict(pkg.UNIT), "session": mod.SESSION, "title": mod.TITLE,
        "cohort": f"{pkg.UNIT['code']} {pkg.UNIT['title']} ({pkg.UNIT['programme']}, {pkg.UNIT['level']}). {pkg.UNIT['notes']}",
        "problem": {"id": prob.id, "title": prob.title, "question": prob.question, "summary": prob.summary},
        "scope": scope_text(mod),
        "stages": dict(getattr(mod, "STAGES", {}) or {}),
        "tasks": {tid: {"title": t["title"], "question": t["question"],
                        "guide": render_guide(tid, t["title"], t["guide"]) if t.get("guide") else None}
                  for tid, t in mod.TASKS.items()},
    }


def catalog(include_tasks: bool = True) -> dict:
    """Everything this version of the kit offers, as plain data."""
    out = {"kit": "ai-learning-kit", "version": __version__, "units": [], "problems": []}
    for code in _units.REGISTRY:
        pkg = _units.unit(code)
        entry = dict(pkg.UNIT)
        entry["sessions"] = []
        for scode in getattr(pkg, "SESSIONS", {}):
            mod = session(code, scode)
            s = session_summary(mod) if include_tasks else {"session": scode, "title": mod.TITLE, "problem": mod.PROBLEM}
            entry["sessions"].append(s)
        out["units"].append(entry)
    for pid in _problems.ids():
        out["problems"].append(_problems.get(pid).describe())
    return out
