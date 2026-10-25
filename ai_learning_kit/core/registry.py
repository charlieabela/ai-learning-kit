"""Discovery: which units, sessions and problems this version of the kit contains.

catalog() returns plain data (dicts and lists), so that other tools, such as a
notebook tutor, can list what is available without importing the kit's internals.
"""
from __future__ import annotations

from .. import problems as _problems
from .. import units as _units
from .._version import __version__

SESSION_FIELDS = ("UNIT", "SESSION", "TITLE")          # required in every session module
SESSION_DEFAULTS = {                                     # optional, with these defaults
    "PROBLEM": None,      # an id from the problem library, or None for a session without a case problem
    "TASKS": {},          # graded tasks with an answer book; {} when the notebooks check themselves
    "TOOLS": None,        # dotted path (relative to the session) of the helper module kit.tools() returns
    "DATASETS": (),       # external datasets (ai_learning_kit.datasets) that kit.start fetches
    "EXTRAS": "",         # the pip extra the session's notebooks install, e.g. "graphs"
    "NOTEBOOKS": {},      # role ("prepare", "explore", "practice") -> path inside notebooks/
    "SAVES_WORK": None,   # True when students' work must be kept (default: when there are tasks)
}


def unit(code: str):
    return _units.unit(code)


def session(unit_code: str, session_code: str):
    mod = _units.session(unit_code, session_code)
    missing = [f for f in SESSION_FIELDS if not hasattr(mod, f)]
    if missing:
        raise AttributeError(f"Session module {mod.__name__} lacks {', '.join(missing)}")
    for name, default in SESSION_DEFAULTS.items():
        if not hasattr(mod, name):
            setattr(mod, name, type(default)() if isinstance(default, (dict, tuple)) else default)
    if mod.SAVES_WORK is None:
        mod.SAVES_WORK = bool(mod.TASKS)
    return mod


def tools(mod):
    """The helper module of a session (what kit.tools() returns), or None when it has none."""
    if not mod.TOOLS:
        return None
    from importlib import import_module
    package = mod.__name__ if hasattr(mod, "__path__") else mod.__name__.rpartition(".")[0]
    return import_module(mod.TOOLS, package=package)


def _first_paragraph(text) -> str:
    return (text or "").strip().split("\n\n")[0].replace("\n", " ").strip()


def problem(problem_id: str):
    return _problems.get(problem_id)


def task_summary(task_id: str, task: dict) -> dict:
    return {"id": task_id, "title": task["title"], "question": task["question"],
            "needs_value": bool(task.get("needs_value")), "needs_reasoning": bool(task.get("needs_reasoning")),
            "auto_checked": task.get("check") is not None, "written_checklist": task.get("written") is not None}


def session_summary(mod) -> dict:
    return {"unit": mod.UNIT, "session": mod.SESSION, "title": mod.TITLE, "problem": mod.PROBLEM,
            "stages": dict(getattr(mod, "STAGES", {}) or {}),
            "tasks": [task_summary(tid, t) for tid, t in mod.TASKS.items()],
            "tools": bool(mod.TOOLS), "datasets": list(mod.DATASETS), "extras": mod.EXTRAS,
            "notebooks": dict(mod.NOTEBOOKS)}


def tutor_context(unit_code: str, session_code: str) -> dict:
    """What a tutor needs to pitch its help for a session: cohort, problem, scope and tasks."""
    from .session import render_guide, scope_text
    mod = session(unit_code, session_code)
    pkg = unit(unit_code)
    prob = problem(mod.PROBLEM) if mod.PROBLEM else None
    helpers = tools(mod)
    from .. import datasets as _datasets
    return {
        "unit": dict(pkg.UNIT), "session": mod.SESSION, "title": mod.TITLE,
        "cohort": f"{pkg.UNIT['code']} {pkg.UNIT['title']} ({pkg.UNIT['programme']}, {pkg.UNIT['level']}). {pkg.UNIT['notes']}",
        "problem": ({"id": prob.id, "title": prob.title, "question": prob.question, "summary": prob.summary}
                    if prob else None),
        "about": _first_paragraph(mod.__doc__),
        "tools": ({"name": "T = kit.tools()", "summary": _first_paragraph(helpers.__doc__),
                   "functions": sorted(n for n, v in vars(helpers).items()
                                       if callable(v) and not n.startswith("_") and getattr(v, "__module__", "") == helpers.__name__)}
                  if helpers else None),
        "datasets": [_datasets.describe(n) for n in mod.DATASETS],
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
            s = session_summary(mod) if include_tasks else {"session": scode, "title": mod.TITLE, "problem": mod.PROBLEM,
                                                            "notebooks": dict(mod.NOTEBOOKS)}
            entry["sessions"].append(s)
        out["units"].append(entry)
    for pid in _problems.ids():
        out["problems"].append(_problems.get(pid).describe())
    from .. import datasets as _datasets
    out["datasets"] = _datasets.catalog()
    return out
