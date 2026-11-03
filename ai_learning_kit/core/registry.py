"""Discovery: which topics, problems and datasets this version of the kit contains.

catalog() returns plain data (dicts and lists), so that other tools, such as a
notebook tutor, can list what is available without importing the kit's internals.
"""
from __future__ import annotations

from .. import problems as _problems
from .. import topics as _topics
from .._version import __version__
from . import levels as _levels

TOPIC_FIELDS = ("TOPIC", "TITLE")                       # required in every topic module
TOPIC_DEFAULTS = {                                       # optional, with these defaults
    "PROBLEM": None,      # an id from the problem library, or None for a topic without a case problem
    "TASKS": {},          # graded tasks with an answer book; {} when the notebooks check themselves
    "TOOLS": None,        # dotted path (relative to the topic) of the helper module kit.tools() returns
    "DATASETS": (),       # external datasets (ai_learning_kit.datasets) that kit.start fetches
    "EXTRAS": "",         # the pip extra the topic's notebooks install, e.g. "graphs"
    "SAVES_WORK": None,   # True when learners' work must be kept (default: when there are tasks)
    "BUILDS_ON": (),      # topics whose ideas are assumed
}


def topic(name: str):
    """A topic module with every optional field filled in (see ai_learning_kit.topics)."""
    mod = _topics.topic(name)
    missing = [f for f in TOPIC_FIELDS if not hasattr(mod, f)]
    if missing:
        raise AttributeError(f"Topic module {mod.__name__} lacks {', '.join(missing)}")
    if not hasattr(mod, "LEVEL"):
        mod.LEVEL = _levels.DEFAULT_LEVEL
    for name, default in TOPIC_DEFAULTS.items():
        if not hasattr(mod, name):
            setattr(mod, name, type(default)() if isinstance(default, (dict, tuple)) else default)
    if mod.SAVES_WORK is None:
        mod.SAVES_WORK = bool(mod.TASKS)
    return mod


def tools(mod):
    """The helper module of a topic (what kit.tools() returns), or None when it has none."""
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


def topic_summary(mod) -> dict:
    return {"topic": mod.TOPIC, "title": mod.TITLE, "level": mod.LEVEL, "builds_on": list(mod.BUILDS_ON),
            "problem": mod.PROBLEM, "stages": dict(getattr(mod, "STAGES", {}) or {}),
            "tasks": [task_summary(tid, t) for tid, t in mod.TASKS.items()],
            "tools": bool(mod.TOOLS), "datasets": list(mod.DATASETS), "extras": mod.EXTRAS}


def tutor_context(name: str, level: str | None = None) -> dict:
    """What a tutor needs to pitch its help: the learner's level, and the topic's problem, scope and tasks.

    tutor_context("knowledge-graphs", level="advanced"); level defaults to the topic's own LEVEL.
    """
    from .session import render_guide, scope_text
    mod = topic(name)
    lvl = _levels.normalise_level(level or mod.LEVEL)
    head = {"topic": mod.TOPIC, "title": mod.TITLE, "level": lvl, "builds_on": list(mod.BUILDS_ON),
            "cohort": _levels.describe(lvl)}
    prob = problem(mod.PROBLEM) if mod.PROBLEM else None
    helpers = tools(mod)
    from .. import datasets as _datasets
    return {
        **head,
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
    out = {"kit": "ai-learning-kit", "version": __version__, "topics": [], "problems": [],
           "levels": {k: v["label"] for k, v in _levels.LEVELS.items()}}
    for name in _topics.names():
        mod = topic(name)
        out["topics"].append(topic_summary(mod) if include_tasks else
                             {"topic": mod.TOPIC, "title": mod.TITLE, "level": mod.LEVEL})
    for pid in _problems.ids():
        out["problems"].append(_problems.get(pid).describe())
    from .. import datasets as _datasets
    out["datasets"] = _datasets.catalog()
    return out
