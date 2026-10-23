"""The problem library.

Each problem is a folder in this package holding a `case.py` that defines PROBLEM
(an ai_learning_kit.core.problem.Problem), a `data/` folder with the frozen files,
and the helper modules that do the heavy lifting (extraction, reference values,
pipelines, reporting). Problems are independent of study units: a unit's session
picks a problem and decides which steps students do themselves.

Adding a problem: copy templates/problem/ to a new folder here, fill it in, add
its id below, and run `python -m ai_learning_kit.dev.validate --problem <id>`.
See docs/AUTHORING_A_PROBLEM.md and docs/prompts/new_problem.md.
"""
from importlib import import_module

REGISTRY = {
    "brightwater_q3": "ai_learning_kit.problems.brightwater_q3",
}


def get(problem_id: str):
    """Return the Problem object registered under problem_id."""
    if problem_id not in REGISTRY:
        raise KeyError(f"Unknown problem {problem_id!r}. Available: {', '.join(REGISTRY)}")
    return import_module(REGISTRY[problem_id]).PROBLEM


def ids() -> list:
    return list(REGISTRY)
