"""ari1101kit: a sandbox for ARI1101 Foundations of Data Science.

One running case, the Q3 2026 expenditure compliance review at the fictional
Brightwater Logistics Ltd, is used across the unit's sessions. The kit hides
the steps that are not the topic of the current session, so that you can
concentrate on the step that is.

Typical use in a notebook:

    import ari1101kit as kit
    kit.start("S02", student="Your Name")
    kit.workspace()                 # copy the case files into ./brightwater_q3
    kit.peek("erp")                 # look at a raw file
    rules = kit.extract.policy_rules()
    kit.answer("T2.1", value=len(rules), reasoning="...")
    kit.submit()

All companies, people and documents in the case are fictional.
"""
from __future__ import annotations

import pandas as pd

from ._version import __version__
from .core import extract, pipeline, reference, report
from .core.case import path, peek, sources, workspace
from .core.help import explain, source
from .core.answers import AnswerBook
from . import sessions

__all__ = ["start", "task", "answer", "check", "progress", "submit", "sources", "workspace",
           "peek", "path", "extract", "pipeline", "reference", "report", "explain", "source",
           "__version__"]

_state = {"session": None, "book": None}


def start(session: str, student: str) -> None:
    """Start (or continue) a session's answer book.

    session : the session code, for example "S02"
    student : your name, used to name the answer file
    """
    mod = sessions.get(session)
    _state["session"] = mod
    _state["book"] = AnswerBook(mod.SESSION, student, mod.TASKS)
    print(f"ari1101kit {__version__} · {mod.SESSION}: {mod.TITLE}")
    print(f"{len(mod.TASKS)} tasks. Use kit.task('T1.1') to read a question and kit.progress() to see what is left.")


def _require():
    if _state["book"] is None:
        raise RuntimeError("Run kit.start('S02', student='Your Name') first.")
    return _state["session"], _state["book"]


def task(task_id: str) -> None:
    """Print the question for a task."""
    mod, _ = _require()
    t = mod.TASKS[task_id]
    needs = [w for w, k in (("a value", "needs_value"), ("your reasoning", "needs_reasoning")) if t.get(k)]
    print(f"{task_id} · {t['title']}\n{t['question']}\nRecord: {' and '.join(needs)}.")


def check(task_id: str, value) -> None:
    """Give formative feedback on a value without saving it."""
    mod, _ = _require()
    fn = mod.TASKS[task_id].get("check")
    if fn is None:
        print("No automatic feedback for this task.")
        return
    ok, msg = fn(value)
    mark = {True: "✓", False: "✗", None: "•"}[ok]
    print(f"{mark} {task_id}: {msg}")


def answer(task_id: str, value=None, reasoning: str | None = None, feedback: bool = True) -> None:
    """Save your answer to a task (a value, your reasoning, or both) and show feedback.

    The answer book is saved after every call, in the folder ari1101_answers/.
    Calling answer() again for the same task replaces the earlier answer.
    """
    mod, book = _require()
    book.record(task_id, value, reasoning)
    if feedback and value is not None:
        check(task_id, value)


def progress() -> pd.DataFrame:
    """Show which tasks have been answered."""
    _, book = _require()
    return book.progress()


def submit(download: bool = True):
    """Finalise the answer book, list anything missing, and download it in Colab."""
    _, book = _require()
    return book.submit(download=download)
