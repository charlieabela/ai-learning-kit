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
from .core.feedback import Feedback, as_feedback, combine
from .core import workfolder
from . import sessions

__all__ = ["start", "task", "answer", "check", "progress", "submit", "sources", "workspace",
           "peek", "path", "extract", "pipeline", "reference", "report", "explain", "source",
           "__version__"]

_state = {"session": None, "book": None, "folder": None}


def start(session: str, student: str, folder=None, drive: bool = True) -> None:
    """Start (or continue) a session's answer book, kept in the same folder as this notebook.

    session : the session code, for example "S02"
    student : your name, used to name the answer file
    folder  : optional; a different folder for your work.
    drive   : in Colab, set drive=False to skip connecting Google Drive (your files
              are then deleted when Colab disconnects).

    The kit finds the folder that holds this notebook: on your computer (Jupyter,
    VS Code) that is where the notebook file is; in Colab it connects your Google
    Drive and finds the notebook there. That folder becomes the current folder, so
    your answer book, the case files and every file you save with a plain name,
    such as "policy_rules.json", are kept next to the notebook.
    """
    mod = sessions.get(session)
    base, where = workfolder.choose(mod.SESSION, folder=folder, drive=drive)
    _state["session"] = mod
    _state["folder"] = base
    print(f"ari1101kit {__version__} · {mod.SESSION}: {mod.TITLE}")
    print(f"Your work folder: {base}")
    print(f"  This is {where}.")
    print("  Your answer book, the case files and every file you save with a plain file name are kept here.")
    _state["book"] = AnswerBook(mod.SESSION, student, mod.TASKS, folder=base)
    print(f"{len(mod.TASKS)} tasks. kit.task('T1.1') explains a task; kit.progress() shows what is left.")


def _require():
    if _state["book"] is None:
        raise RuntimeError("Run kit.start('S02', student='Your Name') first.")
    return _state["session"], _state["book"]


def task(task_id: str) -> None:
    """Explain a task: what to do, why it matters, a typical example and what to record."""
    mod, _ = _require()
    t = mod.TASKS[task_id]
    guide = t.get("guide")
    if guide is None:
        needs = [w for w, k in (("a value", "needs_value"), ("your reasoning", "needs_reasoning")) if t.get(k)]
        print(f"{task_id} · {t['title']}\n{t['question']}\nRecord: {' and '.join(needs)}.")
        return
    text = mod.render_markdown(task_id, t["title"])
    try:
        from IPython import get_ipython
        from IPython.display import Markdown, display
        if get_ipython() is not None:
            display(Markdown(text))
            return
    except ImportError:
        pass
    print(text)


def _feedback(task_id: str, value=None, reasoning=None, attempt=None):
    mod, _ = _require()
    if task_id not in mod.TASKS:
        raise KeyError(f"{task_id} is not a task in {mod.SESSION}. Tasks: {', '.join(mod.TASKS)}")
    t = mod.TASKS[task_id]
    value_fb = as_feedback(t["check"](value)) if value is not None and t.get("check") else None
    written_fb = t["written"](reasoning) if reasoning and t.get("written") else None
    fb = combine(value_fb, written_fb)
    if fb is not None:
        print(fb.render(task_id, attempt))
    return value_fb, written_fb


def check(task_id: str, value=None, reasoning: str | None = None) -> None:
    """Give formative feedback on a value and/or reasoning without saving it or counting an attempt."""
    _feedback(task_id, value, reasoning)


def answer(task_id: str, value=None, reasoning: str | None = None, feedback: bool = True) -> None:
    """Save your answer to a task (a value, your reasoning, or both) and show feedback.

    The answer book is saved after every call, in ari1101_answers/ inside your work
    folder. Calling answer() again for the same task replaces the earlier answer and
    counts as a new attempt. Feedback says what is right, what is not and the
    evidence; for written answers it lists the points a good answer covers. After
    three attempts that still need work, a suggested answer is shown. Answers are
    formative: the number of attempts does not affect any mark.
    """
    _, book = _require()
    attempt = book.record(task_id, value, reasoning)
    if feedback:
        value_fb, written_fb = _feedback(task_id, value, reasoning, attempt)
        book.note_result(task_id, value_fb.ok if value_fb else None,
                         None if written_fb is None else not written_fb.open_points)


def progress() -> pd.DataFrame:
    """Show which tasks have been answered."""
    _, book = _require()
    return book.progress()


def submit(download: bool = True):
    """Finalise the answer book, list anything missing, and download it in Colab."""
    _, book = _require()
    return book.submit(download=download)
