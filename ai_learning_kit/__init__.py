"""ai-learning-kit: a sandbox library for the AI study units at the University of Malta.

The kit holds a library of problems (frozen datasets with helper code that does
the heavy lifting) and, for each study unit, sessions that bind a problem to a
sequence of tasks. A session hides the steps that are not its topic, so that
students concentrate on the step that is. Questions are posed per task, answers
are saved in an answer book next to the notebook, and formative feedback is given
at once.

Typical use in a notebook:

    import ai_learning_kit as kit
    kit.start("ARI1101", "S02", student="Your Name")
    kit.workspace()                 # copy the problem's case files next to the notebook
    kit.peek("erp")                 # look at a raw file
    kit.task("T2.1")                # what to do, why, a typical example, what to record
    rules = kit.extract.policy_rules()
    kit.answer("T2.1", value=len(rules), reasoning="...")
    kit.submit()

kit.extract, kit.pipeline, kit.report and kit.reference are the helper modules of
the problem that the current session uses. kit.catalog() lists every unit,
session and problem in this version. Optional: kit.hint() and kit.ask() use a
language model when one is configured (see ai_learning_kit.llm); nothing else does.

All companies, people and documents in the problems are fictional unless a
problem says otherwise.
"""
from __future__ import annotations

import pandas as pd

from ._version import __version__
from .core import registry, workfolder
from .core.answers import AnswerBook
from .core.feedback import Feedback, as_feedback, combine
from .core.help import explain, source
from .core.session import render_guide
from . import llm

__all__ = ["start", "task", "answer", "check", "progress", "submit", "sync", "sources", "workspace",
           "peek", "path", "extract", "pipeline", "reference", "report", "explain", "source",
           "hint", "ask", "catalog", "problem", "session", "llm", "__version__"]

_state = {"session": None, "problem": None, "book": None, "folder": None}


# --------------------------------------------------------------------------- session
def start(unit: str, session: str | None = None, student: str = "", folder=None, drive: bool = True) -> None:
    """Start (or continue) a session: choose the work folder and open your answer book.

    unit    : the study-unit code, for example "ARI1101" (or "ARI1101/S02" in one string)
    session : the session code, for example "S02"
    student : your name, used to name the answer file
    folder  : optional; a different folder for your work. In Colab a relative path
              is a folder in your Google Drive, for example "ARI1101/S02" for
              My Drive › ARI1101 › S02; on your computer it is relative to the
              notebook's folder.
    drive   : in Colab, set drive=False to skip connecting Google Drive (your files
              are then deleted when Colab disconnects).

    The kit finds the folder that holds this notebook: on your computer (Jupyter,
    VS Code) that is where the notebook file is; in Colab it connects your Google
    Drive and finds the notebook there. That folder becomes the current folder, so
    your answer book, the case files and every file you save with a plain name,
    such as "policy_rules.json", are kept next to the notebook.
    """
    if session is None and "/" in str(unit):
        unit, session = str(unit).split("/", 1)
    if session is None:
        raise TypeError('Give the unit and the session, for example kit.start("ARI1101", "S02", student="Your Name").')
    if not student:
        raise TypeError('Give your name: kit.start("ARI1101", "S02", student="Your Name").')
    mod = registry.session(unit, session)
    prob = registry.problem(mod.PROBLEM)
    base, where = workfolder.choose(mod.UNIT, mod.SESSION, folder=folder, drive=drive)
    _state.update(session=mod, problem=prob, folder=base)
    print(f"ai-learning-kit {__version__} · {mod.UNIT} {mod.SESSION}: {mod.TITLE}")
    print(f"Problem: {prob.title}")
    print(f"Your work folder: {base}")
    print(f"  This is {where}.")
    print("  Your answer book, the case files and every file you save with a plain file name are kept here.")
    _state["book"] = AnswerBook(mod.UNIT, mod.SESSION, student, mod.TASKS, folder=base, problem=prob.id)
    first = next(iter(mod.TASKS))
    print(f"{len(mod.TASKS)} tasks. kit.task('{first}') explains a task; kit.progress() shows what is left.")


def _require():
    if _state["book"] is None:
        raise RuntimeError('Run kit.start("ARI1101", "S02", student="Your Name") first.')
    return _state["session"], _state["book"]


def _problem():
    if _state["problem"] is None:
        raise RuntimeError('Run kit.start("<unit>", "<session>", student="Your Name") first: '
                           'the case files belong to the session\'s problem.')
    return _state["problem"]


def session():
    """The current session module (UNIT, SESSION, TITLE, PROBLEM, TASKS)."""
    return _require()[0]


def problem():
    """The current problem (its sources, files and helper modules)."""
    return _problem()


def catalog(include_tasks: bool = False) -> dict:
    """Every unit, session and problem in this version of the kit, as plain data."""
    return registry.catalog(include_tasks=include_tasks)


# --------------------------------------------------------------------------- the problem's files and helpers
def sources() -> pd.DataFrame:
    """List the current problem's evidence sources: short name, file name, size and description."""
    return _problem().sources_table()


def workspace(folder=None, overwrite: bool = False):
    """Copy the current problem's case files into a folder next to this notebook and return its path."""
    return _problem().workspace(folder=folder, overwrite=overwrite)


def peek(name: str, lines: int = 12) -> None:
    """Print the first lines of a source exactly as stored, without interpreting it."""
    return _problem().peek(name, lines=lines)


def path(name: str):
    """The path of a source file inside the installed kit (students normally use workspace())."""
    return _problem().path(name)


class _HelperProxy:
    """kit.extract, kit.pipeline, ...: the current problem's helper module of that name."""

    def __init__(self, namespace: str):
        self._namespace = namespace

    def _module(self):
        return _problem().module(self._namespace)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._module(), name)

    def __dir__(self):
        try:
            return [n for n in dir(self._module()) if not n.startswith("_")]
        except RuntimeError:
            return []

    def __repr__(self):
        try:
            return f"<kit.{self._namespace}: helpers of problem {_problem().id!r}>"
        except RuntimeError:
            return f"<kit.{self._namespace}: no session started>"


extract = _HelperProxy("extract")
pipeline = _HelperProxy("pipeline")
report = _HelperProxy("report")
reference = _HelperProxy("reference")


# --------------------------------------------------------------------------- tasks and answers
def task(task_id: str) -> None:
    """Explain a task: what to do, why it matters, a typical example and what to record."""
    mod, _ = _require()
    t = mod.TASKS[task_id]
    guide = t.get("guide")
    if guide is None:
        needs = [w for w, k in (("a value", "needs_value"), ("your reasoning", "needs_reasoning")) if t.get(k)]
        print(f"{task_id} · {t['title']}\n{t['question']}\nRecord: {' and '.join(needs)}.")
        return
    text = render_guide(task_id, t["title"], guide)
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
        raise KeyError(f"{task_id} is not a task in {mod.UNIT} {mod.SESSION}. Tasks: {', '.join(mod.TASKS)}")
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

    The answer book is saved after every call, in answers/ inside your work folder.
    Calling answer() again for the same task replaces the earlier answer and counts
    as a new attempt. Feedback says what is right, what is not and the evidence; for
    written answers it lists the points a good answer covers. After three attempts
    that still need work, a suggested answer is shown. Answers are formative: the
    number of attempts does not affect any mark.
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


# --------------------------------------------------------------------------- optional language-model help
def hint(task_id: str) -> None:
    """Ask the configured language model for one hint on a task, never the answer.

    The hint is based on the task's guidance and on your last saved answer and its
    feedback. It needs a language model: see ai_learning_kit.llm. Without one, this
    prints how to set one up and points you to kit.task() instead.
    """
    mod, book = _require()
    llm.hint(mod, book, task_id)


def ask(task_id: str, question: str) -> None:
    """Ask the configured language model a question about a task (what a term means, why a step is needed).

    It answers from the task's guidance and the problem's description, and does not
    give the task's answer. Needs a language model: see ai_learning_kit.llm.
    """
    mod, _ = _require()
    llm.ask(mod, _problem(), task_id, question)


# --------------------------------------------------------------------------- saving to Drive and submitting
def sync() -> None:
    """Make sure everything saved so far has reached Google Drive (Colab only).

    Colab copies files to Google Drive in the background, so a file can take a while
    to appear there. sync() sends everything now. kit.submit() calls it for you.
    """
    import os
    from pathlib import Path
    if not workfolder.in_colab() or not workfolder.drive_mounted():
        print("Nothing to send: Google Drive is not connected in this session (files are saved directly).")
        return
    from google.colab import drive as gdrive  # type: ignore
    print("Sending your saved files to Google Drive...")
    os.chdir(workfolder.COLAB_LOCAL)            # Drive cannot be disconnected while we are inside it
    gdrive.flush_and_unmount()
    workfolder.ensure_drive()
    if _state.get("folder") is not None and Path(_state["folder"]).exists():
        os.chdir(_state["folder"])
    print("Done: everything saved so far is in your Google Drive.")


def submit(download: bool = True):
    """Finalise the answer book, list anything missing, and download it in Colab."""
    _, book = _require()
    path = book.submit(download=False)
    if workfolder.in_colab() and workfolder.drive_mounted():
        sync()                                   # make sure the final answer book is in Drive first
    if download:
        book.download()
    return path
