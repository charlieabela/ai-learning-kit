"""Helpers for writing a session module. See docs/AUTHORING_A_SESSION.md.

A session module defines:
    UNIT     = "ARI1101"
    SESSION  = "S02"
    TITLE    = "Data formats: ..."
    PROBLEM  = "brightwater_q3"          # an id from the problem library
    TASKS    = {"T1.1": task(...), ...}  # ordered; ids are shown to students
and, usually, a GUIDE dict in a companion module with four parts per task,
attached with attach_guides(TASKS, GUIDE, example_case=...).
"""
from __future__ import annotations

GUIDE_PARTS = ("what", "why", "example", "record")


def task(title: str, question: str, check=None, needs_value: bool = False,
         needs_reasoning: bool = False, written=None) -> dict:
    """One task: what is shown with kit.task(), what is recorded, and how it is checked.

    check   : function(value) -> Feedback, for the recorded value
    written : function(text) -> Feedback, for the written reasoning (see core.checks.written)
    """
    return dict(title=title, question=question, check=check, needs_value=needs_value,
                needs_reasoning=needs_reasoning, written=written, guide=None)


def attach_guides(tasks: dict, guide: dict, example_case: str) -> None:
    """Attach the four-part guidance to each task and record what the typical example is about."""
    for tid, t in tasks.items():
        if tid not in guide:
            raise KeyError(f"No guidance for task {tid}")
        g = dict(guide[tid])
        g["example_case"] = example_case
        t["guide"] = g


def render_guide(task_id: str, title: str, guide: dict) -> str:
    """The guidance for one task as Markdown, as shown in the notebook and by kit.task()."""
    case = guide.get("example_case")
    label = f"**A typical example** ({case})" if case else "**A typical example**"
    return "\n\n".join([
        f"### Task {task_id} · {title}",
        "**What you do**\n\n" + guide["what"].strip("\n"),
        "**Why it matters**\n\n" + guide["why"].strip("\n"),
        label + "\n\n" + guide["example"].strip("\n"),
        "**What to record**\n\n" + guide["record"].strip("\n"),
    ])
