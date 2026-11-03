"""Helpers for writing a session module. See docs/AUTHORING_A_SESSION.md.

A session module defines:
    UNIT     = "ARI1101"
    SESSION  = "S02"
    TITLE    = "Data formats: ..."
    PROBLEM  = "brightwater_q3"          # an id from the problem library
    TASKS    = {"T1.1": task(...), ...}  # ordered; ids are shown to students
and, usually, a GUIDE dict in a companion module with four parts per task,
attached with attach_guides(TASKS, GUIDE, example_case=...).

Optionally (and recommended), STAGES describes how visible each stage of the
problem is in this session, so that a tutor knows how deeply to explain it:
    STAGES = {
        "extract": stage("Extracting rules and entities from the documents", BLACK_BOX,
                         how="kit.extract", taught_in="ICS2205 Part A"),
        "formats": stage("Reading CSV, JSON and XML", FOCUS),
        "cleaning": stage("Cleaning and de-duplication", OUT_OF_SCOPE, taught_in="ARI1101 Scrub session"),
    }
FOCUS: students do it; explain in depth. BLACK_BOX: the kit does it; say what goes
in, what comes out and why, not how. OUT_OF_SCOPE: do not bring it up; if asked,
say briefly where it is taught.
"""
from __future__ import annotations

GUIDE_PARTS = ("what", "why", "example", "record")

FOCUS, BLACK_BOX, OUT_OF_SCOPE = "focus", "black_box", "out_of_scope"
VISIBILITY = (FOCUS, BLACK_BOX, OUT_OF_SCOPE)


def stage(label: str, visibility: str, how: str | None = None, taught_in: str | None = None,
          note: str | None = None) -> dict:
    """One stage of the problem's workflow and how visible it is in this session.

    how       : where the kit does it (e.g. "kit.extract.policy_rules"), for black boxes
    taught_in : where students meet it properly (e.g. "ICS5111 Session 03")
    note      : anything the tutor should know (e.g. "say only that embeddings place similar nodes close together")
    """
    if visibility not in VISIBILITY:
        raise ValueError(f"visibility must be one of {VISIBILITY}")
    return {"label": label, "visibility": visibility, "how": how, "taught_in": taught_in, "note": note}


def scope_text(mod) -> str:
    """The session's stages as instructions for a tutor (empty when the session declares none)."""
    stages = getattr(mod, "STAGES", None) or {}
    if not stages:
        return ""
    lines = ["SCOPE OF THIS SESSION (follow it when you explain):"]
    groups = ((FOCUS, "In focus: students do these themselves. Explain them in depth, with the Python mechanics and the idea."),
              (BLACK_BOX, "Black boxes: the kit does these for the students. Say what goes in, what comes out and why the "
                          "step is needed; do not explain how it works inside, even if asked; say where it is taught."),
              (OUT_OF_SCOPE, "Out of scope: do not bring these up. If the student asks, answer in one sentence and say where "
                             "it is taught."))
    for vis, head in groups:
        items = [st for st in stages.values() if st["visibility"] == vis]
        if not items:
            continue
        lines.append(head)
        for st in items:
            extra = [x for x in (f"done by {st['how']}" if st.get("how") else None,
                                 f"taught in {st['taught_in']}" if st.get("taught_in") else None,
                                 st.get("note")) if x]
            lines.append(f"- {st['label']}" + (f" ({'; '.join(extra)})" if extra else ""))
    return "\n".join(lines)


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
