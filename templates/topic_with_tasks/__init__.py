"""<Title>: <one sentence>. Tasks, questions and formative checks on the <problem> problem.

A topic with graded tasks and an answer book. Copy this folder to ai_learning_kit/topics/<name_with_underscores>/,
register the topic in ai_learning_kit/topics/__init__.py and run
    python -m ai_learning_kit.dev.validate --topic <name>
See docs/AUTHORING_A_TOPIC.md.
"""
from __future__ import annotations

from functools import lru_cache

from ...core.checks import all_of, any_of, as_int, at_least, numeric_check, saved_only, written
from ...core.feedback import Feedback
from ...core.session import attach_guides, task
from ...problems.<problem_id> import extract, reference
from .guide import EXAMPLE_CASE, GUIDE

TOPIC = "<name>"                    # lower-case words joined by hyphens, as given to kit.start
TITLE = "<Title: what learners do in this topic>"
LEVEL = "introductory"              # introductory, intermediate, advanced or research
BUILDS_ON = ()
PROBLEM = "<problem_id>"


@lru_cache(maxsize=None)
def _ref():
    """Reference values for the checks, computed once from the problem's data."""
    return {
        "example_count": int(reference.example_reference()),
        # add the common wrong values too, so feedback can name the likely cause:
        # "example_count_without_filter": ...,
    }


# --------------------------------------------------------------------------- written-answer checklists
WRITTEN = {
    "T1.2": written([
        ("It names <point one>.", any_of(r"<regex>", r"<regex>"), "<hint shown when not found>"),
        ("It explains <point two>.", all_of(any_of(r"<a>"), any_of(r"<b>")), "<hint>"),
    ], model="<A model answer that passes every point above.>"),
}

# --------------------------------------------------------------------------- tasks
TASKS = {
    "T1.1": task("<Short title>", "<The question, one or two sentences.>",
                 numeric_check(_ref, "example_count",
                               how_to_check="<a line of code the student can run>",
                               explanation="<why the right value is right>",
                               wrong={"example_count_without_filter": "<likely cause of that wrong value>",
                                      0: "<likely cause of zero>"},
                               evidence="<a fact in the data that helps without giving the number away>",
                               right="<what a correct value means>"),
                 needs_value=True),
    "T1.2": task("<Short title>", "<A question answered in writing.>",
                 needs_reasoning=True, written=WRITTEN["T1.2"]),
    "T1.3": task("<Short title>", "<A question the lecturer reads; no automatic check.>",
                 saved_only, needs_value=True),
}

attach_guides(TASKS, GUIDE, example_case=EXAMPLE_CASE)
