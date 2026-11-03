"""The learner's level: how deep and how long explanations should be, independent of any course.

A topic says what is taught; a level says who is learning it. The same topic can be used at any
level, so a notebook, the kit and a tutor agree on depth through these four names only:

    introductory  new to programming and to data science
    intermediate  comfortable with Python and the basics, new to the topic's methods
    advanced      fluent in Python; interested in method, assumptions and evaluation
    research      working at the edge of the topic; interested in open questions and the literature
"""
from __future__ import annotations

LEVELS = {
    "introductory": {
        "label": "Introductory",
        "learner": ("Learners who are new to programming and to data science, still getting used to Python, NumPy, "
                    "pandas and Matplotlib."),
        "guidance": ("Use plain words and short sentences; define every term the first time it is used; explain the "
                     "Python syntax as well as the idea; prefer one concrete example from the notebook's own data to "
                     "any general statement; keep each explanation to a short paragraph or two."),
    },
    "intermediate": {
        "label": "Intermediate",
        "learner": ("Learners who are comfortable with basic Python and pandas and know the fundamentals of data "
                    "analysis, but are new to the methods of this topic."),
        "guidance": ("Explain the method and why each step is needed; mention Python details only when they are the "
                     "difficulty; connect the idea to what the learner already knows; keep explanations focused."),
    },
    "advanced": {
        "label": "Advanced",
        "learner": ("Learners who are fluent in Python and read code and documentation independently, at the level "
                    "of a postgraduate course."),
        "guidance": ("Focus on the method, its assumptions, evaluation design and pitfalls such as leakage and "
                     "over-claiming; use the notation of the field when it helps; leave Python syntax aside unless "
                     "asked."),
    },
    "research": {
        "label": "Research",
        "learner": ("Learners working at research level, for example doctoral students, who relate a method to the "
                    "literature and to their own problems."),
        "guidance": ("Discuss design choices, alternatives and their trade-offs, the limits of the evidence and open "
                     "questions; point to the kind of work that addresses them; be concise and precise."),
    },
}
DEFAULT_LEVEL = "intermediate"


def normalise_level(level: str | None) -> str:
    key = (level or DEFAULT_LEVEL).strip().lower()
    if key not in LEVELS:
        raise ValueError(f"Unknown level {level!r}. Choose one of: {', '.join(LEVELS)}")
    return key


def describe(level: str | None) -> str:
    """The level as one paragraph for a tutor: who the learners are and how to pitch explanations."""
    entry = LEVELS[normalise_level(level)]
    return f"{entry['label']} level. {entry['learner']} {entry['guidance']}"
