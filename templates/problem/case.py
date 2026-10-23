"""<Problem title>: problem definition and sources.

Copy this folder to ai_learning_kit/problems/<problem_id>/, fill in every field,
put the frozen data files in data/, add the id to ai_learning_kit/problems/REGISTRY,
and run:  python -m ai_learning_kit.dev.validate --problem <problem_id>
"""
from __future__ import annotations

from ...core.problem import Problem

PROBLEM = Problem(
    id="<problem_id>",                       # the folder name; lowercase, underscores
    title="<One-line title students see>",
    question="<The question the problem asks, in one sentence.>",
    summary=("<A short paragraph: the setting (fictional organisation), the sources and their formats, "
             "and what makes the question non-trivial: planted issues, boundary cases, a claim to check.>"),
    domain="<e.g. health / scheduling>",
    sources={
        "<short_name>": "<file name in data/>",
    },
    descriptions={
        "<short_name>": "<one line: what the source is and where it came from in the story>",
    },
    text_sources={"<short names of CSV/JSON/XML/TXT sources that peek() can print>"},
    peek_hints={"<binary short name>": "kit.extract.<function>()"},
    modules={
        "extract": "ai_learning_kit.problems.<problem_id>.extract",
        "reference": "ai_learning_kit.problems.<problem_id>.reference",
        # "pipeline": ..., "report": ...   (only the namespaces the problem provides)
    },
    workspace_name="<problem_id>",
    tags=["<formats>", "<skills>", "<level>"],
    fictional=True,
)

SOURCES = PROBLEM.sources


def path(name: str):
    return PROBLEM.path(name)
