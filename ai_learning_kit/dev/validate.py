"""Check that a problem or a topic meets the kit's contracts.

    python -m ai_learning_kit.dev.validate                    # everything registered
    python -m ai_learning_kit.dev.validate --problem brightwater_q3
    python -m ai_learning_kit.dev.validate --topic knowledge-graphs
    python -m ai_learning_kit.dev.validate --notebook path/to/practice.ipynb   # any notebook file

Each check prints OK or a problem description; the exit code is non-zero when
anything fails. Run it before opening a pull request that adds or changes a
problem or a topic. Running it with no arguments also checks that no file in the
package names a study unit or a numbered course session: topics are independent of
any course.
"""
from __future__ import annotations

import re
import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from ..core import registry
from ..core.feedback import Feedback
from ..core.session import GUIDE_PARTS
from .. import problems, topics
from ..core import levels


class Report:
    def __init__(self):
        self.lines, self.failures = [], 0

    def ok(self, msg):
        self.lines.append(f"  OK   {msg}")

    def fail(self, msg):
        self.failures += 1
        self.lines.append(f"  FAIL {msg}")

    def check(self, cond, msg):
        (self.ok if cond else self.fail)(msg)

    def __str__(self):
        return "\n".join(self.lines)


# --------------------------------------------------------------------------- problems
def validate_problem(problem_id: str) -> Report:
    r = Report()
    try:
        p = problems.get(problem_id)
    except Exception as e:
        r.fail(f"{problem_id}: cannot load PROBLEM ({e})")
        return r
    r.check(p.id == problem_id, f"id matches the registry key ({p.id!r})")
    for field in ("title", "question", "summary", "domain"):
        r.check(bool(getattr(p, field, "").strip()), f"{field} is filled in")
    r.check(p.data_dir.is_dir(), f"data folder exists: {p.data_dir}")
    for name, fname in p.sources.items():
        f = p.data_dir / fname
        r.check(f.is_file() and f.stat().st_size > 0, f"source {name!r}: {fname} exists and is not empty")
        r.check(name in p.descriptions, f"source {name!r} has a description")
    for name in p.text_sources:
        r.check(name in p.sources, f"text source {name!r} is a registered source")
    for name in p.sources:
        if name not in p.text_sources:
            r.check(name in p.peek_hints, f"binary source {name!r} has a peek hint (which extractor to use)")
    for ns, modpath in p.modules.items():
        try:
            p.module(ns)
            r.ok(f"helper module {ns!r} imports ({modpath})")
        except Exception as e:
            r.fail(f"helper module {ns!r} does not import: {e}")
    try:
        with redirect_stdout(StringIO()):
            for name in p.sources:
                p.peek(name, lines=2)
        r.ok("peek() works for every source")
    except Exception as e:
        r.fail(f"peek() failed: {e}")
    try:
        with tempfile.TemporaryDirectory() as d, redirect_stdout(StringIO()):
            folder = p.workspace(Path(d) / "ws")
            r.check(sorted(x.name for x in folder.iterdir()) == sorted(p.sources.values()), "workspace() copies every source")
    except Exception as e:
        r.fail(f"workspace() failed: {e}")
    try:
        import json
        json.dumps(p.describe())
        r.ok("describe() is JSON-able")
    except Exception as e:
        r.fail(f"describe() is not JSON-able: {e}")
    return r


# --------------------------------------------------------------------------- topics
COURSE_CODE = re.compile(r"\b[A-Z]{3}\s?\d{4}\b")       # a study-unit code: three capitals and four digits
COURSE_SESSION = re.compile(r"\b[Ss]essions? \d{1,2}\b")  # "Session" or "Sessions" followed by a number
TEXT_SUFFIXES = (".py", ".json", ".md", ".html", ".txt", ".csv", ".ttl", ".toml", ".yml", ".yaml", ".cfg")


def course_references(folder: Path) -> list:
    """Every study-unit code or numbered course session in the text files under folder, as 'file: match'."""
    hits = []
    for f in sorted(folder.rglob("*")):
        if f.is_file() and f.suffix in TEXT_SUFFIXES and "__pycache__" not in f.parts:
            text = f.read_text(encoding="utf-8", errors="ignore")
            for pattern in (COURSE_CODE, COURSE_SESSION):
                hits += [f"{f.relative_to(folder)}: {m.group(0)}" for m in pattern.finditer(text)]
    return hits


def validate_package_text() -> Report:
    """No file of the installed package names a study unit or a numbered course session."""
    r = Report()
    hits = course_references(Path(__file__).resolve().parents[1])
    r.check(not hits, "no study-unit codes or numbered course sessions in the package"
            + (f" ({'; '.join(hits[:8])}{' ...' if len(hits) > 8 else ''})" if hits else ""))
    return r


def validate_topic(name: str) -> Report:
    r = Report()
    try:
        mod = registry.topic(name)
    except Exception as e:
        r.fail(f"{name}: cannot load ({e})")
        return r
    key = topics.normalise_topic(name)
    r.check(mod.TOPIC == key, f"TOPIC is {key!r}")
    r.check(re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", mod.TOPIC or "") is not None,
            "TOPIC is lower case words joined by hyphens")
    r.check(mod.LEVEL in levels.LEVELS, f"LEVEL {mod.LEVEL!r} is one of {', '.join(levels.LEVELS)}")
    later = [b for b in mod.BUILDS_ON if b not in topics.REGISTRY]
    r.ok(f"builds on {', '.join(mod.BUILDS_ON) or 'nothing'}"
         + (f" ({', '.join(later)} not a topic yet)" if later else ""))
    hits = course_references(Path(mod.__file__).parent)
    r.check(not hits, "no study-unit codes or numbered course sessions in the topic's files"
            + (f" ({'; '.join(hits[:5])})" if hits else ""))
    _validate_body(r, mod)
    return r


def _validate_body(r: Report, mod) -> None:
    r.check(bool(mod.TITLE.strip()), "TITLE is filled in")
    if mod.PROBLEM is None:
        r.ok("no case problem (PROBLEM = None)")
    else:
        r.check(mod.PROBLEM in problems.REGISTRY, f"PROBLEM {mod.PROBLEM!r} is in the problem library")
    r.check(len(mod.TASKS) > 0 or bool(mod.TOOLS),
            f"{len(mod.TASKS)} tasks" + (", helper module " + mod.TOOLS if mod.TOOLS else ""))
    if mod.TOOLS:
        try:
            helpers = registry.tools(mod)
            r.check(bool((helpers.__doc__ or "").strip()), f"helper module {mod.TOOLS} imports and has a docstring")
        except Exception as e:
            r.fail(f"helper module {mod.TOOLS} does not import: {e}")
    from .. import datasets
    for name in mod.DATASETS:
        try:
            s = datasets.source(name)
            r.check(bool(s.url and s.sha256 and s.owner and s.licence), f"dataset {name!r}: owner, address, checksum "
                                                                         f"and licence note recorded")
        except KeyError as e:
            r.fail(f"dataset {name!r} is not registered in ai_learning_kit.datasets: {e}")
    try:
        import json
        json.dumps(registry.tutor_context(mod.TOPIC))
        r.ok("tutor_context() is JSON-able")
    except Exception as e:
        r.fail(f"tutor_context() failed: {e}")
    stages = getattr(mod, "STAGES", None)
    if stages:
        from ..core.session import VISIBILITY
        r.check(all(st.get("visibility") in VISIBILITY and st.get("label") for st in stages.values()),
                f"{len(stages)} stages, each with a label and a valid visibility")
    else:
        r.ok("no STAGES declared (optional; a tutor then cannot tell focus from black boxes)")
    needs_check = 0
    for tid, t in mod.TASKS.items():
        for key in ("title", "question"):
            r.check(bool(str(t.get(key, "")).strip()), f"{tid}: {key} is filled in")
        g = t.get("guide")
        r.check(isinstance(g, dict) and all(str(g.get(k, "")).strip() for k in GUIDE_PARTS),
                f"{tid}: guidance has all four parts (what, why, example, record)")
        r.check(t.get("needs_value") or t.get("needs_reasoning"), f"{tid}: records a value or reasoning")
        if t.get("needs_value"):
            r.check(callable(t.get("check")), f"{tid}: has a check for the value")
        if t.get("needs_reasoning"):
            r.check(callable(t.get("written")), f"{tid}: has a checklist for the reasoning")
        w = t.get("written")
        if callable(w):
            try:
                empty = w("")
                r.check(isinstance(empty, Feedback) and empty.open_points, f"{tid}: checklist flags an empty answer")
                model = getattr(w, "model", None) or empty.suggestion
                r.check(bool(model), f"{tid}: checklist has a model answer")
                if model:
                    r.check(not w(model).open_points, f"{tid}: the model answer passes its own checklist")
            except Exception as e:
                r.fail(f"{tid}: checklist raised {e}")
        c = t.get("check")
        if callable(c):
            needs_check += 1
            try:
                fb = c("definitely-not-a-valid-answer")
                r.check(isinstance(fb, Feedback) or (isinstance(fb, tuple) and len(fb) == 2),
                        f"{tid}: check returns Feedback for a wrong-typed value")
            except Exception as e:
                r.fail(f"{tid}: check raised on a wrong-typed value: {e}")


def validate_all() -> int:
    total = 0
    for pid in problems.ids():
        print(f"Problem {pid}")
        rep = validate_problem(pid)
        print(rep)
        total += rep.failures
    for name in topics.names():
        print(f"Topic {name}")
        rep = validate_topic(name)
        print(rep)
        total += rep.failures
    print("Package")
    rep = validate_package_text()
    print(rep)
    total += rep.failures
    print(f"\n{'All checks passed.' if total == 0 else f'{total} check(s) failed.'}")
    return 1 if total else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return validate_all()
    if argv[0] == "--problem" and len(argv) == 2:
        rep = validate_problem(argv[1]); print(rep); return 1 if rep.failures else 0
    if argv[0] == "--notebook" and len(argv) == 2:
        from .notebooks import check_notebook
        fails = 0
        for ok, msg in check_notebook(Path(argv[1])):
            print(f"  {'OK  ' if ok else 'FAIL'} {msg}")
            fails += not ok
        return 1 if fails else 0
    if argv[0] == "--topic" and len(argv) == 2:
        rep = validate_topic(argv[1]); print(rep); return 1 if rep.failures else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
