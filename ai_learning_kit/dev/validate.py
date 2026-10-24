"""Check that a problem or a session meets the kit's contracts.

    python -m ai_learning_kit.dev.validate                    # everything registered
    python -m ai_learning_kit.dev.validate --problem brightwater_q3
    python -m ai_learning_kit.dev.validate --session ARI1101 S02

Each check prints OK or a problem description; the exit code is non-zero when
anything fails. Run it before opening a pull request that adds or changes a
problem or a session.
"""
from __future__ import annotations

import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from ..core import registry
from ..core.feedback import Feedback
from ..core.session import GUIDE_PARTS
from .. import problems, units


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


# --------------------------------------------------------------------------- sessions
def validate_session(unit_code: str, session_code: str) -> Report:
    r = Report()
    try:
        mod = registry.session(unit_code, session_code)
    except Exception as e:
        r.fail(f"{unit_code} {session_code}: cannot load ({e})")
        return r
    pkg = units.unit(unit_code)
    r.check(mod.UNIT == pkg.UNIT["code"], f"UNIT is {pkg.UNIT['code']!r}")
    r.check(mod.SESSION == units.normalise_session(session_code), f"SESSION is {mod.SESSION!r}")
    r.check(bool(mod.TITLE.strip()), "TITLE is filled in")
    r.check(mod.PROBLEM in problems.REGISTRY, f"PROBLEM {mod.PROBLEM!r} is in the problem library")
    r.check(len(mod.TASKS) > 0, f"{len(mod.TASKS)} tasks")
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
    return r


def validate_all() -> int:
    total = 0
    for pid in problems.ids():
        print(f"Problem {pid}")
        rep = validate_problem(pid)
        print(rep)
        total += rep.failures
    for u in units.REGISTRY:
        for code in units.unit(u).SESSIONS:
            print(f"Session {u} {code}")
            rep = validate_session(u, code)
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
    if argv[0] == "--session" and len(argv) == 3:
        rep = validate_session(argv[1], argv[2]); print(rep); return 1 if rep.failures else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
