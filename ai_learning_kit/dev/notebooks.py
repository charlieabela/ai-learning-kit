"""The notebook contract: what a notebook built on the kit must look like, and the check for it.

Notebooks are not stored in the kit: the lecturer keeps and distributes them (Drive, the VLE), and
students open them in Notebook Tutor, which keeps their copy in its clone, or in Colab. The kit holds
what notebooks depend on (helpers, data sources, reference results), and this module checks a
notebook file wherever it is:

    python -m ai_learning_kit.dev.notebooks check  1_prepare_setup.ipynb ...
    python -m ai_learning_kit.dev.notebooks stamp 1.0.0 1_prepare_setup.ipynb ...   # install line + metadata

A kit notebook runs unchanged in Notebook Tutor and in Colab, and (best effort) in Jupyter.
That works because everything that depends on where it runs lives in the kit, not in the
notebook. The contract (docs/NOTEBOOK_CONTRACT.md explains each rule):

1. metadata["ai_learning_kit"] records topic, title, role, kit_version and install (and,
   optionally, level), so that a tutor recognises the notebook.
2. The first code cell is exactly the install line for this kit version, with the topic's
   pip extra (install_line()). Notebook Tutor skips it, because it ships the kit.
3. The topic is started with kit.start("<topic>"...) in one of the first three code cells, before
   any other use of the kit, and T = kit.tools() follows it when the topic has helpers. New
   notebooks put both in the second code cell.
4. No code that depends on the environment: no Google Drive or Colab calls, no uploads, no
   ZIP handling, no sys.path changes, no further pip installs, no hard-coded /content paths.
   Data comes from the kit (ai_learning_kit.datasets or the topic's helpers).
5. Every T.<name> the notebook uses exists in the topic's helper module.
6. Saved outputs contain no errors, except in cells tagged "raises-exception".
7. The notebook names the kit version it was checked with (stamp() writes it).
8. Notebooks are shared with learners and others: no cell names a study unit or a numbered course
   session, and none links to the lecturer's Google Drive.

Advice, printed as NOTE and never a failure: in a practice notebook, each exercise has a plan from question to
code (plan_notes()).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .. import _version

REPO = "https://github.com/charlieabela/ai-learning-kit"
ROLES = ("setup", "prepare", "explore", "practice", "complete", "solutions")
# setup: checks the learner's environment; explore: the walkthrough; complete: one notebook covering every stage
LEVELS = ("introductory", "intermediate", "advanced", "research")

FORBIDDEN = [
    (r"\bgoogle\.colab\b|\bdrive\.mount\(", "uses Colab or Google Drive directly (the kit handles both)"),
    (r"\bfiles\.upload\(", "asks for an upload (data must come from the kit)"),
    (r"\bzipfile\b|\.zip\b", "handles ZIP files (resources come from the kit, not from ZIPs)"),
    (r"sys\.path\.(insert|append)", "changes sys.path (helpers come from kit.tools())"),
    (r"/content/drive|MyDrive", "hard-codes a Google Drive path"),
    (r"(?m)^\s*[%!]\s*pip\s+install", "installs packages beyond the first cell (declare them as a pip extra of the kit)"),
    (r"subprocess\.(check_call|run|call)\(\s*\[\s*sys\.executable\s*,\s*['\"]-m['\"]\s*,\s*['\"]pip",
     "installs packages with subprocess (declare them as a pip extra of the kit)"),
]


def _current() -> str:
    """The kit version as it is now (the release step may have just changed it)."""
    return _version.__version__


def install_line(version: str | None = None, extras: str = "") -> str:
    """The first code cell of every notebook for a kit version.

    The archive form needs only pip (no git), and writing name[extras]@url without spaces keeps
    the requirement one word, which Notebook Tutor recognises and skips.
    """
    version = version or _current()
    name = f"ai-learning-kit[{extras}]" if extras else "ai-learning-kit"
    return f'%pip install -q "{name}@{REPO}/archive/refs/tags/v{version}.zip"'


def _code_cells(nb: dict) -> list:
    return [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def check_notebook(path, mod=None, version: str | None = None) -> list:
    """Check one notebook against the contract; returns a list of (ok, message)."""
    version = version or _current()
    from ..core import registry
    out = []
    path = Path(path)
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
        out.append((nb.get("nbformat") == 4, "is an nbformat 4 notebook"))
    except Exception as e:
        return [(False, f"cannot be read as a notebook: {e}")]
    meta = nb.get("metadata", {}).get("ai_learning_kit")
    if not isinstance(meta, dict):
        return out + [(False, 'has no metadata["ai_learning_kit"] (topic, title, role, kit_version, install)')]
    names = ("topic", "title", "role", "kit_version", "install")
    missing = [k for k in names if not meta.get(k)]
    out.append((not missing, f"metadata records {', '.join(names[:-1])} and {names[-1]}"
                + (f" (missing: {', '.join(missing)})" if missing else "")))
    out.append((meta.get("role") in ROLES, f"role {meta.get('role')!r} is one of {', '.join(ROLES)}"))
    if meta.get("level") is not None:
        out.append((meta.get("level") in LEVELS, f"level {meta.get('level')!r} is one of {', '.join(LEVELS)}"))
    if mod is None:
        try:
            mod = registry.topic(meta.get("topic", ""))
        except Exception as e:
            return out + [(False, f"metadata names a topic the kit does not have: {e}")]
    out.append((meta.get("topic") == mod.TOPIC, f"metadata names the topic {mod.TOPIC}"))
    out.append((meta.get("kit_version") == version, f"stamped with kit version {version} (has {meta.get('kit_version')})"))
    expected = install_line(version, mod.EXTRAS)
    code = _code_cells(nb)
    out.append((bool(code) and _src(code[0]).strip() == expected, "first code cell is the install line for this version"))
    out.append((meta.get("install") == expected.lstrip("%"), "metadata install line matches the first cell"))
    pattern = rf'kit\.start\(\s*["\']{re.escape(mod.TOPIC)}["\']'
    call = f'kit.start("{mod.TOPIC}"...)'
    starts = [i for i, c in enumerate(code[1:4], start=1) if re.search(pattern, _src(c))]
    first_use = next((i for i, c in enumerate(code[1:], start=1)
                      if re.search(r"\bkit\.(?!__version__)\w+|\bT\.\w+", _src(c))), None)
    out.append((bool(starts) and first_use is not None and starts[0] <= first_use,
                f'the topic is started with {call} within the first three code cells, before any other use '
                f'of the kit'))
    if mod.TOOLS:
        start = _src(code[starts[0]]) if starts else ""
        out.append(("kit.tools()" in start, "the same cell gets the helpers with T = kit.tools()"))
    body = "\n".join(_src(c) for c in code[1:])
    for rx, why in FORBIDDEN:
        hits = [i for i, c in enumerate(code[1:], start=2) if re.search(rx, _src(c))]
        out.append((not hits, f"no cell {why.split(' (')[0]}" if not hits else f"code cell(s) {hits} {why}"))
    if mod.TOOLS:
        helpers = registry.tools(mod)
        used = sorted(set(re.findall(r"\bT\.([A-Za-z_]\w*)", body)))
        absent = [n for n in used if not hasattr(helpers, n)]
        out.append((not absent, f"every T.<name> used exists in the helpers ({len(used)} used)"
                    + (f"; missing: {', '.join(absent)}" if absent else "")))
    from .validate import COURSE_CODE, COURSE_SESSION
    every = [(i, _src(c)) for i, c in enumerate(nb.get("cells", []), start=1)]
    named = [i for i, t in every if COURSE_CODE.search(t) or COURSE_SESSION.search(t)]
    out.append((not named, "no cell names a study unit or a numbered course session" if not named else
                f"cell(s) {named} name a study unit or a numbered course session"))
    drive = [i for i, t in every if re.search(r"(drive|docs)\.google\.com", t)]
    out.append((not drive, "no cell links to Google Drive" if not drive else f"cell(s) {drive} link to Google Drive"))
    errors = [i for i, c in enumerate(code, start=1)
              if any(o.get("output_type") == "error" for o in c.get("outputs", []))
              and "raises-exception" not in c.get("metadata", {}).get("tags", [])]
    out.append((not errors, "saved outputs contain no errors" if not errors else f"code cell(s) {errors} saved with errors"))
    return out


EXERCISE = re.compile(r"#\s*(TODO|YOUR CODE HERE|(write|add|put) your (own )?(code|answer|solution) here)\b|^\s*___|raise NotImplementedError", re.M | re.I)
PLAN = re.compile(r"\*\*From question to code\b", re.I)


def plan_notes(path) -> list:
    """Advice for a practice notebook: the exercises whose statement has no plan from question to code
    (docs/NOTEBOOK_CONTRACT.md, section 6). An exercise is a code cell with a TODO, YOUR CODE HERE, Write your code here, ___ or
    raise NotImplementedError; its plan is in a markdown cell above it, within its section (back to the heading)."""
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    if (nb.get("metadata", {}).get("ai_learning_kit") or {}).get("role") != "practice":
        return []
    cells, missing = nb.get("cells", []), []
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code" or not EXERCISE.search(_src(c)):
            continue
        found = False
        for x in reversed(cells[:i]):
            if x.get("cell_type") != "markdown":
                continue
            if PLAN.search(_src(x)):
                found = True
                break
            if re.search(r"^#{1,3}\s", _src(x), re.M):         # the start of the exercise's section
                break
        if not found:
            missing.append(i + 1)
    return [f"exercise cell(s) {missing} have no plan from question to code (see the notebook contract, section 6)"] \
        if missing else []


def stamp(path, version: str, mod=None) -> bool:
    """Rewrite a notebook's install line and metadata for a kit version; returns True if it changed."""
    from ..core import registry
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    nb = json.loads(text)
    meta = nb.setdefault("metadata", {}).setdefault("ai_learning_kit", {})
    mod = mod or registry.topic(meta["topic"])
    line = install_line(version, mod.EXTRAS)
    code = _code_cells(nb)
    if not code or not re.search(r"pip\s+install", _src(code[0])) or "ai-learning-kit" not in _src(code[0]):
        raise ValueError(f"{path}: the first code cell is not the kit's install line")
    code[0]["source"] = line
    meta.update(kit_version=version, install=line.lstrip("%"))
    new = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    if new != text:
        path.write_text(new, encoding="utf-8")
        return True
    return False


def main(argv=None) -> int:
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 2 and argv[0] == "check":
        failures = 0
        for f in argv[1:]:
            print(f)
            for ok, msg in check_notebook(Path(f)):
                print(f"  {'OK  ' if ok else 'FAIL'} {msg}")
                failures += not ok
            for msg in plan_notes(Path(f)):
                print(f"  NOTE {msg}")
        return 1 if failures else 0
    if len(argv) >= 3 and argv[0] == "stamp" and re.fullmatch(r"\d+\.\d+\.\d+", argv[1]):
        for f in argv[2:]:
            print(f"{f}: {'stamped' if stamp(Path(f), argv[1]) else 'already stamped'} for {argv[1]}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
