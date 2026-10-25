"""Prepare a release: one command keeps the version, the notebooks, the links and the docs in step.

    python -m ai_learning_kit.dev.release 0.5.0

It
1. sets ai_learning_kit/_version.py to the new version;
2. stamps every notebook in notebooks/ with that version (install line and metadata);
3. rebuilds notebooks/README.md: every notebook with its Open-in-Colab link for that version;
4. updates the install examples in README.md and docs/ to that version;
5. runs the validator, and prints what is left to do by hand.

Run it on a branch, commit the result, open a pull request, merge it, and only then create
the tag v<version> on main (tags are protected and cannot be moved afterwards). Then update
Notebook Tutor to bundle the new version.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from .. import units
from ..core import registry
from .notebooks import NOTEBOOK_ROOT, colab_url, stamp

ROOT = NOTEBOOK_ROOT.parent
VERSION_FILE = ROOT / "ai_learning_kit" / "_version.py"
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
ROLE_ORDER = {"prepare": 0, "explore": 1, "practice": 2, "session": 3, "solutions": 4}
ROLE_LABEL = {"prepare": "Prepare", "explore": "Explore", "practice": "Practice", "session": "Session",
              "solutions": "Solutions"}


def set_version(version: str) -> None:
    VERSION_FILE.write_text(f'__version__ = "{version}"\n', encoding="utf-8")


def notebooks() -> list:
    return sorted(p for p in NOTEBOOK_ROOT.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts)


def update_docs(version: str) -> list:
    changed = []
    patterns = [
        (re.compile(r"(ai-learning-kit(?:\[[\w,]+\])?\s*@\s*https://github\.com/charlieabela/ai-learning-kit/archive/refs/tags/v)"
                    r"\d+\.\d+\.\d+(\.zip)"), rf"\g<1>{version}\g<2>"),
        (re.compile(r"(git\+https://github\.com/charlieabela/ai-learning-kit@v)\d+\.\d+\.\d+"), rf"\g<1>{version}"),
        (re.compile(r"(notebooks install a tag \(`@v)\d+\.\d+\.\d+"), rf"\g<1>{version}"),
    ]
    for doc in DOCS:
        if not doc.is_file():
            continue
        text = doc.read_text(encoding="utf-8")
        new = text
        for rx, repl in patterns:
            new = rx.sub(repl, new)
        if new != text:
            doc.write_text(new, encoding="utf-8")
            changed.append(doc)
    return changed


def write_index(version: str) -> Path:
    rows = {}
    for p in notebooks():
        meta = json.loads(p.read_text(encoding="utf-8")).get("metadata", {}).get("ai_learning_kit", {})
        if meta.get("role") == "solutions":
            continue                         # lecturer-only material is never listed for students
        rows.setdefault((meta.get("unit", "?"), meta.get("session", "?")), []).append((meta, p))
    lines = [
        "# Notebooks",
        "",
        f"Every notebook below is for ai-learning-kit **v{version}**. Each runs in Notebook Tutor (open it from the "
        "tutor) and in Colab (use the link). The links open the copy published with this release, so every student "
        "gets the same notebook; in Colab, choose *File › Save a copy in Drive* to keep your own edits.",
        "",
        "This file is written by `python -m ai_learning_kit.dev.release`; do not edit it by hand.",
        "",
    ]
    for (unit, session) in sorted(rows):
        try:
            mod = registry.session(unit, session)
            title = mod.TITLE
        except Exception:
            title = ""
        lines += [f"## {unit} {session} · {title}".rstrip(" ·"), "", "| Part | Notebook | Open |", "|---|---|---|"]
        for meta, p in sorted(rows[(unit, session)], key=lambda mp: (ROLE_ORDER.get(mp[0].get("role"), 9), mp[1].name)):
            rel = p.relative_to(NOTEBOOK_ROOT).as_posix()
            badge = f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url(rel, version)})"
            lines.append(f"| {ROLE_LABEL.get(meta.get('role'), meta.get('role', ''))} | [{p.name}]({rel}) | {badge} |")
        lines.append("")
    index = NOTEBOOK_ROOT / "README.md"
    index.write_text("\n".join(lines), encoding="utf-8")
    return index


def check_registered_notebooks() -> list:
    """Every notebook named by a session exists, and every notebook in notebooks/ is named by a session."""
    problems_found, named = [], set()
    for code in units.REGISTRY:
        for scode in getattr(units.unit(code), "SESSIONS", {}):
            mod = registry.session(code, scode)
            for rel in mod.NOTEBOOKS.values():
                named.add(rel)
                if not (NOTEBOOK_ROOT / rel).is_file():
                    problems_found.append(f"{code} {scode} names {rel}, which does not exist")
    for p in notebooks():
        rel = p.relative_to(NOTEBOOK_ROOT).as_posix()
        role = json.loads(p.read_text(encoding="utf-8")).get("metadata", {}).get("ai_learning_kit", {}).get("role")
        if rel not in named and role != "solutions":
            problems_found.append(f"{rel} is not listed in its session's NOTEBOOKS")
    return problems_found


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or not re.fullmatch(r"\d+\.\d+\.\d+", argv[0]):
        print(__doc__)
        return 2
    version = argv[0]
    set_version(version)
    stamped = [p for p in notebooks() if stamp(p, version)]
    docs = update_docs(version)
    index = write_index(version)
    print(f"version: {version}")
    print(f"notebooks stamped: {len(stamped)} of {len(notebooks())}")
    print(f"docs updated: {', '.join(d.relative_to(ROOT).as_posix() for d in docs) or 'none'}")
    print(f"index: {index.relative_to(ROOT).as_posix()}")
    missing = check_registered_notebooks()
    for m in missing:
        print(f"  FAIL {m}")
    from importlib import reload
    from .. import _version
    reload(_version)
    from .validate import validate_all
    failures = validate_all() or bool(missing)
    if failures:
        return 1
    print(f"\nNext: add a CHANGELOG entry, commit, open a pull request, merge, then create the tag v{version} on main "
          f"and update Notebook Tutor to bundle it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
