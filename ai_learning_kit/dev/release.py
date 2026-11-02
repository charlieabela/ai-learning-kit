"""Prepare a release: one command keeps the version and the docs in step.

    python -m ai_learning_kit.dev.release 0.5.0

It
1. sets ai_learning_kit/_version.py to the new version;
2. updates the install examples in README.md and docs/ to that version;
3. runs the validator, and prints what is left to do by hand.

Run it on a branch, commit the result, open a pull request, merge it, and only then create
the tag v<version> on main (tags are protected and cannot be moved afterwards). Then update
Notebook Tutor to bundle the new version. Notebooks are not stored in the kit: stamp the ones
you distribute with  python -m ai_learning_kit.dev.notebooks stamp <version> <files>  and check
them with  python -m ai_learning_kit.dev.notebooks check <files>.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION_FILE = ROOT / "ai_learning_kit" / "_version.py"
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


def set_version(version: str) -> None:
    VERSION_FILE.write_text(f'__version__ = "{version}"\n', encoding="utf-8")


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


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or not re.fullmatch(r"\d+\.\d+\.\d+", argv[0]):
        print(__doc__)
        return 2
    version = argv[0]
    set_version(version)
    docs = update_docs(version)
    print(f"version: {version}")
    print(f"docs updated: {', '.join(d.relative_to(ROOT).as_posix() for d in docs) or 'none'}")
    from importlib import reload
    from .. import _version
    reload(_version)
    from .validate import validate_all
    if validate_all():
        return 1
    print(f"\nNext: add a CHANGELOG entry, commit, open a pull request, merge, then create the tag v{version} on main, "
          f"update Notebook Tutor to bundle it, and stamp the notebooks you distribute "
          f"(python -m ai_learning_kit.dev.notebooks stamp {version} <files>).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
