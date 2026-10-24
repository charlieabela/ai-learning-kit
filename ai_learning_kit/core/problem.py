"""A problem: a frozen dataset, its sources, and the helper modules that do the heavy lifting.

A problem is independent of any study unit. The same problem can serve several
units and sessions, each of which decides which steps students do themselves and
which steps the kit does for them. See docs/AUTHORING_A_PROBLEM.md.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from importlib import import_module, resources
from pathlib import Path

import pandas as pd


@dataclass
class Problem:
    id: str                                   # folder name under ai_learning_kit/problems, e.g. "brightwater_q3"
    title: str                                # one line, shown to students
    question: str                             # the question the problem asks, in one sentence
    summary: str                              # a short paragraph: setting, sources, why it is interesting
    domain: str                               # e.g. "finance / compliance"
    sources: dict                             # short name -> file name inside data/
    descriptions: dict                        # short name -> one-line description
    text_sources: set = field(default_factory=set)     # sources that peek() can print as text
    peek_hints: dict = field(default_factory=dict)     # binary source -> "use kit.extract.xxx()"
    modules: dict = field(default_factory=dict)        # namespace -> module path, e.g. {"extract": "...extract"}
    workspace_name: str | None = None         # folder the case files are copied into (defaults to id)
    tags: list = field(default_factory=list)  # free-text tags: formats, skills, level
    fictional: bool = True                    # all companies, people and documents are invented

    # ------------------------------------------------------------------ files
    @property
    def data_dir(self) -> Path:
        return Path(str(resources.files("ai_learning_kit") / "problems" / self.id / "data"))

    def path(self, name: str) -> Path:
        """The path of a source file inside the installed kit.

        `name` is one of the short names listed by `sources()`. Students normally
        use the copies made by `workspace()` instead.
        """
        if name not in self.sources:
            raise KeyError(f"Unknown source {name!r}. Choose one of: {', '.join(self.sources)}")
        return self.data_dir / self.sources[name]

    def sources_table(self) -> pd.DataFrame:
        """List the problem's sources: short name, file name, size and description."""
        rows = []
        for name, fname in self.sources.items():
            p = self.data_dir / fname
            rows.append({"source": name, "file": fname, "size_kB": round(p.stat().st_size / 1024, 1),
                         "description": self.descriptions.get(name, "")})
        return pd.DataFrame(rows)

    def workspace(self, folder=None, overwrite: bool = False) -> Path:
        """Copy the case files into a working folder and return its path.

        The default folder is named after the problem and sits in the current
        folder (the work folder chosen by kit.start). Existing files are kept
        unless overwrite=True, so files you create there are never deleted.
        """
        folder = Path(folder) if folder is not None else Path(self.workspace_name or self.id)
        folder.mkdir(parents=True, exist_ok=True)
        for fname in self.sources.values():
            target = folder / fname
            if overwrite or not target.exists():
                shutil.copy2(self.data_dir / fname, target)
        print(f"Case files are in: {folder.resolve()}/")
        for name, fname in self.sources.items():
            print(f"  {name:<13} {fname}")
        return folder

    def peek(self, name: str, lines: int = 12) -> None:
        """Print the first lines of a source exactly as stored, without interpreting it.

        Text-based sources are shown as raw characters. Binary documents cannot be
        read as plain text; for those the kit shows the first bytes and says which
        extraction function to use.
        """
        p = self.path(name)
        if name not in self.text_sources:
            with open(p, "rb") as f:
                start = f.read(24)
            print(f"{p.name} is a binary file. Its first bytes are {start!r}.")
            if start.startswith(b"PK"):
                print("PK is the signature of a ZIP archive: Word and Excel files are ZIP folders of XML files.")
            elif start.startswith(b"%PDF"):
                print("%PDF is the signature of a PDF file: text, fonts and images laid out on pages.")
            hint = self.peek_hints.get(name)
            print("It cannot be read as plain text." + (f" Use {hint}." if hint else ""))
            return
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= lines:
                    print(f"... ({p.name} continues)")
                    break
                print(line.rstrip("\n"))

    # ------------------------------------------------------------------ helper modules
    def module(self, namespace: str):
        """The helper module registered under a namespace, e.g. problem.module("extract")."""
        if namespace not in self.modules:
            raise AttributeError(f"Problem {self.id!r} has no {namespace!r} helpers. "
                                 f"Available: {', '.join(self.modules) or 'none'}")
        return import_module(self.modules[namespace])

    def describe(self) -> dict:
        """A JSON-able description, used by catalog() and by notebook tutors."""
        return {"id": self.id, "title": self.title, "question": self.question, "summary": self.summary,
                "domain": self.domain, "sources": dict(self.sources), "descriptions": dict(self.descriptions),
                "helpers": sorted(self.modules), "tags": list(self.tags), "fictional": self.fictional}
