"""The topic reference: one page per topic, generated from the kit itself, and the check that it is current.

    python -m ai_learning_kit.dev.reference write    # (re)write docs/reference/
    python -m ai_learning_kit.dev.reference check    # exit 1 when a page differs from what the kit generates

Each page says what a topic covers, its level, the topics it builds on, its data and its pip extra, how a notebook
starts it, and its three layers: what learners do themselves (in focus), what the kit does for them and they may
look inside (on request), what the kit does for them (support) and what is left out. Then every helper of
T = kit.tools(), with its signature and the first paragraph of its docstring, grouped by the layer that names it.
Nothing on the pages is written by hand, so the pages cannot drift from the kit: change a docstring or a stage and
run `write`; the test suite runs `check`.
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "reference"

LAYERS = [("focus", "In focus: learners do these themselves"),
          ("on_request", "On request: the kit does these, and learners may look inside"),
          ("black_box", "Support: the kit does these for the learner"),
          ("out_of_scope", "Left out of this topic")]


def _first_paragraph(text) -> str:
    return " ".join((inspect.cleandoc(text or "").split("\n\n")[0]).split())


def _kit_path(m) -> str:
    """A path default written the same on every machine: relative to the installed package, else its last part."""
    path = Path(m.group(1))
    try:
        return "<kit>/" + path.resolve().relative_to(ROOT / "ai_learning_kit").as_posix()
    except ValueError:
        return f"<{path.name}>"


def _signature(obj) -> str:
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(...)"
    sig = re.sub(r" at 0x[0-9A-Fa-f]+", "", sig)                 # memory addresses (capitals on Windows)
    sig = re.sub(r"(?:Posix|Windows)?Path\('([^']*)'\)", _kit_path, sig)   # a path default: where the kit is installed
    sig = re.sub(r": '([^']*)'", r": \1", sig)                 # annotations kept as text by `from __future__`
    return re.sub(r"-> '([^']*)'", r"-> \1", sig)


def _helpers(mod, tools) -> tuple[list, list, list]:
    """(own functions and classes in source order, names re-exported from another topic, constants)."""
    own, reexported, constants = [], [], []
    for name, value in vars(tools).items():
        if name.startswith("_") or inspect.ismodule(value):
            continue
        if inspect.isfunction(value) or inspect.isclass(value):
            origin = getattr(value, "__module__", "")
            if origin == tools.__name__:
                try:
                    line = inspect.getsourcelines(value)[1]
                except (OSError, TypeError):
                    line = 10 ** 6
                own.append((line, name, value))
            elif origin.startswith("ai_learning_kit.topics."):
                reexported.append((name, origin.split(".")[2].replace("_", "-")))
        elif name.isupper():
            kind = "Path" if isinstance(value, Path) else type(value).__name__    # PosixPath or WindowsPath by machine
            constants.append((name, kind))
    return [(n, v) for _, n, v in sorted(own, key=lambda t: (t[0], t[1]))], sorted(reexported), sorted(constants)


def _named_in(how: str | None) -> set:
    return set(re.findall(r"\bT\.([A-Za-z_]\w*)", how or ""))


def topic_page(name: str) -> str:
    from .. import datasets
    from ..core import levels, registry
    mod = registry.topic(name)
    tools, missing = None, None
    try:
        tools = registry.tools(mod)
    except ImportError as e:                      # a library of the topic's extra is not installed here
        missing = e.name or str(e)
    stages = dict(getattr(mod, "STAGES", {}) or {})
    lines = [f"# {mod.TITLE}", "",
             f"`{mod.TOPIC}` · level **{mod.LEVEL}** · "
             + (f"builds on {', '.join(f'[`{b}`]({b}.md)' for b in mod.BUILDS_ON)}" if mod.BUILDS_ON else "builds on no other topic")
             + f" · pip extra `{mod.EXTRAS or 'none'}`", "",
             _first_paragraph(mod.__doc__), "",
             "*Generated from the kit by `python -m ai_learning_kit.dev.reference write`: do not edit by hand.*", "",
             "## Starting the topic", "", "```python", "import ai_learning_kit as kit",
             f'kit.start("{mod.TOPIC}"' + (', student="Your Name")' if mod.TASKS else ")")]
    if mod.TOOLS:
        lines.append("T = kit.tools()")
    lines += ["```", ""]
    level = levels.LEVELS[mod.LEVEL]
    lines += [f"Written for the **{level['label'].lower()}** level: {level['learner']}", ""]
    if mod.PROBLEM:
        p = registry.problem(mod.PROBLEM)
        lines += [f"**Case problem:** `{p.id}`, {p.title}. {p.question}", ""]
    if mod.DATASETS:
        lines += ["**Data downloaded from its owners** (checked against a checksum; `kit.datasets.status()` shows where it "
                  "is kept):", ""]
        for d in mod.DATASETS:
            s = datasets.source(d)
            lines.append(f"- `{d}`: {s.title}, from {s.owner}. {s.licence}")
        lines.append("")

    lines += ["## The three layers", ""]
    for vis, heading in LAYERS:
        items = [(k, st) for k, st in stages.items() if st["visibility"] == vis]
        if not items:
            continue
        lines += [f"### {heading}", ""]
        for _, st in items:
            extra = []
            if st.get("how"):
                extra.append(f"with `{st['how']}`")
            if st.get("taught_in"):
                extra.append(f"taught in {st['taught_in']}")
            lines.append(f"- {st['label']}" + (f" ({'; '.join(extra)})" if extra else ""))
        lines.append("")

    if mod.TASKS:
        lines += ["## Graded tasks", "", "`kit.task(\"<id>\")` explains a task; `kit.answer(...)` records and checks it.",
                  "", "| Task | Title | Question |", "|---|---|---|"]
        for tid, t in mod.TASKS.items():
            lines.append(f"| {tid} | {t['title']} | {' '.join(t['question'].split())} |")
        lines.append("")

    if missing:
        lines += ["## Helpers: `T = kit.tools()`", "",
                  f"The helpers cannot be listed on this computer, because `{missing}` is not installed. They come "
                  f"with the `{mod.EXTRAS or 'advanced'}` extra; the published reference lists them all.", ""]
    if tools is not None:
        own, reexported, constants = _helpers(mod, tools)
        layer_of = {}
        for vis, _ in LAYERS:
            for st in stages.values():
                if st["visibility"] == vis:
                    for n in _named_in(st.get("how")):
                        layer_of.setdefault(n, vis)
        lines += ["## Helpers: `T = kit.tools()`", "", _first_paragraph(tools.__doc__), "",
                  "`kit.source(T.<name>)` prints a helper's code, `help(T.<name>)` its full description.", ""]
        groups = [(vis, [(n, v) for n, v in own if layer_of.get(n) == vis]) for vis, _ in LAYERS[:3]]
        groups.append(("other", [(n, v) for n, v in own if layer_of.get(n) not in ("focus", "on_request", "black_box")]))
        for vis, items in groups:
            if not items:
                continue
            title = {"focus": "Named in the focus", "on_request": "On request (look inside)",
                     "black_box": "Support", "other": "Not yet placed in a layer"}[vis]
            lines += [f"### {title}", ""]
            if vis == "other":
                lines += ["No stage of the topic names these helpers, so a tutor cannot tell whether learners are meant "
                          "to write this kind of code, look inside it, or only use it. Name each in a stage's `how` to "
                          "place it.", ""]
            for n, v in items:
                kind = "class " if inspect.isclass(v) else ""
                doc = _first_paragraph(v.__doc__) or "(no description)"
                lines.append(f"- **`{kind}T.{n}{_signature(v)}`**: {doc}")
            lines.append("")
        if reexported:
            lines += ["### From other topics", "", "Supplied ready-made here; each is taught in its own topic.", ""]
            for n, origin in reexported:
                lines.append(f"- `T.{n}`, from [`{origin}`]({origin}.md)")
            lines.append("")
        if constants:
            lines += ["### Constants", "", ", ".join(f"`T.{n}` ({t})" for n, t in constants), ""]
    return "\n".join(lines).rstrip() + "\n"


def index_page() -> str:
    import ai_learning_kit as kit
    from .. import topics
    from ..core import levels, registry
    lines = ["# ai-learning-kit reference", "",
             "*Generated from the kit by `python -m ai_learning_kit.dev.reference write`: do not edit by hand.*", "",
             "## Topics", "", "| Topic | Level | Builds on | Extra | What it covers |", "|---|---|---|---|---|"]
    for name in topics.names():
        mod = registry.topic(name)
        lines.append(f"| [`{name}`]({name}.md) | {mod.LEVEL} | {', '.join(mod.BUILDS_ON) or '—'} | "
                     f"{mod.EXTRAS or '—'} | {_first_paragraph(mod.__doc__)} |")
    lines += ["", "## Levels", ""]
    for key, entry in levels.LEVELS.items():
        lines.append(f"- **{key}**: {entry['learner']}")
    lines += ["", "## The kit's own functions", "",
              "Available in every topic, as `kit.<name>` after `import ai_learning_kit as kit`.", ""]
    for name in kit.__all__:
        value = getattr(kit, name, None)
        if inspect.isfunction(value):
            lines.append(f"- **`kit.{name}{_signature(value)}`**: {_first_paragraph(value.__doc__) or '(no description)'}")
    return "\n".join(lines).rstrip() + "\n"


def page(name: str | None = None) -> str:
    """One page of the reference as Markdown, generated from the kit that is installed: the topic called name, or
    the index when name is None or "README". For tutors and other front ends; links between pages are "<topic>.md"."""
    name = (name or "README").removesuffix(".md")
    return index_page() if name in ("README", "index") else topic_page(name)


def pages() -> dict:
    from .. import topics
    out = {"README.md": index_page()}
    for name in topics.names():
        out[f"{name}.md"] = topic_page(name)
    return out


def stale(folder: Path = OUT) -> list:
    """The pages that differ from what the kit generates, are missing, or no longer belong."""
    expected = pages()
    found = {p.name for p in folder.glob("*.md")} if folder.is_dir() else set()
    bad = [n for n, text in expected.items() if not (folder / n).is_file() or (folder / n).read_text(encoding="utf-8") != text]
    return sorted(bad + [n for n in found - set(expected)])


def differences(folder: Path = OUT, limit: int = 40) -> str:
    """The lines that differ between the pages in folder and what the kit generates, as a unified diff."""
    import difflib
    out = []
    for name in stale(folder):
        f = folder / name
        old = f.read_text(encoding="utf-8").splitlines() if f.is_file() else []
        new = pages().get(name, "").splitlines()
        out += list(difflib.unified_diff(old, new, f"committed/{name}", f"generated/{name}", n=0, lineterm=""))[:limit]
    return "\n".join(out)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["write"]:
        OUT.mkdir(parents=True, exist_ok=True)
        expected = pages()
        for p in OUT.glob("*.md"):
            if p.name not in expected:
                p.unlink()
        for n, text in expected.items():
            (OUT / n).write_text(text, encoding="utf-8")
        print(f"wrote {len(expected)} pages in {OUT.relative_to(ROOT)}/")
        return 0
    if argv == ["check"]:
        bad = stale()
        if bad:
            print("The reference is out of date for: " + ", ".join(bad)
                  + ". Run python -m ai_learning_kit.dev.reference write and commit the result.")
            print(differences())
            return 1
        print("The reference matches the kit.")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
