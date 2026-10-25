"""The notebook contract: what every notebook in notebooks/ must look like, and the check for it.

A kit notebook runs unchanged in Notebook Tutor and in Colab, and (best effort) in Jupyter.
That works because everything that depends on where it runs lives in the kit, not in the
notebook. The contract (docs/NOTEBOOK_CONTRACT.md explains each rule):

1. metadata["ai_learning_kit"] records unit, session, title, role, kit_version and install,
   so that a tutor recognises the notebook.
2. The first code cell is exactly the install line for this kit version, with the session's
   pip extra (install_line()). Notebook Tutor skips it, because it ships the kit.
3. The session is started with kit.start("<UNIT>", "<SESSION>"...) in one of the first three code
   cells, before any other use of the kit, and T = kit.tools() follows it when the session has
   helpers. New notebooks put both in the second code cell (see the canonical opening below).
4. No code that depends on the environment: no Google Drive or Colab calls, no uploads, no
   ZIP handling, no sys.path changes, no further pip installs, no hard-coded /content paths.
   Data comes from the kit (ai_learning_kit.datasets or the session's helpers).
5. Every T.<name> the notebook uses exists in the session's helper module.
6. Saved outputs contain no errors, except in cells tagged "raises-exception".
7. The notebook belongs to the kit version it says it does (the release step stamps this).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .. import _version

REPO = "https://github.com/charlieabela/ai-learning-kit"
NOTEBOOK_ROOT = Path(__file__).resolve().parents[2] / "notebooks"
ROLES = ("prepare", "explore", "practice", "session", "solutions")   # session: one notebook for a whole session

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


def colab_url(rel_path: str, version: str | None = None) -> str:
    """Open-in-Colab link for a notebook in notebooks/, pinned to a release tag."""
    version = version or _current()
    rel = str(rel_path).replace("\\", "/")
    return f"https://colab.research.google.com/github/charlieabela/ai-learning-kit/blob/v{version}/notebooks/{rel}"


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
        return out + [(False, 'has no metadata["ai_learning_kit"] (unit, session, role, kit_version, install)')]
    missing = [k for k in ("unit", "session", "title", "role", "kit_version", "install") if not meta.get(k)]
    out.append((not missing, "metadata records unit, session, title, role, kit_version and install"
                + (f" (missing: {', '.join(missing)})" if missing else "")))
    out.append((meta.get("role") in ROLES, f"role {meta.get('role')!r} is one of {', '.join(ROLES)}"))
    if mod is None:
        try:
            mod = registry.session(meta.get("unit", ""), meta.get("session", ""))
        except Exception as e:
            return out + [(False, f"metadata names a session the kit does not have: {e}")]
    out.append((meta.get("unit") == mod.UNIT and meta.get("session") == mod.SESSION,
                f"metadata names {mod.UNIT} {mod.SESSION}"))
    out.append((meta.get("kit_version") == version, f"stamped with kit version {version} (has {meta.get('kit_version')})"))
    expected = install_line(version, mod.EXTRAS)
    code = _code_cells(nb)
    out.append((bool(code) and _src(code[0]).strip() == expected, "first code cell is the install line for this version"))
    out.append((meta.get("install") == expected.lstrip("%"), "metadata install line matches the first cell"))
    pattern = rf'kit\.start\(\s*["\']{mod.UNIT}["\']\s*,\s*["\']{mod.SESSION}["\']'
    starts = [i for i, c in enumerate(code[1:4], start=1) if re.search(pattern, _src(c))]
    first_use = next((i for i, c in enumerate(code[1:], start=1)
                      if re.search(r"\bkit\.(?!__version__)\w+|\bT\.\w+", _src(c))), None)
    out.append((bool(starts) and first_use is not None and starts[0] <= first_use,
                f'the session is started with kit.start("{mod.UNIT}", "{mod.SESSION}"...) within the first three '
                f'code cells, before any other use of the kit'))
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
    errors = [i for i, c in enumerate(code, start=1)
              if any(o.get("output_type") == "error" for o in c.get("outputs", []))
              and "raises-exception" not in c.get("metadata", {}).get("tags", [])]
    out.append((not errors, "saved outputs contain no errors" if not errors else f"code cell(s) {errors} saved with errors"))
    return out


def stamp(path, version: str, mod=None) -> bool:
    """Rewrite a notebook's install line and metadata for a kit version; returns True if it changed."""
    from ..core import registry
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    nb = json.loads(text)
    meta = nb.setdefault("metadata", {}).setdefault("ai_learning_kit", {})
    mod = mod or registry.session(meta["unit"], meta["session"])
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
