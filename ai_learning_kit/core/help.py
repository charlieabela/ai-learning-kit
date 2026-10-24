"""Look inside the black box: explain() prints what a kit function does, source() shows its code."""
from __future__ import annotations

import inspect
import textwrap


def explain(obj) -> None:
    """Print the plain-language description of a kit function or module."""
    name = getattr(obj, "__qualname__", getattr(obj, "__name__", repr(obj)))
    doc = inspect.getdoc(obj) or "No description available."
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        sig = ""
    print(f"{name}{sig}\n")
    print(textwrap.indent(doc, "  "))


def source(obj) -> None:
    """Print the Python source code of a kit function, so you can see how it works."""
    try:
        print(inspect.getsource(obj))
    except (TypeError, OSError):
        print("The source code of this object is not available.")
