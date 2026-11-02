"""Heavy lifting that sessions may hide from students: turn documents into data.

Every public function has a docstring in plain language, because students can
call kit.explain(kit.extract.<function>) and kit.source(...) to look inside.
"""
from __future__ import annotations

from .case import path


def example_extract():
    """<What this returns, in one sentence, and how it found it in the source.>"""
    raise NotImplementedError
