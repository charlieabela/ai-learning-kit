"""Reference values: the correct results, computed from the data, that checks compare with.

Keep these pure functions of the frozen data (cache them). Sessions build their
own cached _ref() dictionary from them, so the numbers are never typed by hand.
"""
from __future__ import annotations

from functools import lru_cache

from . import extract  # noqa: F401
from .case import path  # noqa: F401


@lru_cache(maxsize=None)
def example_reference():
    """<The reference value and the rule it follows.>"""
    raise NotImplementedError
