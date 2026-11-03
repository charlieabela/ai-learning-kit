"""Compatibility shim: `import ari1101kit as kit` keeps working with ai-learning-kit.

Notebooks written for ari1101kit 0.2.x call kit.start("S02", student=...). This
module forwards everything to ai_learning_kit with the unit fixed to ARI1101.
New notebooks should `import ai_learning_kit as kit` and call
kit.start("ARI1101", "S02", student=...).
"""
from __future__ import annotations

import ai_learning_kit as _kit
from ai_learning_kit import *  # noqa: F401,F403
from ai_learning_kit import __all__ as _all, __version__  # noqa: F401

__all__ = list(_all)


def start(session: str, student: str = "", folder=None, drive: bool = True) -> None:
    """ari1101kit-style start: the unit is ARI1101."""
    print("Note: ari1101kit is now part of ai-learning-kit. In new notebooks use "
          "`import ai_learning_kit as kit` and kit.start('ARI1101', 'S02', student=...).")
    return _kit.start("ARI1101", session, student=student, folder=folder, drive=drive)
