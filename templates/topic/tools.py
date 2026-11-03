"""<Title>: the helpers returned by T = kit.tools() after kit.start("<name>").

Support helpers do one step each, so that the notebook stays on the topic's focus; write the ones
marked ON_REQUEST in the topic's STAGES to be read, because kit.source(T.<name>) prints them.
Released helpers never change; a change is a new function.
"""
from pathlib import Path

ROOT = Path(__file__).parent
SEED = 1


def check(exercise: str, *args, **kwargs) -> None:
    """Print one PASS or FAIL line per fact for a practice exercise, with a hint when it fails."""
    raise NotImplementedError(exercise)
