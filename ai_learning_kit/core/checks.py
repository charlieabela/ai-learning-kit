"""Reusable building blocks for a session's formative checks.

Two kinds of check are needed by almost every session:

- numeric_check: a single number the student must get right. The feedback says
  what was recorded, recognises common wrong values and names their likely cause,
  gives evidence from the data without giving the answer away, and gives a line
  of code to verify with. After several attempts the right value and its
  explanation are shown.
- written: a checklist for a written answer. The feedback lists the points a good
  answer covers, marks each as found or not found, and gives a hint for each
  missing point. A model answer is shown after several attempts. The tests look
  for key words, so they can miss a point made in other words: the feedback says
  "not found", never "wrong", and the lecturer reads every written answer.

Session-specific checks (a particular DataFrame shape, a dictionary of formats)
are written in the session module, returning Feedback objects directly.
"""
from __future__ import annotations

import re
from typing import Callable

from .feedback import Feedback


def as_int(value):
    """The value as an int, or None when it is not a whole number (bool counts as not a number)."""
    try:
        if isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- numbers
def numeric_check(ref: Callable[[], dict], key: str, how_to_check: str, explanation: str,
                  wrong: dict | None = None, evidence: str | None = None, right: str | None = None):
    """Build a check for a single count.

    ref          : a function returning the session's reference values (cached)
    key          : the reference value to compare with
    how_to_check : a line of code the student can run to verify
    explanation  : why the reference value is right (shown with the suggested answer)
    wrong        : {reference key or int: explanation} for common wrong values
    evidence     : facts in the data that help, without giving the number away
    right        : what a correct value means (shown with the tick)
    """
    def check(v):
        expected = ref()[key]
        got = as_int(v)
        if got is None:
            return Feedback(False, f"Record a whole number. You recorded {v!r}, a {type(v).__name__}.",
                            ["Check that the value you pass is the count itself, for example len(rows), "
                             "not a DataFrame or a piece of text.", f"To check: {how_to_check}"],
                            suggestion=f"{expected}. {explanation}")
        if got == expected:
            return Feedback(True, f"{got} is right." + (f" {right}" if right else ""))
        details = [f"You recorded {got}."]
        for wrong_key, why in (wrong or {}).items():
            value = wrong_key if isinstance(wrong_key, int) else ref()[wrong_key]
            if got == value and value != expected:
                details.append(f"Likely cause: {why}")
                break
        else:
            details.append(f"Your count is {'higher' if got > expected else 'lower'} than the right count, "
                           "and does not match any of the usual slips. Print the rows you counted and look at them.")
        if evidence:
            details.append(f"Evidence in the data: {evidence}")
        details.append(f"To check: {how_to_check}")
        return Feedback(False, "Not yet right.", details, suggestion=f"{expected}. {explanation}")
    return check


# --------------------------------------------------------------------------- written answers
def any_of(*patterns):
    """A test that passes when the text matches any of the regular expressions (case-insensitive)."""
    compiled = [re.compile(p, re.I | re.S) for p in patterns]
    return lambda text: any(c.search(text) for c in compiled)


def all_of(*tests):
    return lambda text: all(t(text) for t in tests)


def at_least(n, *tests):
    return lambda text: sum(bool(t(text)) for t in tests) >= n


PLACEHOLDERS = ("write your", "which format would you", "your answer here", "todo")


def written(points, model: str, what: str = "reasoning"):
    """Build a checklist check for a written answer.

    points : list of (what a good answer covers, test, hint if it seems to be missing)
    model  : a model answer, shown after several attempts if points are still missing
    what   : the noun used in the feedback ("reasoning", "findings")
    """
    verb = "cover" if what.endswith("s") else "covers"

    def check(text):
        text = " ".join(str(text or "").split())
        if not text or text.lower().startswith(PLACEHOLDERS):
            return Feedback(None, f"No {what} yet: replace the placeholder text with your own answer.",
                            ["A good answer covers: " + "; ".join(p[0][0].lower() + p[0][1:] for p in points) + "."],
                            suggestion=model, open_points=True)
        found = [(label, hint) for label, test, hint in points if test(text)]
        missing = [(label, hint) for label, test, hint in points if not test(text)]
        details = [f"✓ {label}" for label, _ in found]
        details += [f"○ Not found: {label[0].lower() + label[1:]} {hint}" for label, hint in missing]
        if missing:
            details.append("(The kit looks for key words, so a point you made in other words may show as not found. "
                           "Your lecturer reads every written answer.)")
            summary = f"Your {what} {verb} {len(found)} of {len(points)} points that a good answer includes:"
        else:
            summary = f"Your {what} {verb} all {len(points)} points that a good answer includes. Your lecturer reads it."
        return Feedback(None, summary, details, suggestion=model, open_points=bool(missing))
    check.points = points      # kept for validators and for LLM review prompts
    check.model = model
    return check


def saved_only(v):
    """For tasks with no automatic check: the lecturer reads the answer."""
    return Feedback(None, "Saved. Your lecturer reads this answer.")
