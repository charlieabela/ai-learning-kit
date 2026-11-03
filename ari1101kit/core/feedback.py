"""Formative feedback: what is right, what is not, the evidence, and how to verify it.

A check returns a Feedback object. Its lines are printed under a one-line verdict.
On the third and later attempts at a task that is still not right, the check may
add a suggested answer with an explanation, so that no student stays stuck.
"""
from __future__ import annotations

from dataclasses import dataclass, field

REVEAL_AFTER = 3  # attempts before a suggested answer is shown


@dataclass
class Feedback:
    ok: bool | None                     # True correct, False needs work, None not auto-checked
    summary: str
    details: list[str] = field(default_factory=list)
    suggestion: str | None = None       # shown only from REVEAL_AFTER attempts onwards

    def render(self, task_id: str, attempt: int | None = None) -> str:
        mark = {True: "✓", False: "✗", None: "•"}[self.ok]
        head = f"{mark} {task_id}"
        if attempt:
            head += f" · attempt {attempt}"
        lines = [f"{head} · {self.summary}"]
        lines += [f"    {d}" for d in self.details]
        if self.ok is False:
            if self.suggestion and attempt and attempt >= REVEAL_AFTER:
                lines.append("    Suggested answer (shown after several attempts):")
                lines += [f"        {s}" for s in self.suggestion.splitlines()]
            elif self.suggestion and attempt:
                left = REVEAL_AFTER - attempt
                lines.append(f"    Correct the points above and run the cell again. "
                             f"A suggested answer appears after {left} more attempt(s).")
            else:
                lines.append("    Correct the points above and try again.")
        return "\n".join(lines)


def as_feedback(result) -> Feedback:
    """Accept either a Feedback or an (ok, message) pair, for older session modules."""
    if isinstance(result, Feedback):
        return result
    ok, msg = result
    return Feedback(ok, msg)
