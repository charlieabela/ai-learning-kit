"""Formative feedback: what is right, what is not, the evidence, and how to verify it.

A check returns a Feedback object. Its lines are printed under a one-line verdict.
On the third and later attempts at a task that is still not right, a suggested
answer with an explanation is added, so that no student stays stuck.
"""
from __future__ import annotations

from dataclasses import dataclass, field

REVEAL_AFTER = 3  # attempts before a suggested answer is shown


@dataclass
class Feedback:
    ok: bool | None                     # True correct, False needs work, None read by the lecturer
    summary: str
    details: list[str] = field(default_factory=list)
    suggestion: str | None = None       # shown only from REVEAL_AFTER attempts onwards
    open_points: bool = False           # a written answer that seems to miss some points

    def needs_work(self) -> bool:
        return self.ok is False or self.open_points

    def render(self, task_id: str, attempt: int | None = None) -> str:
        mark = {True: "✓", False: "✗", None: "•"}[self.ok]
        head = f"{mark} {task_id}"
        if attempt:
            head += f" · attempt {attempt}"
        lines = [f"{head} · {self.summary}"]
        lines += [f"    {d}" for d in self.details]
        if self.needs_work():
            if self.suggestion and attempt and attempt >= REVEAL_AFTER:
                lines.append("    Suggested answer (shown after several attempts; compare it with yours):")
                lines += [f"        {s}" for s in self.suggestion.splitlines()]
            elif self.suggestion and attempt:
                left = REVEAL_AFTER - attempt
                lines.append(f"    Improve the points above and run the cell again. "
                             f"A suggested answer appears after {left} more attempt{'s' if left != 1 else ''}.")
            else:
                lines.append("    Improve the points above and try again.")
        return "\n".join(lines)


def as_feedback(result) -> Feedback:
    """Accept either a Feedback or an (ok, message) pair, for older session modules."""
    if isinstance(result, Feedback):
        return result
    ok, msg = result
    return Feedback(ok, msg)


def combine(value_fb: Feedback | None, written_fb: Feedback | None) -> Feedback | None:
    """Merge the feedback on a value with the feedback on the written reasoning."""
    if value_fb is None or written_fb is None:
        return value_fb or written_fb
    details = list(value_fb.details)
    details.append(written_fb.summary)
    details += [f"    {d}" for d in written_fb.details]
    parts = []
    if value_fb.ok is False and value_fb.suggestion:
        parts.append("Value: " + value_fb.suggestion)
    if written_fb.open_points and written_fb.suggestion:
        parts.append("Reasoning: " + written_fb.suggestion)
    return Feedback(value_fb.ok, value_fb.summary, details, "\n".join(parts) or None, written_fb.open_points)
