"""The answer book: records your answers and reasoning in a JSON file you can submit.

The file is kept in ari1101_answers/ inside your work folder (see kit.start),
which is the folder of the notebook itself, in Colab as on your own computer.
Answers are formative. Each call to answer() saves immediately, so nothing is lost
if the notebook stops; submit() checks what is missing and prepares the file.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .._version import __version__

ANSWER_DIR = "ari1101_answers"


def _serialise(value):
    """Convert an answer into something JSON can store, keeping enough to review it."""
    if isinstance(value, pd.DataFrame):
        return {"kind": "DataFrame", "shape": list(value.shape),
                "columns": [str(c) for c in value.columns],
                "dtypes": {str(c): str(t) for c, t in value.dtypes.items()},
                "head": json.loads(value.head(15).to_json(orient="records", date_format="iso"))}
    if isinstance(value, pd.Series):
        return {"kind": "Series", "name": str(value.name), "length": len(value),
                "values": json.loads(value.head(30).to_json(orient="values", date_format="iso"))}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if math.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_serialise(v) for v in value]
    if isinstance(value, (list, set)):
        return [_serialise(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialise(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


class AnswerBook:
    def __init__(self, session: str, student: str, tasks: dict, folder: Path | None = None):
        self.session, self.student, self.tasks = session, student, tasks
        answer_dir = Path(folder or Path.cwd()).resolve() / ANSWER_DIR
        answer_dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch if ch.isalnum() else "_" for ch in student).strip("_") or "student"
        self.path = answer_dir / f"{session}_{safe}.json"
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            print(f"Continuing your answer book: {self.path} ({len(self.data['answers'])} answers so far)")
        else:
            self.data = {"session": session, "student": student, "kit_version": __version__,
                         "started": datetime.now().isoformat(timespec="seconds"), "answers": {}}
            self._save()
            print(f"New answer book: {self.path}")

    def _save(self):
        self.data["updated"] = datetime.now().isoformat(timespec="seconds")
        text = json.dumps(self.data, indent=2, ensure_ascii=False)
        with open(self.path, "w", encoding="utf-8") as f:   # write, then force it out of the cache
            f.write(text)
            f.flush()
            os.fsync(f.fileno())

    def record(self, task_id: str, value=None, reasoning: str | None = None):
        if task_id not in self.tasks:
            raise KeyError(f"{task_id} is not a task in {self.session}. Tasks: {', '.join(self.tasks)}")
        task = self.tasks[task_id]
        previous = self.data["answers"].get(task_id, {})
        attempts = previous.get("attempts", 1 if previous else 0) + 1
        entry = {"value": _serialise(value), "reasoning": (reasoning or "").strip(),
                 "saved_at": datetime.now().isoformat(timespec="seconds"), "attempts": attempts}
        if value is None and "value" in previous:      # reasoning added later: keep the value
            entry["value"] = previous["value"]
        if not entry["reasoning"] and previous.get("reasoning"):   # value corrected later: keep the reasoning
            entry["reasoning"] = previous["reasoning"]
        for key in ("result", "first_correct_attempt", "reasoning_points"):
            if key in previous:
                entry[key] = previous[key]
        self.data["answers"][task_id] = entry
        self._save()
        missing = []
        if task.get("needs_value") and entry["value"] is None:
            missing.append("a value")
        if task.get("needs_reasoning") and not entry["reasoning"]:
            missing.append("your reasoning")
        msg = f"Saved {task_id} in {self.path}"
        if missing:
            msg += f" Still to add: {' and '.join(missing)}."
        print(msg)
        return attempts

    def note_result(self, task_id: str, ok=None, written_complete=None) -> None:
        """Store the outcome of the automatic checks with the answer."""
        entry = self.data["answers"].get(task_id)
        if entry is None:
            return
        if ok is not None or "result" not in entry:
            entry["result"] = {True: "correct", False: "not yet", None: "not checked"}[ok]
        if ok is True and "first_correct_attempt" not in entry:
            entry["first_correct_attempt"] = entry.get("attempts")
        if written_complete is not None:
            entry["reasoning_points"] = "all found" if written_complete else "some not found"
        self._save()

    def progress(self) -> pd.DataFrame:
        rows = []
        for tid, task in self.tasks.items():
            a = self.data["answers"].get(tid)
            rows.append({"task": tid, "title": task["title"],
                         "answered": a is not None,
                         "has_reasoning": bool(a and a["reasoning"]) if task.get("needs_reasoning") else None})
        return pd.DataFrame(rows)

    def submit(self, download: bool = True) -> Path:
        prog = self.progress()
        todo = prog[~prog["answered"]]["task"].tolist()
        no_reason = prog[prog["has_reasoning"] == False]["task"].tolist()  # noqa: E712
        self.data["submitted"] = datetime.now().isoformat(timespec="seconds")
        self._save()
        print(f"Answer book: {self.path}")
        print(f"Answered {int(prog['answered'].sum())} of {len(prog)} tasks.")
        if todo:
            print("Not yet answered:", ", ".join(todo))
        if no_reason:
            print("Answered without the reasoning asked for:", ", ".join(no_reason))
        if download:
            self.download()
        return self.path

    def download(self) -> None:
        try:
            from google.colab import files  # type: ignore
            files.download(str(self.path))
            print("The file is being downloaded by your browser. Upload it to the VLE.")
        except ImportError:
            print("Upload this file to the VLE.")
