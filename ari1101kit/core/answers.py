"""The answer book: records your answers and reasoning in a JSON file you can submit.

Answers are formative. Each call to answer() saves immediately, so nothing is lost
if the notebook stops; submit() checks what is missing and prepares the file.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .._version import __version__

ANSWER_DIR = Path("ari1101_answers")


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
    def __init__(self, session: str, student: str, tasks: dict):
        self.session, self.student, self.tasks = session, student, tasks
        ANSWER_DIR.mkdir(exist_ok=True)
        safe = "".join(ch if ch.isalnum() else "_" for ch in student).strip("_") or "student"
        self.path = ANSWER_DIR / f"{session}_{safe}.json"
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
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")

    def record(self, task_id: str, value=None, reasoning: str | None = None):
        if task_id not in self.tasks:
            raise KeyError(f"{task_id} is not a task in {self.session}. Tasks: {', '.join(self.tasks)}")
        task = self.tasks[task_id]
        entry = {"value": _serialise(value), "reasoning": (reasoning or "").strip(),
                 "saved_at": datetime.now().isoformat(timespec="seconds")}
        self.data["answers"][task_id] = entry
        self._save()
        missing = []
        if task.get("needs_value") and value is None:
            missing.append("a value")
        if task.get("needs_reasoning") and not entry["reasoning"]:
            missing.append("your reasoning")
        msg = f"Saved {task_id}."
        if missing:
            msg += f" Still to add: {' and '.join(missing)}."
        print(msg)

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
            try:
                from google.colab import files  # type: ignore
                files.download(str(self.path))
                print("The file is being downloaded by your browser. Upload it to the VLE.")
            except ImportError:
                print("Upload this file to the VLE.")
        return self.path
