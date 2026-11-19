"""Persist the settings and experiment history for one EvoTSAD task."""

import json
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any, Literal, TextIO
from .file_lock import locked


class TaskState:
    """Open an existing task and reconstruct its state from history.jsonl."""

    def __init__(self, task_dir: Path) -> None:
        self.task_dir = task_dir.expanduser().resolve()
        self.task_path = self.task_dir / "task.json"
        self.history_path = self.task_dir / "history.jsonl"
        self.experiments_dir = self.task_dir / "experiments"
        self.task = json.loads(self.task_path.read_text(encoding="utf-8"))
        self.primary_metric: str = self.task["primary_metric"]

    @classmethod
    def create(cls, task_dir: Path, task: dict[str, Any]) -> "TaskState":
        """Create a task directory and save the agreed settings once."""
        metric = task.get("primary_metric")
        if not isinstance(metric, str) or not metric.strip():
            raise ValueError("task.primary_metric must name the selection metric.")
        content = json.dumps(task, ensure_ascii=False, indent=2, allow_nan=False)
        task_dir = task_dir.expanduser().resolve()
        task_dir.mkdir(parents=True)
        (task_dir / "experiments").mkdir()
        (task_dir / "history.jsonl").touch()
        (task_dir / "task.json").write_text(content + "\n", encoding="utf-8")
        return cls(task_dir)

    @property
    def history(self) -> list[dict[str, Any]]:
        # Use read/write mode so Windows can acquire its advisory lock.
        with self.history_path.open("r+", encoding="utf-8") as stream:
            with locked(stream, shared=True):
                return self._read_history(stream)

    @property
    def recorded_trials(self) -> int:
        """Count registered experiments that have reached a terminal state."""
        return sum(event["status"] != "created" for event in self.history)

    @property
    def best_experiment(self) -> dict[str, Any] | None:
        completed = (event for event in self.history if event["status"] == "completed")
        return max(
            completed, key=lambda event: event["metrics"][self.primary_metric], default=None
        )

    def create_experiment(self, spec: dict[str, Any], *, reason: str = "") -> dict[str, Any]:
        """Register one candidate and return its ID, spec path, and output path."""
        content = json.dumps(spec, ensure_ascii=False, indent=2, allow_nan=False)
        with self.history_path.open("a+", encoding="utf-8") as stream:
            with locked(stream):
                history = self._read_history(stream)
                number = 1 + sum(event["status"] == "created" for event in history)
                experiment_dir = self.experiments_dir / f"exp-{number:03d}"
                while experiment_dir.exists():
                    number += 1
                    experiment_dir = self.experiments_dir / f"exp-{number:03d}"
                experiment_dir.mkdir()
                spec_path = experiment_dir / "spec.json"
                spec_path.write_text(content + "\n", encoding="utf-8")
                experiment = {
                    "experiment_id": experiment_dir.name,
                    "spec_path": str(spec_path),
                    "output_dir": str(experiment_dir / "artifacts"),
                    "status": "created",
                    "reason": reason,
                }
                return self._append(stream, experiment)

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        """Return the latest recorded state of one experiment."""
        return self._find_experiment(self.history, experiment_id)

    def record_experiment(
        self,
        experiment_id: str,
        *,
        status: Literal["completed", "failed", "abandoned"] = "completed",
        metrics: dict[str, float] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Record a terminal result supplied by the experiment tool."""
        if status not in {"completed", "failed", "abandoned"}:
            raise ValueError(f"Invalid experiment status: {status}")
        if status == "completed":
            score = (metrics or {}).get(self.primary_metric)
            if type(score) not in (int, float) or not isfinite(score):
                raise ValueError(
                    f"A completed experiment needs a finite {self.primary_metric} score."
                )

        result = {"status": status, "metrics": metrics, "error": error}
        with self.history_path.open("a+", encoding="utf-8") as stream:
            with locked(stream):
                experiment = self._find_experiment(self._read_history(stream), experiment_id)
                if experiment["status"] == "created":
                    return self._append(stream, {**experiment, **result})
                if all(experiment[key] == value for key, value in result.items()):
                    return experiment
                raise ValueError(f"Experiment already has a recorded result: {experiment_id}")

    @staticmethod
    def _find_experiment(history: list[dict[str, Any]], experiment_id: str) -> dict[str, Any]:
        for event in reversed(history):
            if event["experiment_id"] == experiment_id:
                return event
        raise ValueError(f"Unknown experiment: {experiment_id}")

    @staticmethod
    def _read_history(stream: TextIO) -> list[dict[str, Any]]:
        stream.seek(0)
        return [json.loads(line) for line in stream if line.strip()]

    @staticmethod
    def _append(stream: TextIO, event: dict[str, Any]) -> dict[str, Any]:
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        return event
