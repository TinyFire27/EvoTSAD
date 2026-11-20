"""Persist the settings and experiment history for one EvoTSAD task."""

import fcntl
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from shutil import rmtree
from typing import Any, TextIO

from .errors import HarnessExecutionError


@dataclass
class PendingTrial:
    """The reserved paths and history stream for one trial."""

    trial_number: int
    spec_path: Path
    output_dir: Path
    _history_stream: TextIO
    _recorded: bool = field(default=False, init=False)


class TaskState:
    """Open an existing task and reconstruct its state from history.jsonl."""

    def __init__(self, task_dir: Path) -> None:
        self.task_dir = task_dir.expanduser().resolve()
        self.task_path = self.task_dir / "task.json"
        self.history_path = self.task_dir / "history.jsonl"
        self.experiments_dir = self.task_dir / "experiments"

        self.task = json.loads(self.task_path.read_text(encoding="utf-8"))
        self.data: dict[str, str | None] = self.task.get("data", {})
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
        with self.history_path.open(encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_SH)
            return self._read_history(stream)

    @property
    def recorded_trials(self) -> int:
        """Count trials with a successful evaluation."""
        return sum(event["status"] == "completed" for event in self.history)

    @property
    def best_experiment(self) -> dict[str, Any] | None:
        completed = (event for event in self.history if event["status"] == "completed")
        return max(
            completed, key=lambda event: event["metrics"][self.primary_metric], default=None
        )

    @contextmanager
    def prepare_trial(self) -> Iterator[PendingTrial]:
        """Reserve an experiment directory and clean it up if the trial is unrecorded."""
        with self.history_path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            history = self._read_history(stream)
            trial_number = 1 + sum(event["status"] == "completed" for event in history)
            max_trials = self.task["max_trials"]
            if trial_number > max_trials:
                raise ValueError(f"Trial budget reached ({max_trials}/{max_trials}).")

            number = trial_number
            experiment_dir = self.experiments_dir / f"exp-{number:03d}"
            while experiment_dir.exists():
                number += 1
                experiment_dir = self.experiments_dir / f"exp-{number:03d}"
            experiment_dir.mkdir()
            trial = PendingTrial(
                trial_number=trial_number,
                spec_path=experiment_dir / "spec.json",
                output_dir=experiment_dir / "artifacts",
                _history_stream=stream,
            )
            try:
                yield trial
            finally:
                if not trial._recorded:
                    rmtree(experiment_dir)

    def record_trial(
        self, trial: PendingTrial, metrics: dict[str, float], *, reason: str = ""
    ) -> dict[str, Any]:
        """Validate and record the metrics for an active trial."""
        if trial._recorded or trial._history_stream.closed:
            raise HarnessExecutionError("The trial is no longer pending.")
        score = metrics.get(self.primary_metric)
        if type(score) not in (int, float) or not isfinite(score):
            raise HarnessExecutionError(f"Expected a finite {self.primary_metric} score.")

        record = {
            "experiment_id": trial.spec_path.parent.name,
            "trial_number": trial.trial_number,
            "spec_path": str(trial.spec_path),
            "output_dir": str(trial.output_dir),
            "status": "completed",
            "reason": reason,
            "metrics": metrics,
        }
        result = self._append(trial._history_stream, record)
        trial._recorded = True
        return result

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        """Return the latest recorded state of one experiment."""
        return self._find_experiment(self.history, experiment_id)

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
