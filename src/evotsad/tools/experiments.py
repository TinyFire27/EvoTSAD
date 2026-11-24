"""Tools available to the EvoTSAD experiment agent."""

import json
import shlex
from tempfile import TemporaryDirectory
import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from shutil import rmtree
from typing import Any, TextIO

from ..runtime.errors import HarnessExecutionError
from . import ToolExecutor
from .bash import run_bash


TASK_INIT_TOOL = {
    "type": "function",
    "name": "task_init",
    "description": "Initialize a task, replacing the previous task directory and its outputs.",
    "strict": False,
    "parameters": {
        "type": "object",
        "properties": {
            "objective": {"type": "string", "description": "The task goal."},
            "data": {
                "type": "object",
                "properties": {
                    "train": {"type": "string", "description": "Training CSV path."},
                    "val": {
                        "type": ["string", "null"],
                        "description": "Optional training validation CSV for early stopping.",
                    },
                    "eval": {
                        "type": "string",
                        "description": "Labeled CSV for evaluating trials during iteration.",
                    },
                },
                "required": ["train", "eval"],
                "additionalProperties": False,
            },
            "max_trials": {"type": "integer", "description": "The agreed trial budget."},
            "primary_metric": {
                "type": ["string", "null"],
                "description": "Metric for comparing trials. Use null for VUS-PR.",
            },
            "seed": {
                "type": ["integer", "null"],
                "description": "Shared random seed. Use null for 2026.",
            },
        },
        "required": ["objective", "data", "max_trials"],
        "additionalProperties": False,
    },
}

LIST_COMPONENTS_TOOL = {
    "type": "function",
    "name": "list_components",
    "description": "List available components or inspect one component's parameters.",
    "strict": False,
    "parameters": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["detectors", "preprocessors"]},
            "name": {"type": "string", "description": "Exact class name, if known."},
        },
        "required": ["kind"],
        "additionalProperties": False,
    },
}

RUN_TRIAL_TOOL = {
    "type": "function",
    "name": "run_trial",
    "description": "Train and evaluate a candidate pipeline. Successful evaluations use one trial.",
    "strict": False,
    "parameters": {
        "type": "object",
        "properties": {
            "pipeline": {
                "type": "array",
                "description": "Zero or more preprocessing steps followed by one detector. Use an array even for a detector alone.",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "class": {"type": "string", "description": "Exact component class name."},
                        "params": {"type": "object", "description": "Optional constructor parameters."},
                    },
                    "required": ["class"],
                    "additionalProperties": False,
                },
            },
            "reason": {"type": "string", "description": "Why this candidate is worth evaluating."},
        },
        "required": ["pipeline"],
        "additionalProperties": False,
    },
}


@dataclass
class Trial:
    """The experiment ID, paths, and spec for a reserved trial."""

    trial_number: int
    experiment_dir: Path
    experiment_id: str
    spec_path: Path
    spec: dict
    recorded: bool = field(default=False, init=False)


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
        with self.history_path.open(encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_SH)
            return self._read_history(stream)

    @property
    def recorded_trials(self) -> int:
        """Count trials with a successful evaluation."""
        return len(self.history)

    @property
    def best_experiment(self) -> dict[str, Any] | None:
        return max(
            self.history, key=lambda trial: trial["metrics"][self.primary_metric], default=None
        )

    @contextmanager
    def create_trial(self, spec) -> Iterator[Trial]:
        """Reserve an experiment directory and clean it up if the trial is unrecorded."""
        with self.history_path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            history = self._read_history(stream)
            trial_number = len(history) + 1
            max_trials = self.task["max_trials"]
            if trial_number > max_trials:
                raise ValueError(f"Trial budget reached ({max_trials}/{max_trials}).")

            number = trial_number
            experiment_dir = self.experiments_dir / f"exp-{number:03d}"
            while experiment_dir.exists():
                number += 1
                experiment_dir = self.experiments_dir / f"exp-{number:03d}"
            experiment_dir.mkdir()

            trial = Trial(
                trial_number=trial_number,
                spec_path=experiment_dir / "spec.json",
                experiment_dir=experiment_dir,
                experiment_id=experiment_dir.name,
                spec=spec,
            )

            try:
                trial.spec_path.write_text(
                    json.dumps(spec, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                yield trial
            finally:
                if not trial.recorded:
                    rmtree(experiment_dir)

    def record_trial(
        self, trial: Trial, *, reason: str, metrics: dict[str, float]
    ) -> dict[str, Any]:
        """Append a successful trial with the spec used for evaluation."""

        record = {
            "trial_number": trial.trial_number,
            "experiment_id": trial.experiment_id,
            "experiment_dir": str(trial.experiment_dir),
            "spec_path": str(trial.spec_path),
            "spec": trial.spec,
            "reason": reason,
            "metrics": metrics,
        }

        with self.history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

        trial.recorded = True
        return record

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        """Return the latest recorded state of one experiment."""
        for event in reversed(self.history):
            if event["experiment_id"] == experiment_id:
                return event
        raise ValueError(f"Unknown experiment: {experiment_id}")

    @staticmethod
    def _read_history(stream: TextIO) -> list[dict[str, Any]]:
        stream.seek(0)
        return [json.loads(line) for line in stream if line.strip()]


class ExperimentToolExecutor(ToolExecutor):
    """Bind the shared tools and experiment tools to a single task."""

    def __init__(self, workspace: Path, *, timeout: int = 600) -> None:
        super().__init__(workspace, timeout=timeout)
        self.timeout = timeout
        self.task_dir = self.workspace / ".evotsad"
        self.task_state: TaskState | None = None  # 一个实验执行器必须绑定一个 task_state
        self.schemas = [*self.schemas, TASK_INIT_TOOL, LIST_COMPONENTS_TOOL, RUN_TRIAL_TOOL]
        self._handlers["task_init"] = self.task_init
        self._handlers["list_components"] = self.list_components
        self._handlers["run_trial"] = self.run_trial

    def task_init(
        self,
        *,
        objective: str,
        data: dict[str, str | None],
        max_trials: int,
        primary_metric: str | None = None,
        seed: int | None = None,
    ) -> dict:
        """Validate the task settings, then create its persistent state."""
        if not objective.strip():
            raise ValueError("objective must describe the task goal.")
        if type(max_trials) is not int or max_trials < 1:
            raise ValueError("max_trials must be a positive integer.")
        if seed is not None and type(seed) is not int:
            raise ValueError("seed must be an integer.")

        paths = {}
        for split in ("train", "val", "eval"):
            value = data.get(split)
            if split == "val" and value is None:
                paths[split] = None
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"data.{split} must be a CSV path.")
            path = (self.workspace / Path(value).expanduser()).resolve()
            if path.suffix.lower() != ".csv" or not path.is_file():
                raise ValueError(f"data.{split} must point to an existing CSV file: {path}")
            paths[split] = str(path)
        if paths["train"] == paths["eval"]:
            raise ValueError("data.train and data.eval must be different files.")

        metric = "VUS-PR" if primary_metric is None else primary_metric.strip()
        task = {
            "objective": objective.strip(),
            "data": paths,
            "max_trials": max_trials,
            "primary_metric": metric,
            "metrics": {metric: {}},
            "seed": 2026 if seed is None else seed,
        }
        if self.task_dir.is_symlink():
            raise ValueError("The task directory must not be a symbolic link.")
        task_dir = self.task_dir.resolve()
        if task_dir.exists():
            if not (
                (task_dir / "task.json").is_file()
                and (task_dir / "history.jsonl").is_file()
                and (task_dir / "experiments").is_dir()
            ):
                raise ValueError(f"Refusing to replace an existing non-task directory: {task_dir}")
            if any(path and Path(path).is_relative_to(task_dir) for path in paths.values()):
                raise ValueError("Place input CSV files outside the task directory before replacing it.")
            with TemporaryDirectory(dir=task_dir.parent) as directory:
                staging = Path(directory)
                prepared = TaskState.create(staging / "new", task)
                previous = staging / "previous"
                task_dir.rename(previous)
                try:
                    prepared.task_dir.rename(task_dir)
                except OSError:
                    previous.rename(task_dir)
                    raise
            self.task_state = TaskState(task_dir)
        else:
            self.task_state = TaskState.create(task_dir, task)
        return {
            "status": "Task initialized successfully.",
            "task_dir": str(self.task_state.task_dir),
            "task_path": str(self.task_state.task_path),
            "history_path": str(self.task_state.history_path),
            "experiments_dir": str(self.task_state.experiments_dir),
            "task": self.task_state.task,
        }

    def list_components(self, *, kind: str, name: str | None = None) -> dict:
        """Discover available detector and preprocessor classes."""
        categories = {"detectors": "algorithms", "preprocessors": "preprocessors"}
        if kind not in categories:
            raise ValueError("kind must be detectors or preprocessors.")
        args = ["ls", categories[kind], "--task", "anomaly_detection"]
        if name is not None:
            args.extend(["--name", name])
        result = self._libraryts(*args)
        if isinstance(result, list):
            return {
                "status": f"Successfully listed {len(result)} {kind}.",
                "components": result,
            }
        return {"status": f"Successfully retrieved details for {name}.", **result}

    def run_trial(self, *, pipeline: list[dict], reason: str = "") -> dict:
        """Evaluate a candidate pipeline using the initialized task settings."""
        state = self.task_state
        if state is None:
            raise ValueError("Call task_init before running a trial.")
        if not isinstance(pipeline, list) or not pipeline:
            raise ValueError("pipeline must be a non-empty list of steps.")
        for step in pipeline:
            name = step.get("class") if isinstance(step, dict) else None
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Each pipeline step must specify a class name.")
            params = step.get("params", {})
            if step.keys() - {"class", "params"} or not isinstance(params, dict):
                raise ValueError("Pipeline steps accept a class and an optional params object.")

        task_data = state.task["data"]
        data = {"train": task_data["train"], "test": task_data["eval"]}
        if task_data.get("val") is not None:
            data["val"] = task_data["val"]

        # 构建好实验 spec
        spec = {
            "task": "anomaly_detection",
            "data": data,
            "estimator": pipeline,
            "metrics": state.task["metrics"],
            "seed": state.task["seed"],
        }

        # with 尝试执行实验，如果失败就清除
        with state.create_trial(spec) as trial:
            result = self._libraryts("run", "--spec", str(trial.spec_path), "--output-dir", str(trial.experiment_dir))
            metrics = result["metrics"]
            record = state.record_trial(trial, reason=reason, metrics=metrics)  # 预算 - 1

        best = state.best_experiment or record
        return {
            "status": "completed",
            "trial_number": trial.trial_number,
            "experiment_id": trial.experiment_id,
            "experiment_dir": str(trial.experiment_dir),
            "spec_path": str(trial.spec_path),
            "metrics": metrics,
            "progress": f"{record['trial_number']:02d}/{state.task['max_trials']}",
            "best_experiment": {
                "trial_number": best["trial_number"],
                "experiment_id": best["experiment_id"],
                "experiment_dir": best["experiment_dir"],
                "spec_path": best["spec_path"],
                "metrics": best["metrics"]
            },
        }

    def _libraryts(self, *args: str) -> dict | list:
        result = run_bash(
            shlex.join(["libraryts", *args]), cwd=self.workspace, timeout=self.timeout
        )
        exit_code = result["exit_code"]
        stdout = result["stdout"]

        if exit_code == 0:
            try:
                return json.loads(stdout)
            except json.JSONDecodeError as exc:
                raise HarnessExecutionError(
                    "LibraryTS exited successfully but returned invalid JSON."
                ) from exc

        if exit_code == 1:
            try:
                payload = json.loads(stdout)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                error = payload.get("error")
                if isinstance(error, str) and error.strip():
                    raise RuntimeError(error)

        detail = result["stderr"].strip() or stdout.strip()
        message = f"LibraryTS failed with exit code {exit_code}."
        if detail:
            message = f"{message} {detail}"
        raise HarnessExecutionError(message)
