"""Tools available to the EvoTSAD experiment agent."""

import json
import shlex
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from ..runtime.errors import HarnessExecutionError
from ..runtime.task_state import TaskState
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

        data = {"train": state.data["train"], "test": state.data["eval"]}
        if state.data.get("val") is not None:
            data["val"] = state.data["val"]
        spec = {
            "task": "anomaly_detection",
            "data": data,
            "estimator": pipeline,
            "metrics": state.task["metrics"],
            "seed": state.task["seed"],
        }
        content = json.dumps(spec, ensure_ascii=False, indent=2, allow_nan=False)
        with state.prepare_trial() as pending:
            pending.spec_path.write_text(content + "\n", encoding="utf-8")
            metrics = self._evaluate_spec(pending.spec_path, pending.output_dir)
            trial = state.record_trial(pending, metrics, reason=reason)
        best = state.best_experiment or trial
        return {
            "experiment_id": trial["experiment_id"],
            "status": trial["status"],
            "timestamp": trial["timestamp"],
            "progress": f"{trial['trial_number']:02d}/{state.task['max_trials']}",
            "spec_path": trial["spec_path"],
            "output_dir": trial["output_dir"],
            "metrics": trial["metrics"],
            "best_experiment": {
                "experiment_id": best["experiment_id"],
                "trial_number": best["trial_number"],
                "spec_path": best["spec_path"],
                "output_dir": best["output_dir"],
                "metrics": best["metrics"],
                "timestamp": best["timestamp"],
            },
        }

    def _evaluate_spec(self, spec_path: Path, output_dir: Path) -> dict[str, float]:
        result = self._libraryts("run", "--spec", str(spec_path), "--output-dir", str(output_dir))
        metrics = result.get("metrics") if isinstance(result, dict) else None
        if not isinstance(metrics, dict):
            raise HarnessExecutionError(
                "LibraryTS returned no evaluation metrics after a successful run."
            )
        return metrics

    def _libraryts(self, *args: str) -> dict | list:
        result = run_bash(
            (subprocess.list2cmdline(["libraryts", *args]) if os.name == "nt"
             else shlex.join(["libraryts", *args])), cwd=self.workspace, timeout=self.timeout
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
