"""Tools available to the EvoTSAD experiment agent."""

import json
import shlex
import os
import subprocess
import csv
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from shutil import copy2, rmtree
from tempfile import TemporaryDirectory
from copy import deepcopy
from datetime import datetime, timezone
from time import perf_counter
import platform
from importlib.metadata import version, PackageNotFoundError
from typing import Any, TextIO
from math import isfinite

from ..runtime.data_profile import training_profile
from ..evolution.contrastive import model_reflector

from ..runtime.errors import HarnessExecutionError
from ..runtime.file_lock import locked
from ..evolution.memory import ExperienceMemory
from . import ToolExecutor, function_tool
from .bash import run_bash


BASELINE_DETECTORS = ("SubLOF", "SubOCSVM", "SubPCA", "IForest", "HBOS")


TASK_INIT_TOOL = {
    "type": "function",
    "name": "task_init",
    "description": "Initialize a task, replacing the previous task directory and its outputs.",
    "strict": False,
    "parameters": {
        "type": "object",
        "properties": {
            "objective": {"type": "string", "description": "The task goal."},
            "task_description": {"type": "string", "description": "Full user task wording and amendments, including all agreed constraints; do not summarize."},
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
        "required": ["objective", "task_description", "data", "max_trials"],
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

SEARCH_MEMORY_TOOL = {
    "type": "function", "name": "search_memory",
    "description": "Search successful experiments from previous tasks by keyword; an empty query returns recent experiences.",
    "strict": False,
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string"}, "limit": {"type": "integer"}
    }, "additionalProperties": False},
}


FINISH_TASK_TOOL = {
    "type": "function", "name": "finish_task",
    "description": "End the entire task and publish its memory once. Call when stopping early; budget exhaustion ends automatically. No more trials are allowed after this.",
    "strict": False,
    "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "additionalProperties": False},
}


GET_HISTORY_TOOL = function_tool(
    "get_history", "Return the active task's successful trial records in trial_history, in recorded order.",
)
GET_EXPERIMENT_TOOL = function_tool(
    "get_experiment", "Return one full record from the active task in experiment, by its exact ID.",
    experiment_id="Exact experiment ID, for example exp-001.",
)
GET_BEST_EXPERIMENT_TOOL = function_tool(
    "get_best_experiment", "Return the active task's highest-scoring record in best_experiment.",
)


@dataclass
class Trial:
    trial_number: int
    experiment_dir: Path
    spec_path: Path
    spec: dict
    history_stream: TextIO
    recorded: bool = field(default=False, init=False)

    @property
    def experiment_id(self) -> str:
        return self.experiment_dir.name


class TaskState:
    """Persist task settings and successful trials in a flat artifact layout."""

    def __init__(self, task_dir: Path) -> None:
        self.task_dir = task_dir.expanduser().resolve()
        self.task_path = self.task_dir / "task.json"
        self.profile_path = self.task_dir / "profile.json"
        self.history_path = self.task_dir / "history.jsonl"
        self.experiments_dir = self.task_dir / "experiments"
        self.task = json.loads(self.task_path.read_text(encoding="utf-8"))
        self.primary_metric: str = self.task["primary_metric"]

    @classmethod
    def create(cls, task_dir: Path, task: dict[str, Any]) -> "TaskState":
        if not isinstance(task.get("primary_metric"), str) or not task["primary_metric"].strip():
            raise ValueError("task.primary_metric must name the selection metric.")
        task_dir = task_dir.expanduser().resolve()
        task_dir.mkdir(parents=True)
        (task_dir / "experiments").mkdir()
        (task_dir / "history.jsonl").touch()
        (task_dir / "task.json").write_text(
            json.dumps(task, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return cls(task_dir)

    def experiment_spec(self, pipeline: list[dict]) -> dict:
        data = self.task["data"]
        spec_data = {"train": data["train"], "test": data["eval"]}
        if data.get("val") is not None:
            spec_data["val"] = data["val"]
        return {
            "task": "anomaly_detection",
            "data": spec_data,
            "estimator": [
                {"class": step["class"], "init_params": step.get("params", {})} for step in pipeline
            ],
            "metrics": self.task["metrics"],
            "seed": self.task["seed"],
        }

    @property
    def history(self) -> list[dict[str, Any]]:
        with self.history_path.open("a+", encoding="utf-8") as stream:
            with locked(stream, shared=True):
                return self._read_history(stream)

    @property
    def recorded_trials(self) -> int:
        return len(self.history)

    @property
    def best_experiment(self) -> dict[str, Any] | None:
        return max(self.history, key=lambda trial: trial["metrics"][self.primary_metric], default=None)

    @contextmanager
    def create_trial(self, spec: dict) -> Iterator[Trial]:
        with self.history_path.open("a+", encoding="utf-8") as stream:
            with locked(stream):
                trial_number = len(self._read_history(stream)) + 1
                max_trials = self.task["max_trials"]
                if trial_number > max_trials:
                    raise ValueError(f"Trial budget reached ({max_trials}/{max_trials}).")
                number = trial_number
                experiment_dir = self.experiments_dir / f"exp-{number:03d}"
                while experiment_dir.exists():
                    number += 1
                    experiment_dir = self.experiments_dir / f"exp-{number:03d}"
                experiment_dir.mkdir()
                trial = Trial(trial_number, experiment_dir, experiment_dir / "spec.json", spec, stream)
                try:
                    trial.spec_path.write_text(
                        json.dumps(spec, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8",
                    )
                    yield trial
                finally:
                    if not trial.recorded:
                        rmtree(experiment_dir)

    def record_trial(self, trial: Trial, *, reason: str, metrics: dict[str, float]) -> dict[str, Any]:
        if trial.recorded or trial.history_stream.closed:
            raise HarnessExecutionError("The trial is no longer pending.")
        score = metrics.get(self.primary_metric)
        if type(score) not in (int, float) or not isfinite(score):
            raise HarnessExecutionError(f"Expected a finite {self.primary_metric} score.")
        record = {
            "trial_number": trial.trial_number,
            "experiment_id": trial.experiment_id,
            "experiment_dir": str(trial.experiment_dir),
            "spec_path": str(trial.spec_path),
            "spec": trial.spec,
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "metrics": metrics,
        }
        trial.history_stream.seek(0, 2)
        trial.history_stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        trial.history_stream.flush()
        trial.recorded = True
        return record

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        for record in reversed(self.history):
            if record["experiment_id"] == experiment_id:
                return record
        raise ValueError(f"Unknown experiment: {experiment_id}")

    @staticmethod
    def _read_history(stream: TextIO) -> list[dict[str, Any]]:
        stream.seek(0)
        return [json.loads(line) for line in stream if line.strip()]


class ExperimentToolExecutor(ToolExecutor):
    """Bind the shared tools and experiment tools to a single task."""

    def __init__(self, workspace: Path, *, timeout: int = 600, memory_mode: str = "all", update_memory: bool = True) -> None:
        super().__init__(workspace, timeout=timeout)
        self.timeout = timeout
        self.task_dir = self.workspace / ".evotsad"
        self.task_state: TaskState | None = None  # 一个实验执行器必须绑定一个 task_state
        self.memory = ExperienceMemory(self.workspace, memory_mode)
        self.update_memory = update_memory
        self.reflector = None
        self.attempts = []
        self.pending_user_text = []
        self.finish_result = None
        self.schemas = [
            *self.schemas, TASK_INIT_TOOL, LIST_COMPONENTS_TOOL, RUN_TRIAL_TOOL,
            SEARCH_MEMORY_TOOL, FINISH_TASK_TOOL,
            GET_HISTORY_TOOL, GET_EXPERIMENT_TOOL, GET_BEST_EXPERIMENT_TOOL,
        ]
        self._handlers["task_init"] = self.task_init
        self._handlers["list_components"] = self.list_components
        self._handlers["run_trial"] = self.run_trial
        self._handlers["search_memory"] = self.search_memory
        self._handlers["finish_task"] = self.finish_task
        self._handlers["get_history"] = self.get_history
        self._handlers["get_experiment"] = self.get_experiment
        self._handlers["get_best_experiment"] = self.get_best_experiment

    def set_reflection_model(self, model) -> None:
        self.reflector = model_reflector(model)

    def note_user_input(self, text: str) -> None:
        if not text.strip():
            return
        if self.task_state is None or self.task_state.task.get("status") == "ended":
            self.pending_user_text.append(text)
        else:
            task = self.task_state.task
            task["task_description"] += "\n\nUser amendment (verbatim):\n" + text
            self._save_task()

    def _save_task(self) -> None:
        self.task_state.task_path.write_text(
            json.dumps(self.task_state.task, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")

    def _current_profile(self) -> dict:
        state = self.task_state
        if state is None:
            return {}
        profile = json.loads(state.profile_path.read_text(encoding="utf-8"))
        if not isinstance(profile, dict):
            raise ValueError("profile.json must contain a JSON object.")
        state.task["profile"] = profile
        self._save_task()
        return profile

    def search_memory(self, *, query: str = "", limit: int = 5) -> dict:
        task = self.task_state.task if self.task_state else {}
        return {"mode": self.memory.mode, "experiences": self.memory.search(
            query, limit, profile=self._current_profile() if task else None,
            primary_metric=task.get("primary_metric"))}

    def finish_task(self, *, reason: str = "Task completed.") -> dict:
        state = self.task_state
        if state is None:
            raise ValueError("Call task_init first.")
        if state.task.get("status") == "ended":
            return self.finish_result or {"status": "ended", "memory_warning": "No memory result available; recovery is not implemented."}
        self._current_profile()
        state.task["status"] = "ended"
        state.task["end_reason"] = reason
        state.task["ended_at"] = datetime.now(timezone.utc).isoformat()
        self._save_task()
        if not self.update_memory:
            self.finish_result = {"status": "ended", "memory": {"mode": self.memory.mode, "saved": False, "updated": False, "reason": "Memory update disabled; existing memory remains readable."}}
        else:
            try:
                publication = self.memory.finalize(state.task, self.attempts, self.reflector)
                self.finish_result = {"status": "ended", "memory": publication}
            except Exception as exc:
                self.finish_result = {"status": "ended", "memory_warning": str(exc)}
        return self.finish_result

    def _record_attempt(self, attempt: dict) -> None:
        self.attempts.append(deepcopy(attempt))
        with (self.task_dir / "attempts.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(attempt, ensure_ascii=False, allow_nan=False)+"\n")

    def _baselines(self, state: TaskState) -> dict[str, dict]:
        results = {}
        with Path(state.task["data"]["eval"]).open(encoding="utf-8-sig", newline="") as stream:
            eval_rows = sum(1 for _ in csv.reader(stream)) - 1
        train_rows = state.task["profile"]["training_rows"]
        win_len = min(16, max(2, train_rows // 4), eval_rows)
        for name in BASELINE_DETECTORS:
            params = {}
            if name in {"SubLOF", "SubOCSVM", "SubPCA"}:
                if train_rows <= win_len or eval_rows < 2:
                    results[name] = {
                        "params": None, "metrics": None,
                        "error": "Windowed baselines need at least 3 training rows and 2 evaluation rows.",
                    }
                    continue
                params["win_len"] = win_len
                if name == "SubLOF":
                    params["n_neighbors"] = min(20, train_rows - win_len)
            spec = state.experiment_spec([{"class": name, "params": params}])
            with TemporaryDirectory(dir=state.task_dir) as directory:
                work_dir = Path(directory)
                spec_path = work_dir / "spec.json"
                spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
                try:
                    result = self._libraryts(
                        "run", "--spec", str(spec_path), "--output-dir", str(work_dir / "output")
                    )
                    metrics = result.get("metrics") if isinstance(result, dict) else None
                    if not isinstance(metrics, dict):
                        raise HarnessExecutionError("LibraryTS returned no evaluation metrics.")
                except (RuntimeError, HarnessExecutionError) as exc:
                    results[name] = {"params": params, "metrics": None, "error": str(exc)}
                else:
                    results[name] = {"params": params, "metrics": metrics, "error": None}
        return results

    def task_init(
        self,
        *,
        objective: str,
        data: dict[str, str | None],
        max_trials: int,
        primary_metric: str | None = None,
        seed: int | None = None,
        task_description: str | None = None,
    ) -> dict:
        """Validate the task settings, then create its persistent state."""
        if self.task_state is not None and self.task_state.task.get("status") != "ended":
            raise ValueError("Finish the current task before starting another one.")
        if task_description is not None and (not isinstance(task_description, str) or not task_description.strip()):
            raise ValueError("task_description must contain the complete task wording.")
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
        full_description = task_description or objective
        if self.pending_user_text:
            full_description = "User messages (verbatim):\n" + "\n\n".join(self.pending_user_text) + "\n\nConfirmed task requirements:\n" + full_description
        try:
            library_version = version("libraryts")
        except PackageNotFoundError:
            library_version = None
        training = training_profile(Path(paths["train"]))
        observations = [{
            "finding": "Training series structure",
            "evidence": f"{training['training_rows']} rows; {training['numeric_features']} numeric features; "
                        f"missing ratio {training['missing_ratio']}",
            "implication": "Consider missing values and feature count when selecting preprocessing and detectors.",
        }]
        profile = {
            **training,
            "length": training["training_rows"],
            "missing_ratio": training["missing_ratio"],
            "observations": observations,
        }
        task = {
            "status": "active", "memory_mode": self.memory.mode, "update_memory": self.update_memory,
            "task_description": full_description,
            "profile": profile,
            "software": {"python": platform.python_version(), "libraryts": library_version},
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
        state = self.task_state
        baselines = self._baselines(state)
        profile["baselines"] = baselines
        state.profile_path.write_text(
            json.dumps(profile, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        self._save_task()
        self.attempts = []
        self.finish_result = None
        self.pending_user_text = []
        return {
            "status": "Task initialized successfully.",
            "memory": self.search_memory(),
            "task_dir": str(self.task_state.task_dir),
            "task_path": str(self.task_state.task_path),
            "profile_path": str(state.profile_path),
            "history_path": str(self.task_state.history_path),
            "experiments_dir": str(self.task_state.experiments_dir),
            "task": self.task_state.task,
            "baselines": baselines,
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
        if state.task.get("status") == "ended":
            raise ValueError("This task has ended; start a new task to run more trials.")
        if not isinstance(pipeline, list) or not pipeline:
            raise ValueError("pipeline must be a non-empty list of steps.")
        for step in pipeline:
            name = step.get("class") if isinstance(step, dict) else None
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Each pipeline step must specify a class name.")
            params = step.get("params", {})
            if step.keys() - {"class", "params"} or not isinstance(params, dict):
                raise ValueError("Pipeline steps accept a class and an optional params object.")

        spec = state.experiment_spec(pipeline)
        attempt = {
            "task_description": state.task["task_description"],
            "pipeline": deepcopy(pipeline), "seed": state.task["seed"], "reason": reason,
            "software": state.task["software"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        # Reserve outside the catch: a budget rejection is not an executed attempt.
        with state.create_trial(spec) as pending:
            started = perf_counter()
            try:
                metrics = self._evaluate_spec(pending.spec_path, pending.experiment_dir)
                trial = state.record_trial(pending, metrics=metrics, reason=reason)
            except Exception as exc:
                error = str(exc)
                lowered = error.lower()
                category = ("resource" if any(t in lowered for t in ("out of memory", "timeout", "timed out"))
                            else "configuration" if any(t in lowered for t in ("parameter", "argument", "shape", "column", "window", "dimension"))
                            else "unknown")
                attempt.update(status="execution_failed", error=error, error_type=type(exc).__name__,
                               failure_category=category, elapsed_seconds=perf_counter()-started)
                self._record_attempt(attempt)
                raise
            attempt.update(status="completed", metrics=metrics, elapsed_seconds=perf_counter()-started)
            self._record_attempt(attempt)
        best = state.best_experiment or trial
        result = {
            "experiment_id": trial["experiment_id"], "status": trial["status"],
            "timestamp": trial["timestamp"],
            "progress": f"{trial['trial_number']:02d}/{state.task['max_trials']}",
            "spec_path": trial["spec_path"], "experiment_dir": trial["experiment_dir"],
            "metrics": trial["metrics"], "best_experiment": best,
        }
        if state.recorded_trials >= state.task["max_trials"]:
            result["task_completion"] = self.finish_task(reason="Trial budget exhausted.")
        return result

    def get_history(self) -> dict[str, Any]:
        if self.task_state is None:
            raise ValueError("No active task.")
        history = self.task_state.history
        return {
            "status": "Retrieved the trial history." if history else "No successful trials yet.",
            "trial_history": history,
        }

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        if self.task_state is None:
            raise ValueError("No active task.")
        return {
            "status": f"Retrieved experiment {experiment_id}.",
            "experiment": self.task_state.get_experiment(experiment_id),
        }

    def get_best_experiment(self) -> dict[str, Any]:
        if self.task_state is None:
            raise ValueError("No active task.")
        best = self.task_state.best_experiment
        return {
            "status": "Retrieved the best experiment." if best is not None else "No successful trials yet.",
            "best_experiment": best,
        }

    def _evaluate_spec(self, spec_path: Path, output_dir: Path) -> dict:
        with TemporaryDirectory(dir=self.task_dir) as directory:
            run_dir = Path(directory) / "output"
            result = self._libraryts("run", "--spec", str(spec_path), "--output-dir", str(run_dir))
            metrics = result.get("metrics") if isinstance(result, dict) else None
            if not isinstance(metrics, dict):
                raise HarnessExecutionError("LibraryTS returned no evaluation metrics.")
            artifacts = {
                "model": run_dir / "model" / "model.zip",
                "scores": run_dir / "prediction" / "scores.csv",
                "metrics": run_dir / "evaluation" / "metrics.json",
            }
            for name, source in artifacts.items():
                if not source.is_file():
                    raise HarnessExecutionError(f"LibraryTS returned no {name} artifact: {source}")
            files = {}
            for name, source in artifacts.items():
                target = output_dir / source.name
                copy2(source, target)
                files[name] = target.name
            manifest = {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "files": {"spec": spec_path.name, **files},
            }
            (output_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
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
