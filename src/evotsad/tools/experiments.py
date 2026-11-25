"""Tools available to the EvoTSAD experiment agent."""

import csv
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
from . import ToolExecutor, function_tool
from .bash import run_bash


BASELINE_DETECTORS = ("SubLOF", "SubOCSVM", "SubPCA", "IForest", "HBOS")


TASK_INIT_TOOL = {
    "type": "function",
    "name": "task_init",
    "description": "Initialize a task, replacing its previous outputs. Evaluate five fixed detectors with default parameters and return their metrics separately from the trial budget.",
    "strict": False,
    "parameters": {
        "type": "object",
        "properties": {
            "objective": {"type": "string", "description": "The task goal."},
            "data": {
                "type": "object",
                "properties": {
                    "train": {"type": "string", "description": "Training CSV with a value column. Optional columns: timestamp, label."},
                    "val": {
                        "type": ["string", "null"],
                        "description": "Optional early-stopping CSV with the same column rules as train.",
                    },
                    "eval": {
                        "type": "string",
                        "description": "Evaluation CSV with value and label columns, and an optional timestamp.",
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

GET_HISTORY_TOOL = function_tool(
    "get_history", "Return the active task's successful trial records in trial_history, in recorded order.",
)
GET_EXPERIMENT_TOOL = function_tool(
    "get_experiment", "Return one full record from the active task in experiment, by its exact ID.",
    experiment_id="Exact experiment ID, for example exp-001.",
)
GET_BEST_EXPERIMENT_TOOL = function_tool(
    "get_best_experiment",
    "Return the active task's highest-scoring record in best_experiment. The field is null before the first successful trial.",
)


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
        self.baselines_path = self.task_dir / "baselines.json"
        self.profile_path = self.task_dir / "profile.json"
        self.history_path = self.task_dir / "history.jsonl"
        self.experiments_dir = self.task_dir / "experiments"

        self.task = json.loads(self.task_path.read_text(encoding="utf-8"))
        self.data = self.task["data"]
        self.primary_metric: str = self.task["primary_metric"]

    @classmethod
    def create(cls, task_dir: Path, task: dict[str, Any]) -> "TaskState":
        """Create a task directory and save the agreed settings once."""
        content = json.dumps(task, ensure_ascii=False, indent=2, allow_nan=False)
        task_dir = task_dir.expanduser()
        task_dir.mkdir(parents=True)
        (task_dir / "experiments").mkdir()
        (task_dir / "history.jsonl").touch()
        (task_dir / "task.json").write_text(content + "\n", encoding="utf-8")
        return cls(task_dir)

    def experiment_spec(self, pipeline: list[dict]) -> dict:
        """Build the LibraryTS spec shared by baselines and budgeted trials."""
        return {
            "task": "anomaly_detection",
            "data": {
                "train": self.data["train"],
                "val": self.data.get("val"),
                "test": self.data["eval"],
            },
            "estimator": pipeline,
            "metrics": self.task["metrics"],
            "seed": self.task["seed"],
        }

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
        self.schemas = [
            *self.schemas, TASK_INIT_TOOL, LIST_COMPONENTS_TOOL, RUN_TRIAL_TOOL,
            GET_HISTORY_TOOL, GET_EXPERIMENT_TOOL, GET_BEST_EXPERIMENT_TOOL,
        ]
        self._handlers["task_init"] = self.task_init
        self._handlers["list_components"] = self.list_components
        self._handlers["run_trial"] = self.run_trial
        self._handlers["get_history"] = self.get_history
        self._handlers["get_experiment"] = self.get_experiment
        self._handlers["get_best_experiment"] = self.get_best_experiment

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

        # 数据格式检查
        paths = {}
        for split in ("train", "val", "eval"):
            # 1. 解析 CSV 路径。train 和 eval 必填，val 可省略。
            raw_path = data.get(split)
            if split == "val" and raw_path is None:
                paths[split] = None
                continue
            if not raw_path:
                raise ValueError(f"data.{split} must be a CSV path.")
            path = (self.workspace / Path(raw_path).expanduser()).resolve()
            if path.suffix.lower() != ".csv":
                raise ValueError(f"data.{split} must be a .csv file: {path}")
            paths[split] = str(path)

            # 2. 核对表头。每段都需要 value，只有 eval 强制要求 label。
            with path.open(encoding="utf-8-sig", newline="") as stream:
                columns = next(csv.reader(stream), [])
            expected = ["value"]
            if split == "eval" or "label" in columns:
                expected.append("label")
            if "timestamp" in columns:
                expected.append("timestamp")
            if sorted(columns) != sorted(expected):
                raise ValueError(f"data.{split} expects columns {expected}, got {columns}: {path}")

        # 3. 确认训练和评估使用不同文件。
        if paths["train"] == paths["eval"]:
            raise ValueError("data.train and data.eval must be different files.")

        metric = "VUS-PR" if primary_metric is None else primary_metric.strip()
        if not metric:
            raise ValueError("task.primary_metric must name the selection metric.")
        
        task = {
            "objective": objective.strip(),
            "data": paths,
            "max_trials": max_trials,
            "primary_metric": metric,
            "metrics": {metric: {}},
            "seed": 2026 if seed is None else seed,
        }

        # 初始化任务目录和任务状态
        self.task_state = None
        if self.task_dir.exists():
            rmtree(self.task_dir)
        state = TaskState.create(self.task_dir, task)

        # 评测 baselines
        baselines = self._baselines(state)
        n_success = sum(result["error"] is None for result in baselines.values())
        state.baselines_path.write_text(
            json.dumps(baselines, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        self.task_state = state
        return {
            "status": (
                f"Task initialized. Automatically evaluated {len(baselines)} baseline detectors "
                f"using their default parameters ({n_success} succeeded)."
            ),
            "task_dir": str(state.task_dir),
            "task_path": str(state.task_path),
            "baselines_path": str(state.baselines_path),
            "history_path": str(state.history_path),
            "experiments_dir": str(state.experiments_dir),
            "task": state.task,
            "baselines": baselines,
        }

    def _baselines(self, state: TaskState) -> dict:
        """Keep reference scores only; a detector failure is reported by name."""
        results = {}
        for name in BASELINE_DETECTORS:
            spec = state.experiment_spec([{"class": name}])
            with TemporaryDirectory(dir=state.task_dir) as directory:
                work_dir = Path(directory)
                spec_path = work_dir / "spec.json"
                spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
                try:
                    result = self._libraryts(
                        "run", "--spec", str(spec_path), "--output-dir", str(work_dir / "output")
                    )
                except RuntimeError as exc:
                    results[name] = {"params": "default", "metrics": None, "error": str(exc)}
                else:
                    results[name] = {"params": "default","metrics": result["metrics"], "error": None}
        return results

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
            raise ValueError("No active task. Call task_init before running a trial.")
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
