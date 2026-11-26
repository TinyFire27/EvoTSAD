"""Run chat-equivalent sessions from a JSONL file."""

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .agent.core import DefaultAgent
from .chat import MODEL_CONFIG_PATH, ROOT_PATH
from .evolution.memory import MEMORY_MODES
from .models import ResponseModel
from .models.router import resolve_model_config
from .runtime import SessionStore
from .runtime.bootstrap import build_instructions
from .tools.experiments import ExperimentToolExecutor


def read_sessions(path: Path) -> list[dict[str, Any]]:
    """Validate the full input before creating any model sessions."""
    sessions = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Line {line_number}: expected a session object, not a blank line.")
            try:
                session = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_number}: invalid JSON: {exc.msg}.") from exc
            if not isinstance(session, dict):
                raise ValueError(f"Line {line_number}: session must be an object.")
            if session.get("memory_mode", "all") not in MEMORY_MODES:
                raise ValueError(f"Line {line_number}: unknown memory_mode.")
            if type(session.get("update_memory", True)) is not bool:
                raise ValueError(f"Line {line_number}: update_memory must be a boolean.")
            for key in ("model", "title"):
                if key in session and (not isinstance(session[key], str) or not session[key].strip()):
                    raise ValueError(f"Line {line_number}: {key} must be nonempty text.")
            tasks = session.get("tasks")
            if not isinstance(tasks, list) or not tasks:
                raise ValueError(f"Line {line_number}: tasks must be a nonempty array.")
            for index, task in enumerate(tasks, 1):
                if not isinstance(task, dict) or not isinstance(task.get("prompt"), str) or not task["prompt"].strip():
                    raise ValueError(f"Line {line_number}, task {index}: prompt must be nonempty text.")
                if task["prompt"].strip().split(maxsplit=1)[0] in {
                    "/new", "/switch", "/sessions", "/prompt", "/image", "/exit"
                }:
                    raise ValueError(f"Line {line_number}, task {index}: terminal commands are not task prompts.")
            sessions.append(session)
    if not sessions:
        raise ValueError("Input JSONL contains no sessions.")
    return sessions


def run_batch(input_path: Path, output_path: Path, *, workspace: Path | None = None) -> int:
    workspace = (workspace or Path.cwd()).resolve(strict=True)
    input_path = input_path.resolve(strict=True)
    output_path = output_path.resolve()
    if input_path == output_path:
        raise ValueError("Input and output paths must differ.")
    sessions = read_sessions(input_path)
    load_dotenv(ROOT_PATH / ".env")
    instructions = build_instructions(workspace)
    # Resolve configurations before creating the output or any sessions.
    configs = [resolve_model_config(item.get("model"), config_path=MODEL_CONFIG_PATH) for item in sessions]

    with output_path.open("x", encoding="utf-8", newline="\n") as output:
        for number, (item, config) in enumerate(zip(sessions, configs), 1):
            tools = ExperimentToolExecutor(
                workspace=workspace,
                memory_mode=item.get("memory_mode", "all"),
                update_memory=item.get("update_memory", True),
            )
            session = SessionStore(
                ROOT_PATH, workspace, model_selector=config.model_selector, title=item.get("title")
            )
            agent = DefaultAgent(
                backbone=ResponseModel(config), instructions=instructions, session=session, tools=tools
            )
            record = {**item, "tasks": []}
            for index, task in enumerate(item["tasks"], 1):
                try:
                    response = agent.agent_loop(task["prompt"].strip())
                except Exception as exc:
                    record["tasks"].append({**task, "error": f"{type(exc).__name__}: {exc}"})
                    output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    output.flush()
                    raise RuntimeError(f"Session {number}, task {index} failed: {exc}") from exc
                record["tasks"].append({**task, "response": response})
            output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            print(f"Completed session {number}/{len(sessions)}", file=sys.stderr)
    return len(sessions)


def main() -> int:
    parser = ArgumentParser(description="Run EvoTSAD chat sessions from a JSONL file")
    parser.add_argument("input", type=Path, help="Input JSONL with one session per line")
    parser.add_argument("output", type=Path, help="New output JSONL file")
    args = parser.parse_args()
    try:
        run_batch(args.input, args.output)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Batch failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
