"""Tool schemas and dispatch for an EvoTSAD task."""

import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from copy import deepcopy

from openai.types.responses import ResponseFunctionCallOutputItemListParam

from .bash import run_bash
from .files import read_file, read_image, write_file


def function_tool(name: str, description: str, **properties: str) -> dict:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                name: {"type": "string", "description": description}
                for name, description in properties.items()
            },
            "required": list(properties),
            "additionalProperties": False,
        },
    }


TOOLS = [
    function_tool(
        "bash",
        "Execute Bash in the workspace and return stdout, stderr and the exit code.",
        command="A Bash command or script.",
    ),
    function_tool(
        "read_file",
        "Read a UTF-8 text file in the current workspace.",
        path="A path in the workspace.",
    ),
    function_tool(
        "read_image",
        "Read a PNG, JPEG, or WebP image in the current workspace for visual inspection.",
        path="An image file path inside the workspace.",
    ),
    function_tool(
        "write_file",
        "Create a UTF-8 text file in the current workspace. Use a new path for a revision.",
        path="A new path in the workspace.",
        content="Text to write.",
    ),
]


class ToolExecutor:
    """Bind the agent's tools to a workspace and dispatch model calls."""

    def __init__(self, workspace: Path, *, timeout: int = 600) -> None:
        self.workspace = workspace.expanduser().resolve(strict=True)
        if not self.workspace.is_dir():
            raise NotADirectoryError(self.workspace)

        self.schemas = TOOLS
        self._handlers = {
            "bash": partial(run_bash, cwd=workspace, timeout=timeout),
            "read_file": partial(read_file, cwd=self.workspace),
            "read_image": partial(read_image, cwd=self.workspace),
            "write_file": partial(write_file, cwd=self.workspace),
        }

    def execute(
        self, name: str, arguments: str
    ) -> str | ResponseFunctionCallOutputItemListParam:
        """Execute a tool with the model's JSON-encoded arguments."""
        handler = self._handlers.get(name)
        if handler is None:
            result = {"error": f"Unknown tool: {name}"}
        else:
            try:
                result = handler(**json.loads(arguments))
            except Exception as exc:
                result = {"error": str(exc)}

        if isinstance(result, list):
            return deepcopy(result)
        elif isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False)
        else:
            raise TypeError(f"Unsupported tool result type: {type(result).__name__}")
