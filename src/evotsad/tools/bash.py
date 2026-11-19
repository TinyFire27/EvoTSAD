"""Execute a Bash command and capture its result."""

import os
import signal
import subprocess
from contextlib import suppress
from pathlib import Path

from ..runtime.sandbox import CommandSandbox


def run_bash(command: str, *, cwd: Path, timeout: int) -> dict:
    sandbox = CommandSandbox(workspace=cwd)
    environment = os.environ.copy()
    environment.update(sandbox.environment)
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    with subprocess.Popen(
        sandbox.wrap(command),
        cwd=sandbox.workspace,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        start_new_session=os.name != "nt",
        creationflags=creationflags,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            with suppress(ProcessLookupError):
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
            if isinstance(exc, KeyboardInterrupt):
                raise
            stdout, stderr = process.communicate()
            return {
                "exit_code": 124,
                "stdout": stdout,
                "stderr": f"{stderr}\nCommand timed out after {timeout}s.",
            }

    return {"exit_code": process.returncode, "stdout": stdout, "stderr": stderr}
