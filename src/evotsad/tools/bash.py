"""Execute a Bash command and capture its result."""

import os
import signal
import subprocess
from contextlib import suppress
from pathlib import Path

from ..runtime.sandbox import MacOSSandbox


def run_bash(command: str, *, cwd: Path, timeout: int) -> dict:
    sandbox = MacOSSandbox(workspace=cwd)
    with subprocess.Popen(
        sandbox.wrap(command),
        cwd=sandbox.workspace,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            with suppress(ProcessLookupError):
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
