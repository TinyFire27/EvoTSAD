"""Prepare a cross-platform command environment."""

import json
import os
import shutil
from pathlib import Path


class CommandSandbox:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.expanduser().resolve(strict=True)
        workspace_path = json.dumps(str(self.workspace), ensure_ascii=False)
        self.profile = "\n".join(
            [
                "(version 1)",
                "(deny default)",
                "(allow process-exec process-fork)",
                "(allow process-info* (target same-sandbox))",
                "(allow signal (target same-sandbox))",
                "(allow sysctl-read)",
                # Allow MPS to open the GPU and compile Metal kernels.
                "(allow iokit-open "
                '(iokit-user-client-class "AGXDeviceUserClient") '
                '(iokit-registry-entry-class "IOSurfaceRootUserClient"))',
                "(allow iokit-get-properties)",
                '(allow mach-lookup (global-name "com.apple.MTLCompilerService"))',
                "(allow ipc-posix-shm ipc-posix-sem)",
                "(allow network*)",
                "(allow file-read*)",
                f'(allow file-write* (literal "/dev/null") (subpath {workspace_path}))',
            ]
        )

    def wrap(self, command: str) -> list[str]:
        """Build a command line using the strongest available shell isolation."""
        cache_dir = self.workspace / ".sandbox"
        for name in ("tmp", "matplotlib", "numba", "cache"):
            (cache_dir / name).mkdir(parents=True, exist_ok=True)

        bash = shutil.which("bash") if os.name != "nt" else None
        sandbox_exec = shutil.which("sandbox-exec") if os.name == "posix" else None
        if sandbox_exec:
            return [sandbox_exec, "-p", self.profile, bash or "/bin/sh", "-c", command]
        if bash:
            return [bash, "--noprofile", "--norc", "-c", command]
        if os.name == "nt":
            return [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/s", "/c", command]
        return ["/bin/sh", "-c", command]

    @property
    def environment(self) -> dict[str, str]:
        cache_dir = self.workspace / ".sandbox"
        return {
            "TMPDIR": str(cache_dir / "tmp"),
            "MPLCONFIGDIR": str(cache_dir / "matplotlib"),
            "NUMBA_CACHE_DIR": str(cache_dir / "numba"),
            "XDG_CACHE_HOME": str(cache_dir / "cache"),
        }
