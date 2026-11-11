"""Build a macOS filesystem sandbox for command execution."""

import json
from pathlib import Path


class MacOSSandbox:
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
        """Run commands with caches and temporary files inside the workspace."""
        cache_dir = self.workspace / ".sandbox"
        prepare = (
            '/bin/mkdir -p "$TMPDIR" "$MPLCONFIGDIR" '
            '"$NUMBA_CACHE_DIR" "$XDG_CACHE_HOME" || exit\n'
        )
        return [
            "/usr/bin/sandbox-exec",
            "-p",
            self.profile,
            "/usr/bin/env",
            f"TMPDIR={cache_dir / 'tmp'}",
            f"MPLCONFIGDIR={cache_dir / 'matplotlib'}",
            f"NUMBA_CACHE_DIR={cache_dir / 'numba'}",
            f"XDG_CACHE_HOME={cache_dir / 'cache'}",
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            prepare + command,
        ]
