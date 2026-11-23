"""Exceptions reported by the harness to the user."""


class HarnessExecutionError(OSError):
    """The harness could not execute a command or read its expected response."""
