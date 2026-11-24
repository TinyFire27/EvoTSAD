"""A small, workspace-local store of reusable experiment experiences."""

import json
from pathlib import Path

from ..runtime.file_lock import locked

MEMORY_MODES = ("all", "improvements", "none")


class ExperienceMemory:
    """Append successful trials and retrieve recent experiences by keyword."""

    def __init__(self, workspace: Path, mode: str = "all") -> None:
        if mode not in MEMORY_MODES:
            raise ValueError(f"Unknown memory mode: {mode!r}")
        self.mode = mode
        filename = "experience.jsonl" if mode == "all" else f"experience-{mode}.jsonl"
        self.path = workspace / ".agent" / filename

    def remember(self, experience: dict) -> None:
        if self.mode == "none":
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as stream:
            with locked(stream):
                stream.write(json.dumps(experience, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()

    def search(self, query: str = "", limit: int = 5) -> list[dict]:
        if type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("limit must be an integer between 1 and 50.")
        if self.mode == "none":
            return []
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as stream:
            with locked(stream, shared=True):
                records = [json.loads(line) for line in stream if line.strip()]
        query = query.strip().casefold()
        return [record for record in reversed(records)
                if query in json.dumps(record, ensure_ascii=False).casefold()][:limit]
