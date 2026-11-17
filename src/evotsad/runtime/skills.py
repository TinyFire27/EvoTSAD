"""Discover workspace skills and let the model read them on demand."""

import re
from pathlib import Path


class SkillsManager:
    """Index skills under the workspace and read their instructions on demand."""

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir.expanduser().resolve()
        self._root = self.workspace_dir / ".agent" / "skills"
        self.skills: dict[str, dict[str, str]] = {}

    def discover(self) -> None:
        """Refresh the index using metadata from the workspace's skill files."""
        self.skills.clear()
        if self._root.parent.is_symlink() or self._root.is_symlink():
            return

        for path in sorted(self._root.glob("*/SKILL.md")):
            if path.parent.is_symlink() or path.is_symlink() or not path.is_file():
                continue
            try:
                metadata = _read_metadata(path)
            except (OSError, UnicodeError):
                continue
            if metadata.get("name") and metadata.get("description"):
                metadata["path"] = path.relative_to(self.workspace_dir).as_posix()
                self.skills.setdefault(metadata["name"], metadata)

    def format_prompt_block(self) -> str:
        """Render the skill catalog and tell the model how to read an entry."""
        if not self.skills:
            return ""

        entries = [
            f"- {skill['name']}: {skill['description']}\n  Path: `{skill['path']}`"
            for skill in self.skills.values()
        ]
        return (
            "## Available Skills\n\n"
            "The skills below are available in the current workspace. "
            "When a skill's description matches the current task, activate it by using "
            "`read_file` to read its `SKILL.md` at the listed path, then follow its instructions. "
            "Resolve relative reference paths from the directory containing `SKILL.md`.\n\n"
            + "\n".join(entries)
        )

    def load(self, name: str) -> str:
        """Read the current SKILL.md text for a discovered skill."""
        skill = self.skills.get(name)
        if skill is None:
            raise ValueError(f"Unknown workspace skill: {name}")

        path = self.workspace_dir / skill["path"]
        if path.resolve() != path or not path.is_relative_to(self._root):
            raise ValueError(f"Skill is outside the workspace skills directory: {name}")
        return path.read_text(encoding="utf-8-sig")


def _read_metadata(path: Path) -> dict[str, str]:
    """Read the name and description from simple YAML frontmatter."""
    with path.open(encoding="utf-8-sig") as skill_file:
        if skill_file.readline().rstrip() != "---":
            return {}
        lines = []
        for line in skill_file:
            if line.rstrip() == "---":
                break
            lines.append(line)
        else:
            return {}

    metadata = {}
    fields = re.findall(
        r"^(name|description):[ \t]*([^\n]*)(\n(?:[ \t]+[^\n]*\n)*)?",
        "".join(lines),
        flags=re.MULTILINE,
    )
    for key, value, continuation in fields:
        value = value.strip()
        if key == "description" and continuation.strip():
            if value in {">", ">-", ">+", "|", "|-", "|+"}:
                value = ""
            value = " ".join((value + continuation).split())
        if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
            value = value[1:-1].strip()
        metadata[key] = value
    return metadata
