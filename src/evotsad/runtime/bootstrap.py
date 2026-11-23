"""Build system instructions for an EvoTSAD session."""

from pathlib import Path
from typing import Literal

from .skills import SkillsManager

PROMPTS_NAME = ("IDENTITY.md", "TOOLS.md", "EXPERIMENT.md")
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

def build_instructions(workspace: Path) -> str:
    """Combine the mode's instructions with the current workspace's skill catalog."""

    sections = [
        (PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()
        for filename in PROMPTS_NAME
    ]

    skills = SkillsManager(workspace)
    skills.discover()
    if catalog := skills.format_prompt_block():
        sections.append(catalog)
    return "\n\n".join(sections)
