"""File tools scoped to a task workspace."""

from base64 import b64encode
from pathlib import Path

from openai.types.responses import ResponseFunctionCallOutputItemListParam

READ_LIMIT = 20_000
IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def workspace_path(cwd: Path, path: str) -> Path:
    resolved = (cwd / path).resolve()
    if resolved == cwd or not resolved.is_relative_to(cwd):
        raise ValueError("Use a file path inside the workspace.")
    return resolved


def read_file(cwd: Path, path: str) -> dict:
    source = workspace_path(cwd, path)
    if not source.is_file():
        raise ValueError("The requested path must be a file.")
    with source.open(encoding="utf-8") as handle:
        content = handle.read(READ_LIMIT + 1)
    return {
        "path": str(source.relative_to(cwd)),
        "content": content[:READ_LIMIT],
        "truncated": len(content) > READ_LIMIT,
    }


def image_data_url(cwd: Path, path: str) -> str:
    """Read a workspace image and encode it as a Base64 data URL."""
    source = workspace_path(cwd, path)
    if not source.is_file():
        raise ValueError("The requested path must be a file.")
    mime_type = IMAGE_MIME_TYPES.get(source.suffix.lower())
    if mime_type is None:
        raise ValueError("Use a PNG, JPEG, or WebP image.")

    encoded = b64encode(source.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def read_image(cwd: Path, path: str) -> ResponseFunctionCallOutputItemListParam:
    """Return a workspace image as a model input content block."""
    return [{"type": "input_image", "image_url": image_data_url(cwd, path)}]


def write_file(cwd: Path, path: str, content: str) -> dict:
    target = workspace_path(cwd, path)
    relative = target.relative_to(cwd)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        handle.write(content)
    return {"path": str(relative)}
