"""Persist conversation history and a session index."""

import fcntl
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

DEFAULT_SESSION_TITLE = "Untitled"


class SessionStore:
    def __init__(
        self,
        root: Path,
        workspace: Path,
        session_id: str | None = None,
        *,
        model_selector: str | None = None,
        title: str | None = None,
    ):
        self.workspace = workspace.expanduser().resolve()
        self.session_dir = root / "sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.session_dir / "index.json"

        if session_id is None:
            self.model_selector = model_selector
            self.title = DEFAULT_SESSION_TITLE if title is None else title
            self.current_id = self.create_session(model_selector, title=self.title)
        else:
            metadata = self._read_index(self.index_path).get(session_id)
            if metadata is None:
                raise FileNotFoundError(f"Unknown session: {session_id}")
            if metadata.get("workspace") != str(self.workspace):
                raise ValueError(f"Session {session_id} is outside the current workspace.")
            self.current_id = session_id
            self.model_selector = metadata.get("model_selector")
            self.title = metadata.get("title", DEFAULT_SESSION_TITLE)

        self.session_path = self.session_dir / f"{self.current_id}.jsonl"

    def create_session(
        self, model_selector: str | None, *, title: str = DEFAULT_SESSION_TITLE
    ) -> str:
        """Create an empty conversation and return its identifier."""
        session_id = uuid4().hex
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._edit_index() as index:
            (self.session_dir / f"{session_id}.jsonl").touch(exist_ok=False)
            index[session_id] = {
                "title": title,
                "workspace": str(self.workspace),
                "model_selector": model_selector,
                "created_at": timestamp,
                "updated_at": timestamp,
            }
        return session_id

    def load_session(self) -> list[dict]:
        """Rebuild Responses API input items from the current conversation."""
        with self.session_path.open(encoding="utf-8") as session_file:
            return [json.loads(line) for line in session_file if line.strip()]

    def write(self, output: list | dict) -> None:
        """Append one item or a list of items to the session."""
        if isinstance(output, dict):
            output = [output]
        if not output:
            return

        # 序列化成 list[dict]
        items = [
            item.model_dump(mode="json", exclude_unset=True)
            if isinstance(item, BaseModel) else item
            for item in output
        ]

        # 落盘
        content = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items)
        with self._edit_index() as index:
            with self.session_path.open("a", encoding="utf-8") as session_file:
                session_file.write(content)
            index[self.current_id]["updated_at"] = (
                datetime.now(timezone.utc).isoformat()
            )

    def set_title(self, title: str) -> None:
        with self._edit_index() as index:
            index[self.current_id]["title"] = title
        self.title = title

    @classmethod
    def list_sessions(cls, root: Path, workspace: Path) -> list[dict]:
        """List a workspace's conversations without opening a session."""
        workspace_path = str(workspace.expanduser().resolve())
        index = cls._read_index(root / "sessions" / "index.json")
        sessions = [
            {"session_id": session_id, **metadata}
            for session_id, metadata in index.items()
            if metadata.get("workspace") == workspace_path
        ]
        return sorted(sessions, key=lambda item: item["updated_at"], reverse=True)

    @staticmethod
    def _read_index(index_path: Path) -> dict:
        if not index_path.exists():
            return {}
        return json.loads(index_path.read_text(encoding="utf-8"))

    @contextmanager
    def _edit_index(self):
        """Serialize index updates and replace the index atomically."""
        with (self.session_dir / ".index.lock").open("a") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            index = self._read_index(self.index_path)
            yield index
            temporary_path = self.session_dir / "index.tmp"
            temporary_path.write_text(
                json.dumps(index, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary_path.replace(self.index_path)
