"""Render the REPL interface in the terminal."""

import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.filters import to_filter
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.markdown import Markdown, TableElement
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


_APP_NAME = "EvoTS-AD"
_APP_VERSION = f"{_APP_NAME} v0.1.0 (2026.11.11)"
_TAGLINE = "Your local agent for developing time-series anomaly detection algorithms"
_GOLD = "bold #ffda24"
_INFO = "#9ca3af"
_COMMAND = "bold #e5e7eb"
_ERROR = "#ff9b8f"
_LOGO = (
    ("███████╗ ██╗   ██╗  ██████╗  ████████╗ ███████╗         █████╗  ██████╗", "#ffdf19"),
    ("██╔════╝ ██║   ██║ ██╔═══██╗ ╚══██╔══╝ ██╔════╝        ██╔══██╗ ██╔══██╗", "#ffd51b"),
    ("█████╗   ██║   ██║ ██║   ██║    ██║    ███████╗ █████╗ ███████║ ██║  ██║", "#ffc414"),
    ("██╔══╝   ╚██╗ ██╔╝ ██║   ██║    ██║    ╚════██║ ╚════╝ ██╔══██║ ██║  ██║", "#f3a91c"),
    ("███████╗  ╚████╔╝  ╚██████╔╝    ██║    ███████║        ██║  ██║ ██████╔╝", "#dc932b"),
    ("╚══════╝   ╚═══╝    ╚═════╝     ╚═╝    ╚══════╝        ╚═╝  ╚═╝ ╚═════╝", "#b8742d"),
)


class ChatTerminal:
    """Render the chat and manage its interactive input."""

    def __init__(self) -> None:
        console = Console()
        self._console = console
        self._editor: PromptSession[str] | None = None
        if console.is_terminal and not console.is_dumb_terminal and sys.stdin.isatty():
            self._editor = PromptSession(
                erase_when_done=True,
                style=Style.from_dict({
                    "status": "fg:#e7c775",
                    "rule": "fg:#d58c32",
                    "speaker": "bold ansicyan",
                    "hint": "italic fg:#aa966d",
                    "bottom-toolbar": "noreverse",
                }),
            )
            # The editor should grow only when the input wraps.
            self._editor.app.layout.current_window.dont_extend_height = to_filter(True)

    def print_welcome(
        self, *, default_model: str | None = None, workspace: Path | None = None
    ) -> None:
        console = self._console
        environment = Group(
            Text("Environment", style=_GOLD),
            Text.assemble(("Model       ", "dim"), default_model or "From TOML"),
            Text("Workspace", style="dim"),
            Text(str(workspace or Path.cwd()), overflow="fold"),
        )
        commands = Group(
            Text("REPL commands", style=_GOLD),
            Text("/new [--title TITLE] [--model PROVIDER/MODEL]", style="dim"),
            Text("/sessions  ·  /switch <id>  ·  /image <path>  ·  /exit", style="dim"),
        )
        heading = f"{_APP_VERSION} · {_TAGLINE}"
        body = Group(environment, Text(), commands)
        if not console.is_terminal:
            console.print(Text(heading, style="dim"), soft_wrap=True)
            console.print(body)
            return

        if console.width >= 96:
            columns = Table.grid(padding=(0, 2), expand=True)
            columns.add_column(ratio=1, overflow="fold")
            columns.add_column(ratio=2, overflow="fold")
            columns.add_row(environment, commands)
            body = columns

        if console.width < len(heading) + 6:
            heading = _APP_VERSION
            subtitle = _TAGLINE
            if console.width < len(heading) + 6:
                heading = _APP_NAME
                subtitle = f"{_APP_VERSION}\n{_TAGLINE}"
            body = Group(Text(subtitle, style=_GOLD), Text(), body)

        logo = Text("EVOTS-AD", style=_GOLD)
        if console.width >= max(len(line) for line, _ in _LOGO) and not console.options.ascii_only:
            logo = Text("\n").join(Text(line, style=color) for line, color in _LOGO)

        console.print()
        console.print(logo)
        console.print()
        console.print(
            Panel(
                body,
                title=Text(heading, style=_GOLD),
                border_style="#d58c32",
                padding=(1, 2),
            )
        )

    def print_info(self, text: str) -> None:
        result = Text(text, style=_INFO, overflow="fold")
        self._console.print(Padding(result, (0, 0, 0, 2), expand=False))

    def echo_repl(self, text: str) -> None:
        self._console.print(Text(f"\n⚙ {text}", style=_COMMAND), overflow="fold")

    def show_sessions(self, sessions: list[dict]) -> None:
        lines = [f'{entry["session_id"]}  {entry["title"]}' for entry in sessions]
        self.print_info("\n".join(lines) if lines else "No sessions in this workspace.")

    def read_input(
        self,
        *,
        model_selector: str | None,
        thinking: str | None,
        title: str | None,
        session_id: str | None,
        caption: bool = False,
    ) -> str:
        label = "Text (optional)> " if caption else "You > "
        editor = self._editor
        if editor is None:
            return self._console.input(Text(label, style="dim" if caption else "bold cyan"))

        status = Text(
            f"{model_selector or 'Default model'}  |  thinking {thinking or 'default'}  |  "
            f"{title or 'No session'}  |  {session_id or '—'}"
        )

        def width() -> int:
            return max(1, editor.app.output.get_size().columns - 1)

        def message() -> FormattedText:
            columns = width()
            status_lines = status.wrap(self._console, columns, overflow="fold")
            status_text = "\n".join(line.plain for line in status_lines)
            return FormattedText([
                ("", "\n"),
                ("class:status", status_text + "\n"),
                ("class:rule", "─" * columns + "\n"),
                ("class:hint" if caption else "class:speaker", label),
            ])

        return editor.prompt(
            message,
            bottom_toolbar=lambda: FormattedText([("class:rule", "─" * width())]),
            placeholder=FormattedText([
                ("class:hint", "Press Enter to skip." if caption else "Type a message or a REPL command…")
            ]),
        )

    def echo_user(self, text: str) -> None:
        """Print a submitted user message in the terminal history."""
        if self._editor is not None:
            self._console.print(Text.assemble("\n", ("You > ", "bold cyan"), text), overflow="fold")

    def echo_assistant(self, text: str) -> None:
        console = self._console
        if not console.is_terminal:
            print(text, file=console.file)
            return

        label = Text("Assistant > ", style="bold #c69bfa")
        reply = Table.grid(padding=0, expand=True)
        reply.add_column(width=label.cell_len, no_wrap=True)
        reply.add_column(ratio=1, overflow="fold")
        reply.add_row(label, _ReplyMarkdown(text, code_theme="ansi_dark"))

        console.print()
        console.print(Padding(reply, (0, 1, 0, 0), style="#e5e7eb on #253044"))

    def print_error(self, text: str) -> None:
        console = Console(stderr=True)
        if not console.is_terminal:
            print(text, file=sys.stderr)
            return

        marker = Text("⚠️  ", style=f"bold {_ERROR}")
        error = Table.grid(padding=0, expand=True)
        error.add_column(width=marker.cell_len, no_wrap=True)
        error.add_column(ratio=1, style=_ERROR, overflow="fold")
        error.add_row(marker, Text(text))
        console.print(Padding(error, (1, 0, 0, 0), expand=False))


class _WrappingTableElement(TableElement):
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        for renderable in super().__rich_console__(console, options):
            if isinstance(renderable, Table):
                for column in renderable.columns:
                    column.overflow = "fold"
            yield renderable


class _ReplyMarkdown(Markdown):
    elements = {**Markdown.elements, "table_open": _WrappingTableElement}
