"""Terminal entry point for EvoTSAD."""

import shlex
from argparse import ArgumentError, ArgumentParser
from pathlib import Path

from dotenv import load_dotenv
from openai import APIError
from openai.types.responses import ResponseInputMessageContentListParam

from .agent.core import DefaultAgent, EmptyModelResponseError
from .models import ResponseModel
from .models.router import resolve_model_config
from .runtime import SessionStore
from .tools import ToolExecutor
from .tools.files import image_data_url
from .utils.terminal import ChatTerminal

ROOT_PATH = Path(__file__).resolve().parents[2] / ".agent"
MODEL_CONFIG_PATH = ROOT_PATH / "config.toml"
DEFAULT_INSTRUCTIONS = (
    "You are EvoTSAD, an assistant for time-series anomaly detection. "
    "Use the available workspace tools as needed."
)


def image_message(image_url: str, caption: str) -> ResponseInputMessageContentListParam:
    content: ResponseInputMessageContentListParam = [
        {"type": "input_image", "image_url": image_url, "detail": "auto"}
    ]
    if caption.strip():
        content.insert(0, {"type": "input_text", "text": caption})
    return content


def parse_new_args(arguments: str) -> tuple[str | None, str | None]:
    parser = ArgumentParser(
        prog="/new",
        add_help=False,
        allow_abbrev=False,
        exit_on_error=False,
    )
    parser.add_argument("--title")
    parser.add_argument("--model")

    try:
        options, extra = parser.parse_known_args(shlex.split(arguments))
    except ArgumentError as exc:
        raise ValueError(str(exc)) from exc

    if extra:
        raise ValueError(f"Unrecognized /new arguments: {' '.join(extra)}")

    title = options.title
    if title is not None:
        title = title.strip()
        if not title:
            raise ValueError("Session title cannot be empty.")

    return title, options.model


def main() -> int:
    load_dotenv(ROOT_PATH / ".env")
    workspace = Path.cwd().resolve()
    tools = ToolExecutor(workspace=workspace)
    terminal = ChatTerminal()
    instructions = DEFAULT_INSTRUCTIONS  # 后续改成系统提示词加载器
    agent: DefaultAgent | None = None

    # 默认配置
    default_model = None
    default_thinking = None
    default_model_error = None
    try:
        default_config = resolve_model_config(config_path=MODEL_CONFIG_PATH)
    except (OSError, ValueError) as exc:
        default_config = None
        default_model_error = str(exc)
    else:
        default_model = default_config.model_selector
        default_thinking = default_config.reasoning

    terminal.print_welcome(default_model=default_model, workspace=workspace)
    if default_model_error is not None:
        terminal.print_error(f"Configure the default model in config.toml. {default_model_error}")

    # Chat 循环
    while True:
        try:
            if agent is None:
                prompt_status = dict(
                    model_selector=default_model,
                    thinking=default_thinking,
                    title=None,
                    session_id=None,
                )
            else:
                active_config = agent.backbone.config
                prompt_status = dict(
                    model_selector=active_config.model_selector,
                    thinking=active_config.reasoning,
                    title=agent.session.title,
                    session_id=agent.session.current_id,
                )

            # 输入框，等待 enter，enter 后暂时消失
            user_input = terminal.read_input(**prompt_status).strip()
            if not user_input:
                continue

            message: str | ResponseInputMessageContentListParam = user_input

            if user_input.startswith("/"):
                terminal.echo_repl(user_input)
                command, *args = user_input.split(maxsplit=1)

                if command == "/exit":
                    terminal.print_info("Exiting.")
                    break

                if command == "/image":
                    if not args:
                        raise ValueError("Use /image <path>.")
                    image_path = args[0]
                    image_url = image_data_url(workspace, image_path)
                    terminal.print_info("Image attached.")
                    caption = terminal.read_input(**prompt_status, caption=True)
                    message = image_message(image_url, caption)
                    terminal.echo_user(f"🖼  {image_path}  {caption.strip()}".rstrip())

                elif command == "/new":
                    title, model_selector = parse_new_args(args[0] if args else "")
                    config = resolve_model_config(model_selector, config_path=MODEL_CONFIG_PATH)
                    backbone = ResponseModel(config)
                    session = SessionStore(
                        ROOT_PATH, workspace, model_selector=config.model_selector, title=title
                    )
                    agent = DefaultAgent(
                        backbone=backbone, instructions=instructions, session=session, tools=tools
                    )
                    terminal.print_info(
                        f"Created session {session.title} ({session.current_id})."
                    )
                    continue

                elif command == "/switch":
                    if not args:
                        raise ValueError("Use /switch <session-id>.")
                    session = SessionStore(ROOT_PATH, workspace, session_id=args[0])
                    config = resolve_model_config(session.model_selector, config_path=MODEL_CONFIG_PATH)
                    backbone = ResponseModel(config)
                    agent = DefaultAgent(
                        backbone=backbone, instructions=instructions, session=session, tools=tools
                    )
                    terminal.print_info(
                        f"Switched to session {session.title} ({session.current_id})."
                    )
                    continue

                elif command == "/sessions":
                    if args:
                        raise ValueError("Use /sessions without arguments.")
                    sessions = SessionStore.list_sessions(ROOT_PATH, workspace)
                    terminal.show_sessions(sessions)
                    continue
            else:
                terminal.echo_user(user_input)

            if agent is None:
                config = default_config or resolve_model_config(config_path=MODEL_CONFIG_PATH)
                backbone = ResponseModel(config)
                session = SessionStore(
                    ROOT_PATH, workspace, model_selector=config.model_selector
                )
                agent = DefaultAgent(
                    backbone=backbone, instructions=instructions, session=session, tools=tools
                )

            output_text = agent.agent_loop(message)
            terminal.echo_assistant(output_text)
        except (KeyboardInterrupt, EOFError):  # 主动打断或主输入和图片说明输入遇到 EOF
            break
        except (OSError, ValueError) as exc:  # 缺参数、路径错误、模型配置有问题、会话 ID 不存在、文件系统错误
            terminal.print_error(str(exc))
        except (APIError, EmptyModelResponseError) as exc:
            terminal.print_error(f"Model request failed. {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
