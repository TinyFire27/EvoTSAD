"""Drive model turns and feed tool results back to the session."""

from copy import deepcopy

from openai import APIError
from openai.types.responses import ResponseInputMessageContentListParam

from ..models import ResponseModel
from ..runtime import SessionStore
from ..tools import ToolExecutor


class EmptyModelResponseError(RuntimeError):
    """A model response ended without a final answer or a tool call."""


class DefaultAgent:
    def __init__(
        self,
        backbone: ResponseModel,
        instructions: str,
        session: SessionStore,
        tools: ToolExecutor,
    ) -> None:
        self.backbone = backbone
        self.instructions = instructions
        self.session = session
        self.tools = tools
        if hasattr(tools, "set_reflection_model"):
            tools.set_reflection_model(backbone)

        self.messages = session.load_session()

    def agent_loop(
        self, user_input: str | ResponseInputMessageContentListParam
    ) -> str:
        """Continue the conversation until the model returns a final reply."""
        if hasattr(self.tools, "note_user_input"):
            text = user_input if isinstance(user_input, str) else "\n".join(
                item["text"] for item in user_input if item.get("type") == "input_text")
            self.tools.note_user_input(text)
        user_item = {"role": "user", "content": deepcopy(user_input)}
        self.session.write(user_item)
        self.messages.append(user_item)

        while True:
            try:
                response = self.backbone.request(
                    messages=self.messages,
                    instructions=self.instructions,
                    tools=self.tools.schemas,
                    allow_tools=True,
                )
            except APIError as exc:
                exc.add_note("DefaultAgent failed while requesting a model response.")
                raise

            # 收集工具调用
            tool_calls = [
                item for item in response.output if item.type == "function_call"
            ]

            final_text = response.output_text if not tool_calls else None
            if not tool_calls and not final_text:
                raise EmptyModelResponseError(
                    f"Model response {response.id} returned no final text."
                )

            items = list(response.output)
            for call in tool_calls:
                result = self.tools.execute(call.name, call.arguments)
                tool_item = {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": result,
                }
                items.append(tool_item)
            self.session.write(items)
            self.messages.extend(items)
            if final_text:
                return final_text
