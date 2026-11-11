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

        self.messages = session.load_session()

    def agent_loop(
        self, user_input: str | ResponseInputMessageContentListParam
    ) -> str:
        """Continue the conversation until the model returns a final reply."""
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

            tool_calls = [
                item for item in response.output if item.type == "function_call"
            ]
            final_text = response.output_text if not tool_calls else None
            if not tool_calls and not final_text:
                raise EmptyModelResponseError(
                    f"Model response {response.id} returned no final text."
                )

            self.session.write(response.output)
            self.messages.extend(response.output)
            if final_text:
                return final_text

            for call in tool_calls:
                result = self.tools.execute(call.name, call.arguments)
                tool_item = {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": result,
                }
                self.session.write(tool_item)
                self.messages.append(tool_item)
