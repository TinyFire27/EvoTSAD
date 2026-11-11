"""A configured model using the OpenAI Responses API."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openai import OpenAI
from openai.types.responses import Response

if TYPE_CHECKING:
    from . import LLMConfig


class ResponseModel:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key
        )
        self.model_name = config.model_name

    def request(
        self,
        *,
        messages: list,
        instructions: str,
        tools: list[dict],
        allow_tools: bool,
    ) -> Response:
        option_params = {
            "max_output_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "reasoning": (
                {"effort": self.config.reasoning}
                if self.config.reasoning is not None
                else None
            ),
        }

        option_params = {key: value for key, value in option_params.items() if value is not None}
        
        return self._client.responses.create(
            model=self.model_name,
            instructions=instructions,
            input=messages,
            tools=tools,
            tool_choice="auto" if allow_tools else "none",
            store=False,
            include=["reasoning.encrypted_content"],
            **option_params
        )
