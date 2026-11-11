from dataclasses import dataclass
from .response_model import ResponseModel

@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model_name: str
    model_selector: str | None = None
    max_tokens: int | None = None
    reasoning: str | None = None
    temperature: float | None = None
    top_p: float | None = None
