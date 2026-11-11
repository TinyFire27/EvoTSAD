"""Resolve a provider/model selector into a runtime model configuration."""

import os
import tomllib
from pathlib import Path

from . import LLMConfig

def resolve_model_config(
    model: str | None = None, *, config_path: Path
) -> LLMConfig:
    """Resolve a model from the supplied TOML file."""

    with config_path.expanduser().open("rb") as file:
        config = tomllib.load(file)

    selected = model if model is not None else config.get("default_model")
    if not isinstance(selected, str):
        raise ValueError("Set default_model in the model configuration.")

    provider_id, separator, model_id = selected.partition("/")
    if not separator or not provider_id or not model_id:
        raise ValueError(f"Expected a provider/model selector, got {selected!r}.")

    providers = config.get("providers")
    if not isinstance(providers, dict):
        raise ValueError("The model configuration requires a [providers] table.")
    provider = providers.get(provider_id)
    if not isinstance(provider, dict):
        raise ValueError(f"Unknown model provider {provider_id!r}.")

    models = provider.get("models")
    model_options = models.get(model_id) if isinstance(models, dict) else None
    if not isinstance(model_options, dict):
        raise ValueError(f"Unknown configured model {selected!r}.")

    base_url = provider.get("base_url")
    api_key_env = provider.get("api_key_env")
    if not isinstance(base_url, str) or not base_url:
        raise ValueError(f"Set base_url for provider {provider_id!r}.")
    if not isinstance(api_key_env, str) or not api_key_env:
        raise ValueError(f"Set api_key_env for provider {provider_id!r}.")
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"Set {api_key_env} to use provider {provider_id!r}.")

    try:
        return LLMConfig(
            base_url=base_url,
            api_key=api_key,
            model_name=model_id,
            model_selector=selected,
            **model_options,
        )
    except TypeError as exc:
        raise ValueError(f"Invalid model options for {selected!r}. {exc}") from exc
