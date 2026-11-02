"""
The one function a partner needs: a PydanticAI model object that talks to Token Factory.

Token Factory is OpenAI-compatible, so we use PydanticAI's OpenAIChatModel with an
OpenAIProvider pointed at the Token Factory base URL. Nothing else changes in the
partner's code; that is the point of the integration.
"""

from __future__ import annotations

from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

from .config import Settings, get_settings


def token_factory_profile() -> OpenAIModelProfile:
    """
    Capabilities we have verified on Token Factory. Keep this honest: the conformance kit
    (`nebius-agent verify`) is what updates it. If a model fails json_schema output, flip
    `openai_supports_strict_tool_definition` / `supports_json_schema_output` here.
    """
    return OpenAIModelProfile(
        supports_tools=True,
        supports_json_schema_output=True,
        supports_json_object_output=True,
        default_structured_output_mode="tool",  # most portable across open models; `native` is verified separately
        openai_supports_strict_tool_definition=False,  # open models: don't send `strict: true`
        supports_forced_tool_choice=True,
        openai_chat_supports_max_completion_tokens=False,  # vLLM-style servers want `max_tokens`
        openai_chat_thinking_field="reasoning_content",  # DeepSeek / vLLM reasoning parsers use this field
        supports_auto_cache=True,  # Token Factory prefix cache reports prompt_tokens_details.cached_tokens
    )


def token_factory_model_settings() -> OpenAIChatModelSettings:
    """
    Settings every Token Factory agent should carry. Found live, not assumed:
    Token Factory (vLLM-style) sends *cumulative* `usage` in every streamed chunk. PydanticAI
    adds per-chunk usage unless told the stream is cumulative, which over-counts tokens ~100x on
    long streams. `openai_continuous_usage_stats=True` makes it replace instead of add.
    """
    return OpenAIChatModelSettings(openai_continuous_usage_stats=True)


def make_client(settings: Settings | None = None) -> AsyncOpenAI:
    s = settings or get_settings()
    s.validate()
    return AsyncOpenAI(base_url=s.base_url, api_key=s.api_key or "mock")


def make_model(model_name: str | None = None, settings: Settings | None = None) -> OpenAIChatModel:
    s = settings or get_settings()
    client = make_client(s)
    return OpenAIChatModel(
        model_name or s.model,
        provider=OpenAIProvider(openai_client=client),
        profile=token_factory_profile(),
    )
