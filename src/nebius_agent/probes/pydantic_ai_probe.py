"""Probe: does PydanticAI tool calling + typed output work on this Token Factory model?"""

import asyncio
import json
import os
import time

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
MODEL = os.environ.get("NEBIUS_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")

model = OpenAIChatModel(
    MODEL,
    provider=OpenAIProvider(base_url=BASE_URL, api_key=os.environ.get("NEBIUS_API_KEY", "mock")),
    profile=OpenAIModelProfile(openai_supports_strict_tool_definition=False, openai_chat_supports_max_completion_tokens=False),
)


class Verdict(BaseModel):
    city: str
    temperature_c: float
    one_line_summary: str


calls: list[dict] = []
agent = Agent(model, output_type=Verdict, instructions="Call get_weather for the city, then answer.", retries=2)


@agent.tool_plain
def get_weather(city: str) -> dict:
    """Current weather for a city."""
    calls.append({"city": city})
    return {"city": city, "temperature_c": 11.0, "sky": "overcast"}


async def main() -> int:
    t0 = time.perf_counter()
    result = await agent.run("What is the weather in Amsterdam right now?")
    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    u = result.usage
    out = {
        "framework": "pydantic_ai",
        "model": MODEL,
        "tool_called": bool(calls),
        "tool_args": calls[0] if calls else None,
        "output": result.output.model_dump(),
        "usage": {"requests": u.requests, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens},
        "elapsed_ms": elapsed,
    }
    print("PROBE_RESULT " + json.dumps(out))
    return 0 if calls else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
