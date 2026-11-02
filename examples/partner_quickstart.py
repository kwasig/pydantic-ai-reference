"""
The whole integration in 25 lines: what a partner copies into their own codebase.

    export NEBIUS_API_KEY=...            # https://tokenfactory.nebius.com
    python examples/partner_quickstart.py
"""

import asyncio
import os

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# 1. Point PydanticAI at Token Factory. That is the only Nebius-specific line.
model = OpenAIChatModel(
    os.getenv("NEBIUS_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507"),
    provider=OpenAIProvider(
        base_url=os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"), api_key=os.environ["NEBIUS_API_KEY"]
    ),
)


# 2. A typed result and a typed tool; PydanticAI turns both into JSON schema for the model.
class CityReport(BaseModel):
    city: str
    temperature_c: float
    summary: str


agent = Agent(model, output_type=CityReport, instructions="Use the tool, then answer.")


@agent.tool_plain
def get_weather(city: str) -> dict:
    """Current weather for a city (stub)."""
    return {"city": city, "temperature_c": 11.0, "sky": "overcast"}


# 3. Run. Tool call, validation and the typed output are handled by the framework.
Agent.instrument_all()  # OpenTelemetry spans; configure an exporter to ship them (see src/nebius_agent/tracing.py)
report = asyncio.run(agent.run("What's the weather in Amsterdam?")).output
print(report)
