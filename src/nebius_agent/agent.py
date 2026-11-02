"""
The reference agent.

    agent = build_agent()                      # PydanticAI Agent on Token Factory
    result = await agent.run("...", deps=deps) # result.output is a ModelRecommendation

Three things the exercise asks for are visible here:
  * tool calling through the framework's abstractions (plain typed functions in `tools.py`)
  * structured output (`ModelRecommendation`, delivered via a tool call by default, or via
    `response_format: json_schema` when output_mode="native")
  * a trace (PydanticAI's built-in OpenTelemetry instrumentation; see `tracing.py`)
"""

from __future__ import annotations

from typing import Literal

import httpx
from pydantic import BaseModel, Field
from pydantic_ai import Agent, InstrumentationSettings, NativeOutput, PromptedOutput, ToolOutput
from pydantic_ai.capabilities import Instrumentation

from .config import Settings, get_settings
from .provider import make_client, make_model
from .tools import ALL_TOOLS, BenchmarkResult, Deps
from .tracing import configure_tracing

OutputMode = Literal["tool", "native", "prompted", "text"]


class ModelRecommendation(BaseModel):
    """What the scout hands back to the partner."""

    workload: Literal["coding-agent", "voice", "chat", "batch", "rag", "other"] = Field(
        description="The workload class you inferred from the request"
    )
    recommended_model: str = Field(description="The Token Factory model id you recommend")
    rationale: str = Field(description="Two or three sentences: why this model, citing the measurements")
    measurements: list[BenchmarkResult] = Field(default_factory=list, description="Every benchmark you ran")
    latency_budget_met: bool = Field(description="True if the recommended model meets the latency budget implied by the workload")
    next_step: str = Field(description="One concrete next step for the partner, e.g. move to a dedicated endpoint")


INSTRUCTIONS = """
You are the Nebius Token Factory latency scout, a forward-deployed engineer's assistant.
Given a partner's workload, decide which served model to recommend and why.

Method:
1. Call list_models to see what is served (filter by family if the user names one).
2. Call benchmark_model on one or two plausible candidates with the partner's own prompt.
3. If the partner's question needs facts from the live web, call web_search (Tavily).
4. Judge against the workload's latency budget:
   - voice: time to first token under 300 ms matters most
   - coding-agent: tokens/second matters most (many sequential calls)
   - chat: both, but tokens/second over 40 is usually enough
   - batch / rag / other: cost and quality first; latency is secondary
5. Return the recommendation. Be concrete: name the model, quote the numbers you measured,
   and say plainly when premium real-time inference is not the right answer (batch, offline evals, embeddings).
""".strip()


def build_deps(settings: Settings | None = None) -> Deps:
    s = settings or get_settings()
    return Deps(settings=s, client=make_client(s), http=httpx.AsyncClient())


def build_agent(
    model_name: str | None = None,
    *,
    settings: Settings | None = None,
    output_mode: OutputMode = "tool",
    instrument: bool = True,
    trace_exporter: str | None = None,
) -> Agent[Deps, ModelRecommendation | str]:
    s = settings or get_settings()
    model = make_model(model_name, s)

    output_type: object
    if output_mode == "tool":
        output_type = ToolOutput(ModelRecommendation, name="final_result")
    elif output_mode == "native":
        output_type = NativeOutput(ModelRecommendation)
    elif output_mode == "prompted":
        output_type = PromptedOutput(ModelRecommendation)
    else:
        output_type = str

    capabilities = []
    if instrument:
        provider = configure_tracing(s, trace_exporter)
        capabilities.append(Instrumentation(InstrumentationSettings(tracer_provider=provider, include_content=True)))

    return Agent(
        model,
        name="token-factory-latency-scout",
        deps_type=Deps,
        output_type=output_type,  # type: ignore[arg-type]
        instructions=INSTRUCTIONS,
        tools=ALL_TOOLS,
        retries=2,
        capabilities=capabilities,
    )
