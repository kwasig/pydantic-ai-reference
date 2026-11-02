"""
These tests exercise Token Factory THROUGH PydanticAI's abstractions, which is what the
exercise asks us to verify rather than assume: the tool schema PydanticAI derives, the
tool-call round trip the framework drives, structured output in both modes, streaming,
and the OpenTelemetry spans the framework emits.
"""

from __future__ import annotations

import json

import pytest
from pydantic_ai import Agent, RunContext

from nebius_agent.agent import ModelRecommendation, build_agent, build_deps
from nebius_agent.tools import Deps, tool_schemas_preview
from nebius_agent.tracing import memory_spans


async def test_tool_is_called_through_pydantic_ai(settings):
    """The model's tool call must reach OUR Python function via the framework, with validated args."""
    calls: list[dict] = []

    async def list_models(ctx: RunContext[Deps], contains: str | None = None, limit: int = 20) -> list[str]:
        """List model ids currently served by Token Factory."""
        calls.append({"contains": contains, "limit": limit})
        page = await ctx.deps.client.models.list()
        return [m.id for m in page.data][:limit]

    from nebius_agent.provider import make_model

    agent = Agent(make_model(settings=settings), deps_type=Deps, tools=[list_models], output_type=str)
    result = await agent.run("Which models are served? Use list_models.", deps=build_deps(settings))
    assert calls, "tool function was never invoked by the framework"
    assert isinstance(calls[0]["limit"], int)
    assert "Qwen" in result.output
    assert result.usage.requests == 2  # one tool-call turn + one answer turn


async def test_structured_output_via_tool_mode(settings):
    agent = build_agent(settings=settings, output_mode="tool")
    result = await agent.run("Recommend a model for a coding agent; benchmark one candidate.", deps=build_deps(settings))
    out = result.output
    assert isinstance(out, ModelRecommendation)
    assert out.recommended_model
    assert out.measurements and out.measurements[0].runs >= 1
    assert out.workload in ("coding-agent", "voice", "chat", "batch", "rag", "other")


async def test_structured_output_via_native_json_schema(settings):
    agent = build_agent(settings=settings, output_mode="native")
    result = await agent.run("Recommend a model for a voice agent.", deps=build_deps(settings))
    assert isinstance(result.output, ModelRecommendation)
    assert isinstance(result.output.latency_budget_met, bool)


async def test_streaming_text_delivers_multiple_deltas(settings):
    agent = build_agent(settings=settings, output_mode="text")
    deltas: list[str] = []
    async with agent.run_stream("Tell me what you can do, briefly.", deps=build_deps(settings)) as result:
        async for d in result.stream_text(delta=True, debounce_by=None):
            deltas.append(d)
    assert len(deltas) > 1, "expected several streamed deltas"
    assert "".join(deltas).strip()


async def test_trace_spans_follow_genai_semantic_conventions(settings):
    agent = build_agent(settings=settings, output_mode="tool", trace_exporter="memory")
    await agent.run("Recommend a model for chat. Benchmark one model.", deps=build_deps(settings))
    spans = memory_spans()
    names = [s.name for s in spans]
    assert any(n.startswith("invoke_agent") for n in names), names
    assert any(n.startswith("chat ") for n in names), names
    assert any(n.startswith("execute_tool") for n in names), names
    chat = next(s for s in spans if s.name.startswith("chat "))
    assert chat.attributes.get("gen_ai.operation.name") == "chat"
    assert chat.attributes.get("gen_ai.request.model") == settings.model
    # usage is recorded on the span, so a backend can bill/attribute tokens per partner
    assert any(k.endswith("input_tokens") for k in chat.attributes)


def test_tool_schemas_are_well_formed():
    schemas = json.loads(tool_schemas_preview())
    assert set(schemas) >= {"list_models", "benchmark_model", "web_search", "run_python"}
    bm = schemas["benchmark_model"]["parameters"]
    assert bm["properties"]["model"]["type"] == "string"
    assert "model" in bm["required"]
    assert "Measure time-to-first-token" in schemas["benchmark_model"]["description"]


async def test_run_python_local_sandbox(settings):
    from nebius_agent.tools import run_python

    class Ctx:  # minimal RunContext stand-in
        deps = build_deps(settings)

    out = await run_python(Ctx(), "print(6*7)")  # type: ignore[arg-type]
    assert out["exit_code"] == 0 and out["stdout"].strip() == "42"


@pytest.mark.parametrize("mode", ["tool", "native", "prompted"])
async def test_all_output_modes_validate(settings, mode):
    agent = build_agent(settings=settings, output_mode=mode)  # type: ignore[arg-type]
    result = await agent.run("Recommend a model for RAG.", deps=build_deps(settings))
    assert isinstance(result.output, ModelRecommendation)
