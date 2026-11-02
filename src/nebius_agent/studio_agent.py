"""
The FDE agent behind the Studio UI.

It is a forward-deployed engineer in a box: given a partner ask, it proves a framework works on
Token Factory by running a real probe in a sandbox, checks the endpoint with the conformance kit,
benchmarks candidate models on the partner's prompt, grounds itself with Tavily if needed, and
returns a typed report whose `spoken_summary` the Studio says out loud through Cartesia.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_ai import Agent, InstrumentationSettings, RunContext, ToolOutput
from pydantic_ai.capabilities import Instrumentation

from .config import Settings, get_settings
from .probes import PROBES, parse_probe_result
from .provider import make_client, make_model
from .sandbox import Sandbox, get_sandbox
from .tools import BenchmarkResult, Deps, benchmark_model, list_models, web_search
from .tracing import configure_tracing

LogFn = Callable[[str, dict[str, Any]], None]


@dataclass
class StudioDeps(Deps):
    """Deps + a sandbox + an event sink so tools can stream progress to the UI."""

    sandbox: Sandbox = field(default=None)  # type: ignore[assignment]
    emit: LogFn = field(default=lambda kind, data: None)


# ----------------------------------------------------------------------------- output
class ProbeOutcome(BaseModel):
    framework: str
    backend: str = Field(description="daytona | local | disabled")
    tool_called: bool
    exit_code: int
    elapsed_ms: float
    summary: str = Field(description="One sentence: what the probe proved or where it failed")


class ConformanceSummary(BaseModel):
    model: str
    passed: int
    partial: int
    failed: int
    notes: str = Field(description="The checks that were PARTIAL or FAIL, named, or 'all pass'")


class FdeReport(BaseModel):
    """What the agent hands back. The Studio renders it and speaks `spoken_summary`."""

    title: str = Field(description="Short title for this engagement, e.g. 'PydanticAI on Token Factory: verified'")
    workload: Literal["coding-agent", "voice", "chat", "batch", "rag", "integration", "other"]
    integration_verified: bool = Field(description="True only if a probe ran and its tool call actually happened")
    recommended_model: str
    probes: list[ProbeOutcome] = Field(default_factory=list)
    conformance: ConformanceSummary | None = None
    measurements: list[BenchmarkResult] = Field(default_factory=list)
    findings: list[str] = Field(description="3-6 concrete findings with numbers, for the partner README")
    next_steps: list[str] = Field(description="2-4 next steps, e.g. move to a dedicated endpoint, file a ticket")
    spoken_summary: str = Field(description="At most 60 words, first person, plain speech, no markdown; this is read aloud")


INSTRUCTIONS = """
You are a Nebius forward-deployed engineer. A partner has asked for something; your job is to PROVE
things on Token Factory, not assume them, and hand back a report a partner could act on.

Tools, and when to use them:
- list_models: see what Token Factory serves (filter by family if the partner named one).
- run_probe: run a real framework probe in a sandbox against Token Factory. This is the only way to
  claim an integration works. Use it whenever the ask mentions a framework, an integration, or
  "does X work". Prefer pydantic_ai unless the partner names another.
- run_conformance: the 12-check endpoint/model kit. Use it when the ask is about readiness, a new
  model, or when a probe fails (it tells you whether the model or the framework is at fault).
- benchmark_model: TTFT and tokens/s on the partner's own prompt. Use for model choice, voice,
  coding agents, or any latency question. Benchmark two candidates when choosing.
- web_search: live facts from the web via Tavily, only when the ask needs them.

Rules:
- Be concrete. Every finding carries a number or a named check.
- integration_verified is true only if a probe reported tool_called=true.
- If the sandbox is disabled, say so in findings and keep going with what you can measure.
- Say plainly when premium real-time inference is NOT the answer (batch, offline evals, embeddings).
- spoken_summary: at most 60 words, first person, no markdown, as if telling the partner on a call.
""".strip()


# ----------------------------------------------------------------------------- tools
async def run_probe(
    ctx: RunContext[StudioDeps],
    framework: Literal["pydantic_ai", "openai_sdk", "langgraph"] = "pydantic_ai",
    model: str | None = None,
) -> dict[str, Any]:
    """Run a framework probe in a sandbox against Token Factory and report whether its tool call really happened.

    Args:
        framework: which probe to run. pydantic_ai (typed tools + typed output), openai_sdk (raw wire baseline), langgraph (ReAct agent).
        model: Token Factory model id; defaults to the configured model.
    """
    probe = PROBES[framework]
    s = ctx.deps.settings
    model = model or s.model
    emit = ctx.deps.emit
    sandbox = ctx.deps.sandbox
    emit("sandbox", {"status": "start", "backend": sandbox.name, "framework": probe.framework, "model": model, "files": [probe.file]})

    def log(line: str) -> None:
        emit("sandbox_log", {"line": line})

    env = {"NEBIUS_BASE_URL": s.base_url, "NEBIUS_API_KEY": s.api_key or "mock", "NEBIUS_MODEL": model, "PYDANTIC_AI_NO_BANNER": "1"}
    command = probe.command if sandbox.name == "local" else f"{probe.install} && {probe.command}"
    fallback_note = ""
    try:
        result = await sandbox.run({probe.file: probe.source}, command, env=env, timeout_s=300, log=log)
    except Exception as e:  # noqa: BLE001 - a tool must report, not crash the run
        err = f"{type(e).__name__}: {str(e)[:300]}"
        log(f"{sandbox.name}: failed: {err}")
        if sandbox.name != "local" and s.allow_local_exec:
            from .sandbox import LocalSandbox

            fallback_note = f"{sandbox.name} unreachable ({err}); fell back to the local sandbox"
            log("falling back to the local sandbox (ALLOW_LOCAL_EXEC=1)")
            result = await LocalSandbox().run({probe.file: probe.source}, probe.command, env=env, timeout_s=300, log=log)
        else:
            from .sandbox import SandboxResult

            result = SandboxResult(sandbox.name, -1, "", err, 0.0)
    parsed = parse_probe_result(result.stdout) or {}
    outcome = {
        "framework": probe.framework,
        "framework_key": framework,
        "backend": result.backend,
        "sandbox_id": result.sandbox_id,
        "exit_code": result.exit_code,
        "tool_called": bool(parsed.get("tool_called")),
        "tool_args": parsed.get("tool_args"),
        "output": parsed.get("output"),
        "usage": parsed.get("usage"),
        "probe_elapsed_ms": parsed.get("elapsed_ms"),
        "sandbox_elapsed_ms": result.duration_ms,
        "stderr_tail": result.stderr[-600:] if result.stderr else "",
        "notes": probe.notes + (f"; {fallback_note}" if fallback_note else ""),
    }
    emit("sandbox", {"status": "done", **outcome})
    return outcome


async def run_conformance(ctx: RunContext[StudioDeps], model: str | None = None) -> dict[str, Any]:
    """Run the 12-check conformance kit against Token Factory for one model: tools, structured output, streaming, prefix cache.

    Args:
        model: Token Factory model id; defaults to the configured model.
    """
    from conformance.checks import ALL_CHECKS

    s = ctx.deps.settings
    model = model or s.model
    emit = ctx.deps.emit
    client: AsyncOpenAI = ctx.deps.client
    results = []
    emit("conformance", {"status": "start", "model": model, "total": len(ALL_CHECKS)})
    for check in ALL_CHECKS:
        r = await check(client, model)
        row = {"name": r.name, "status": r.status, "detail": r.detail, "ms": r.ms}
        results.append(row)
        emit("conformance_check", row)
    summary = {
        "model": model,
        "passed": sum(1 for r in results if r["status"] == "PASS"),
        "partial": sum(1 for r in results if r["status"] == "PARTIAL"),
        "failed": sum(1 for r in results if r["status"] == "FAIL"),
        "not_pass": [f"{r['name']}={r['status']}" for r in results if r["status"] != "PASS"],
        "results": results,
    }
    emit("conformance", {"status": "done", **{k: v for k, v in summary.items() if k != "results"}})
    return summary


async def benchmark(
    ctx: RunContext[StudioDeps],
    model: str,
    prompt: str = "Reply to a customer who asks for the status of their order, in 80 words.",
    runs: int = 2,
    max_tokens: int = 160,
) -> BenchmarkResult:
    """Measure time-to-first-token and tokens/second for one model on one prompt (streaming), median of runs.

    Args:
        model: a model id from list_models.
        prompt: the partner's own prompt; benchmark on real traffic, not a leaderboard.
        runs: repeated runs (the second run shows whether the prefix cache hit).
        max_tokens: cap on generated tokens per run.
    """
    ctx.deps.emit("benchmark", {"status": "start", "model": model, "runs": runs})
    r = await benchmark_model(ctx, model, prompt, runs, max_tokens)  # type: ignore[arg-type]
    ctx.deps.emit("benchmark", {"status": "done", **r.model_dump()})
    return r


STUDIO_TOOLS = [list_models, run_probe, run_conformance, benchmark, web_search]


# ----------------------------------------------------------------------------- agent
def build_studio_deps(settings: Settings | None = None, emit: LogFn | None = None) -> StudioDeps:
    s = settings or get_settings()
    return StudioDeps(settings=s, client=make_client(s), http=httpx.AsyncClient(), sandbox=get_sandbox(s), emit=emit or (lambda k, d: None))


def build_studio_agent(
    settings: Settings | None = None, model_name: str | None = None, *, instrument: bool = True, trace_exporter: str | None = None
) -> Agent[StudioDeps, FdeReport]:
    s = settings or get_settings()
    caps = []
    if instrument:
        provider = configure_tracing(s, trace_exporter)
        caps.append(Instrumentation(InstrumentationSettings(tracer_provider=provider, include_content=True)))
    return Agent(
        make_model(model_name, s),
        name="nebius-fde-studio",
        deps_type=StudioDeps,
        output_type=ToolOutput(FdeReport, name="final_result"),
        instructions=INSTRUCTIONS,
        tools=STUDIO_TOOLS,
        retries=2,
        capabilities=caps,
    )


SCENARIOS: list[dict[str, str]] = [
    {
        "id": "verify-pydantic-ai",
        "title": "Verify PydanticAI on Token Factory",
        "prompt": "A partner wants to ship a PydanticAI agent on Token Factory. Prove tool calling works through the framework with a real probe in the sandbox, run the conformance kit on the model, and benchmark it on a short support reply. Report what a partner README should say.",
        "blurb": "Sandbox probe + 12-check kit + benchmark. The core FDE motion.",
    },
    {
        "id": "voice-model-choice",
        "title": "Pick a model for a voice agent",
        "prompt": "A voice-agent partner (think Cartesia Line) needs the LLM leg to answer inside a 300 ms time-to-first-token budget. List the served models, benchmark two plausible candidates on a short customer-service reply, and recommend one. Say plainly if none meets the budget on the shared endpoint.",
        "blurb": "Two benchmarks, a latency budget, a recommendation you can speak.",
    },
    {
        "id": "framework-compare",
        "title": "Compare LangGraph vs PydanticAI on one model",
        "prompt": "Run the pydantic_ai probe and the langgraph probe in the sandbox against the same model. Tell me which frameworks' tool calling actually works on Token Factory, how long each took, and which one you'd recommend a coding-agent partner start with.",
        "blurb": "Two real probes in Daytona, one verdict.",
    },
    {
        "id": "new-model-readiness",
        "title": "Is this model agent-ready?",
        "prompt": "A new model just landed on Token Factory. Run the conformance kit on the configured model and summarise, check by check, what an agent builder can rely on (tools, parallel calls, json_schema output, streaming, prefix cache) and what to avoid.",
        "blurb": "The report we hand a partner on day one.",
    },
]
