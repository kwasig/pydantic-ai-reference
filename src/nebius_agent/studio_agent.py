"""
The FDE agent behind the Studio UI.

It is a forward-deployed engineer in a box: given a partner ask, it proves a framework works on
Token Factory by running a real probe in a sandbox, checks the endpoint with the conformance kit,
benchmarks candidate models on the partner's prompt, grounds itself with Tavily if needed, and
returns a typed report whose `spoken_summary` the Studio says out loud through Cartesia.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_ai import Agent, InstrumentationSettings, ModelRetry, RunContext, ToolOutput
from pydantic_ai.capabilities import Instrumentation

from .config import Settings, get_settings
from .probes import PROBES, parse_probe_result
from .provider import make_client, make_model, token_factory_model_settings
from .sandbox import Sandbox, get_sandbox
from .tools import BenchmarkResult, Deps, benchmark_model, list_models, measure, web_search
from .tracing import configure_tracing

LogFn = Callable[[str, dict[str, Any]], None]


@dataclass
class StudioDeps(Deps):
    """Deps + a sandbox + an event sink so tools can stream progress to the UI."""

    sandbox: Sandbox = field(default=None)  # type: ignore[assignment]
    emit: LogFn = field(default=lambda kind, data: None)
    eval_model: str | None = None  # the model the partner asked about; tools default to it
    save_artifact: Callable[[str, bytes], str] | None = None  # (name, bytes) -> url, provided by the Studio


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
- list_models: see what Token Factory serves (filter by family if the partner named one). Only use
  model ids that appear in its output; if a requested model is missing, say so and use the nearest.
- run_probe: run a real framework probe in a sandbox against Token Factory. This is the only way to
  claim an integration works. Use it whenever the ask mentions a framework, an integration, or
  "does X work". Prefer pydantic_ai unless the partner names another.
- run_coding_session: a real coding-agent session inside the sandbox (files + shell): the agent on the
  model writes a module and tests, runs pytest to green, renders a chart. Use it for any coding-agent
  ask; it shows what the model does over many sequential tool calls, which a one-shot probe cannot.
- run_conformance: the 13-check endpoint/model kit (tools, parallel calls, forced choice, json_schema,
  streaming deltas, prefix cache). ALWAYS run it once on the model you recommend: the partner's
  README needs the check table, and it tells you whether the model or the framework is at fault.
- benchmark_model: TTFT and tokens/s on the partner's own prompt. Use for model choice, voice,
  coding agents, or any latency question. Benchmark two candidates when choosing.
- provision_dedicated_endpoint: when no model meets the partner's latency budget on the shared
  endpoint, provision a temporary dedicated endpoint for the best candidate, benchmark it there, and
  tear it down. It refuses unless the operator allowed it; then recommend it as a next step instead.
- web_search: live facts from the web via Tavily, only when the ask needs them.

Rules:
- A complete run calls run_probe, run_conformance and benchmark (two candidates when choosing a
  model). Skip one only if the partner's ask makes it meaningless, and say so in findings.
- Be concrete. Every finding carries a number or a named check.
- Benchmarks: quote tokens_per_s only when stream_note is empty; if it says "bursty", quote
  end_to_end_tokens_per_s and say the stream was buffered. Mention reasoning tokens when present.
- integration_verified is true only if a probe reported tool_called=true.
- If the sandbox is disabled, say so in findings and keep going with what you can measure.
- Say plainly when premium real-time inference is NOT the answer (batch, offline evals, embeddings).
- spoken_summary: at most 60 words, first person, no markdown, as if telling the partner on a call.
""".strip()


# ----------------------------------------------------------------------------- tools
async def _served_model(ctx: RunContext[StudioDeps], model: str | None) -> str:
    """Resolve a model id against what Token Factory serves; ask the agent to retry with a real id otherwise."""
    model = (model or ctx.deps.eval_model or ctx.deps.settings.model).strip()
    if model in ("configured_model", "configured model", "default", "default_model"):
        return ctx.deps.settings.model
    try:
        page = await ctx.deps.client.models.list()
        served = [m.id for m in page.data]
    except Exception:  # noqa: BLE001 - if the list fails, let the call itself decide
        return model
    if served and model not in served:
        near = [m for m in served if model.split("/")[-1].lower()[:8] in m.lower()][:5]
        raise ModelRetry(f"model {model!r} is not served on Token Factory. Use an id from list_models, e.g. {near or served[:5]}.")
    return model


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
    model = await _served_model(ctx, model)
    emit = ctx.deps.emit
    sandbox = ctx.deps.sandbox
    emit(
        "sandbox",
        {
            "status": "start",
            "probe": framework,
            "backend": sandbox.name,
            "framework": probe.framework,
            "model": model,
            "files": [probe.file],
        },
    )

    def log(line: str) -> None:
        emit("sandbox_log", {"probe": framework, "line": line})

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
    emit("sandbox", {"status": "done", "probe": framework, **outcome})
    return outcome


async def run_coding_session(ctx: RunContext[StudioDeps], model: str | None = None) -> dict[str, Any]:
    """Run a real coding-agent session inside the sandbox: a PydanticAI agent on the model gets a filesystem and a
    shell, implements a module, writes pytest tests, runs them to green and renders a matplotlib chart. Reports every
    tool call, the independent pytest result, the files written and the chart. Use it for coding-agent asks.

    Args:
        model: Token Factory model id the coding agent runs on; defaults to the model under evaluation.
    """
    probe = PROBES["coding_session"]
    s = ctx.deps.settings
    model = await _served_model(ctx, model)
    emit = ctx.deps.emit
    sandbox = ctx.deps.sandbox
    key = "coding_session"
    emit(
        "sandbox",
        {"status": "start", "probe": key, "backend": sandbox.name, "framework": probe.framework, "model": model, "files": [probe.file]},
    )

    def log(line: str) -> None:
        if not line.startswith("PROBE_RESULT"):
            emit("sandbox_log", {"probe": key, "line": line})

    env = {
        "NEBIUS_BASE_URL": s.base_url,
        "NEBIUS_API_KEY": s.api_key or "mock",
        "NEBIUS_MODEL": model,
        "PYDANTIC_AI_NO_BANNER": "1",
        "MPLBACKEND": "Agg",
    }
    command = probe.command if sandbox.name == "local" else f"{probe.install} && {probe.command}"
    fallback_note = ""
    try:
        result = await sandbox.run({probe.file: probe.source}, command, env=env, timeout_s=600, log=log)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:300]}"
        log(f"{sandbox.name}: failed: {err}")
        if sandbox.name != "local" and s.allow_local_exec:
            from .sandbox import LocalSandbox

            fallback_note = f"{sandbox.name} unreachable ({err}); fell back to the local sandbox"
            result = await LocalSandbox().run({probe.file: probe.source}, probe.command, env=env, timeout_s=600, log=log)
        else:
            from .sandbox import SandboxResult

            result = SandboxResult(sandbox.name, -1, "", err, 0.0)
    parsed = parse_probe_result(result.stdout) or {}
    tests = parsed.get("tests") or {}
    chart_b64 = parsed.pop("chart_png_b64", None)
    chart_url = None
    if chart_b64 and ctx.deps.save_artifact:
        chart_url = ctx.deps.save_artifact("chart.png", base64.b64decode(chart_b64))
    outcome = {
        "framework": probe.framework,
        "framework_key": key,
        "backend": result.backend,
        "sandbox_id": result.sandbox_id,
        "exit_code": result.exit_code,
        "tool_called": bool(parsed.get("tool_called")),
        "steps": parsed.get("steps") or [],
        "tests": tests,
        "tests_passed": bool(tests.get("ran")) and tests.get("failed", 1) == 0 and tests.get("passed", 0) > 0,
        "files": parsed.get("files") or [],
        "chart_url": chart_url,
        "output": parsed.get("output"),
        "usage": parsed.get("usage"),
        "probe_elapsed_ms": parsed.get("elapsed_ms"),
        "sandbox_elapsed_ms": result.duration_ms,
        "stderr_tail": result.stderr[-600:] if result.stderr else "",
        "notes": probe.notes + (f"; {fallback_note}" if fallback_note else ""),
    }
    emit("sandbox", {"status": "done", "probe": key, **outcome})
    # what the agent reasons over: no base64, short tails
    return {k: v for k, v in outcome.items() if k not in ("stderr_tail",)}


async def provision_dedicated_endpoint(
    ctx: RunContext[StudioDeps],
    model: str | None = None,
    prompt: str = "Reply to a customer who asks for the status of their order, in 80 words.",
    max_minutes: int = 12,
) -> dict[str, Any]:
    """Provision a temporary dedicated Token Factory endpoint for the model, wait until it is ready, benchmark the
    partner's prompt on it, and tear it down. Use it when no model meets the latency budget on the shared
    endpoint and the operator has allowed it (ALLOW_DEDICATED=1); otherwise it returns allowed=false and you
    must recommend the dedicated endpoint as a next step instead.

    Args:
        model: Token Factory model id; defaults to the model under evaluation.
        prompt: the partner's prompt to benchmark on the dedicated endpoint.
        max_minutes: hard cap; the endpoint is deleted when this elapses even if provisioning is not done.
    """
    from .dedicated import DedicatedClient, dedicated_allowed

    s = ctx.deps.settings
    model = await _served_model(ctx, model)
    emit = ctx.deps.emit
    if not dedicated_allowed() or s.is_mock or not s.api_key:
        emit(
            "dedicated",
            {"status": "skipped", "model": model, "reason": "ALLOW_DEDICATED is not set" if not s.is_mock else "mock Token Factory"},
        )
        return {
            "allowed": False,
            "model": model,
            "reason": "dedicated endpoints are not enabled for this run; recommend one as a next step",
        }

    client = DedicatedClient(s.api_key)
    ep = None
    t0 = time.perf_counter()
    try:
        tpl = await client.find_template(model)
        if not tpl:
            emit("dedicated", {"status": "skipped", "model": model, "reason": "no dedicated-endpoint template for this model"})
            return {"allowed": True, "model": model, "provisioned": False, "reason": "no template for this model"}
        ep = await client.create(model)
        emit(
            "dedicated",
            {
                "status": "creating",
                "model": model,
                "name": ep.name,
                "endpoint_id": ep.endpoint_id,
                "gpu_type": ep.gpu_type,
                "gpu_count": ep.gpu_count,
                "region": ep.region,
                "max_minutes": max_minutes,
            },
        )

        def log(line: str) -> None:
            emit("dedicated_log", {"line": line})

        ready = await client.wait_ready(ep, timeout_s=max_minutes * 60, log=log)
        if not ready:
            emit("dedicated", {"status": "timeout", "model": model, "name": ep.name, "waited_s": round(time.perf_counter() - t0)})
            return {
                "allowed": True,
                "model": model,
                "provisioned": True,
                "ready": False,
                "reason": f"not ready within {max_minutes} min; deleted",
            }
        provision_s = round(ep.ready_at - ep.created_at)
        emit("dedicated", {"status": "ready", "model": model, "name": ep.name, "provision_s": provision_s, "base_url": ep.base_url})
        # benchmark on the dedicated endpoint with the same code path as the shared one
        from openai import AsyncOpenAI

        ded_client = AsyncOpenAI(base_url=ep.base_url, api_key=s.api_key)
        emit("benchmark", {"status": "start", "model": f"{model} (dedicated)", "runs": 3})
        r = await measure(ded_client, ep.routing_key, prompt=prompt, runs=3, max_tokens=160)
        d = r.model_dump()
        d["model"] = f"{model} (dedicated)"
        emit("benchmark", {"status": "done", **d})
        return {
            "allowed": True,
            "model": model,
            "provisioned": True,
            "ready": True,
            "endpoint": ep.name,
            "gpu": f"{ep.gpu_count}x {ep.gpu_type}",
            "region": ep.region,
            "provision_s": provision_s,
            "dedicated_benchmark": d,
            "billable_minutes": ep.billable_minutes,
            "torn_down": True,
        }
    except Exception as e:  # noqa: BLE001
        emit("dedicated", {"status": "error", "model": model, "error": f"{type(e).__name__}: {str(e)[:300]}"})
        return {"allowed": True, "model": model, "provisioned": ep is not None, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    finally:
        if ep is not None:
            ok = await client.delete(ep.endpoint_id, ep.name)
            emit(
                "dedicated",
                {
                    "status": "deleted",
                    "model": model,
                    "name": ep.name,
                    "ok": ok,
                    "billable_minutes": ep.billable_minutes,
                    "total_s": round(time.perf_counter() - t0),
                },
            )
        await client.aclose()


async def run_conformance(ctx: RunContext[StudioDeps], model: str | None = None) -> dict[str, Any]:
    """Run the 13-check conformance kit against Token Factory for one model: tools, structured output, streaming, prefix cache.

    Args:
        model: Token Factory model id; defaults to the configured model.
    """
    from conformance.checks import ALL_CHECKS

    model = await _served_model(ctx, model)
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
    model = await _served_model(ctx, model)
    ctx.deps.emit("benchmark", {"status": "start", "model": model, "runs": runs})
    try:
        r = await benchmark_model(ctx, model, prompt, runs, max_tokens)  # type: ignore[arg-type]
    except ModelRetry:
        raise
    except Exception as e:
        ctx.deps.emit("benchmark", {"status": "error", "model": model, "error": f"{type(e).__name__}: {str(e)[:200]}"})
        raise ModelRetry(
            f"benchmark of {model} failed: {type(e).__name__}: {str(e)[:200]}. Choose another served model or report it as a finding."
        ) from e
    ctx.deps.emit("benchmark", {"status": "done", **r.model_dump()})
    return r


STUDIO_TOOLS = [list_models, run_probe, run_coding_session, run_conformance, benchmark, provision_dedicated_endpoint, web_search]


# ----------------------------------------------------------------------------- agent
def build_studio_deps(settings: Settings | None = None, emit: LogFn | None = None, eval_model: str | None = None) -> StudioDeps:
    s = settings or get_settings()
    return StudioDeps(
        settings=s,
        client=make_client(s),
        http=httpx.AsyncClient(),
        sandbox=get_sandbox(s),
        emit=emit or (lambda k, d: None),
        eval_model=eval_model,
    )


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
        instructions=INSTRUCTIONS
        + f"\n\nThe configured default model is {s.model}; when a request says 'the configured model' pass that id.",
        tools=STUDIO_TOOLS,
        retries=2,
        model_settings=token_factory_model_settings(),
        capabilities=caps,
    )


SCENARIOS: list[dict[str, str]] = [
    {
        "id": "voice-cartesia",
        "model": "MiniMaxAI/MiniMax-M3.1-Flash",
        "title": "Voice agent · Cartesia + Token Factory",
        "workload": "voice",
        "framework": "pydantic_ai",
        "budget_ms": "300",
        "prompt": "We are building a real-time voice agent for customer support with Cartesia for speech and Token Factory for the LLM leg. The LLM must answer inside a 300 ms time-to-first-token budget. Prove tool calling works through PydanticAI with a real probe in the sandbox, list the served models, benchmark two fast candidates on a short customer-service reply (MiniMaxAI/MiniMax-M3.1-Flash and Qwen/Qwen3-30B-A3B-Instruct-2507 if served, else the two smallest instruct models), and recommend one. If none meets the budget on the shared endpoint, provision a temporary dedicated endpoint for the best candidate, benchmark it there and tear it down; if that is not allowed, say what a dedicated endpoint would change and recommend it as the next step.",
        "blurb": "Latency budget, two benchmarks, a verified integration, a spoken verdict.",
    },
    {
        "id": "coding-daytona",
        "model": "moonshotai/Kimi-K3",
        "title": "Coding agent · Daytona sandboxes",
        "workload": "coding-agent",
        "framework": "langgraph",
        "budget_ms": "",
        "prompt": "We run a coding agent that executes in Daytona sandboxes and makes dozens of sequential model calls per task, so tokens per second is what we feel. Run a coding-agent session in the sandbox on moonshotai/Kimi-K3 (fall back to the configured model if it is not served) so we can see it write code, tests and a chart over many tool calls; also run the langgraph probe on it, run the conformance kit on it, benchmark it against moonshotai/Kimi-K2.7-Code on a code-edit prompt, and tell us which framework to start with and what throughput to expect.",
        "blurb": "A coding agent works in a Daytona sandbox: code, tests, chart. Then the kit and throughput.",
    },
    {
        "id": "migrate-support-bot",
        "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "title": "Migrate a support bot from a closed API",
        "workload": "chat",
        "framework": "openai_sdk",
        "budget_ms": "1000",
        "prompt": "We are moving a customer-support chatbot off a closed API to open models on Token Factory. It uses the OpenAI SDK directly with one tool (order lookup) and needs structured JSON answers. Prove the raw OpenAI SDK tool call works against deepseek-ai/DeepSeek-V4.1-Flash-1017 (fall back to the configured model if not served), run the conformance kit on it to confirm json_schema output and streaming, benchmark it against Qwen/Qwen3-30B-A3B-Instruct-2507 on a support reply, search the web for the current Token Factory rate limits and pricing page, and tell us what to change in our code.",
        "blurb": "Wire-level proof, structured output checked, migration notes.",
    },
    {
        "id": "model-readiness",
        "model": "moonshotai/Kimi-K3",
        "title": "Is this model agent-ready?",
        "workload": "integration",
        "framework": "pydantic_ai",
        "budget_ms": "",
        "prompt": "A new model just landed on Token Factory: moonshotai/Kimi-K3 (use the configured model if it is not served). Run the conformance kit on it and summarise, check by check, what an agent builder can rely on (tools, parallel calls, json_schema output, streaming, prefix cache) and what to avoid. Then run the pydantic_ai probe on it to confirm a framework round-trip, and benchmark it against Qwen/Qwen3.5-397B-A17B on a short agent step.",
        "blurb": "The day-one report for a new model.",
    },
]


SNIPPETS: dict[str, str] = {
    "pydantic_ai": '''from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIChatModel(
    "{model}",
    provider=OpenAIProvider(base_url="{base_url}", api_key=NEBIUS_API_KEY),
)
agent = Agent(model, output_type=YourOutput, instructions="...")

@agent.tool_plain
def your_tool(arg: str) -> dict:
    """What the tool does."""
    ...

result = await agent.run("...")   # tool calls, validation and typed output handled by PydanticAI''',
    "langgraph": '''from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model="{model}", base_url="{base_url}", api_key=NEBIUS_API_KEY, temperature=0)

@tool
def your_tool(arg: str) -> dict:
    """What the tool does."""
    ...

agent = create_react_agent(llm, [your_tool])
state = agent.invoke({{"messages": [("user", "...")]}})''',
    "openai_sdk": """from openai import OpenAI

client = OpenAI(base_url="{base_url}", api_key=NEBIUS_API_KEY)
r = client.chat.completions.create(
    model="{model}",
    messages=[{{"role": "user", "content": "..."}}],
    tools=[{{"type": "function", "function": {{"name": "your_tool", "parameters": {{...}}}}}}],
)
call = r.choices[0].message.tool_calls[0]   # run it, append a role="tool" message, call again""",
}


def integration_snippet(framework_key: str, model: str, base_url: str) -> str:
    return SNIPPETS.get(framework_key, SNIPPETS["pydantic_ai"]).format(model=model, base_url=base_url)
