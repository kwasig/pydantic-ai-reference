"""
Tools for the reference agent.

The agent is a "latency scout": the thing an FDE does on day one with a partner.
Given a workload, it lists what Token Factory serves, measures a couple of candidates
on the partner's own prompt, optionally grounds itself with Tavily (Nebius's search
product) and returns a typed recommendation.

Every tool is a plain async function with typed arguments; PydanticAI turns the
signature + docstring into the JSON schema that Token Factory receives. That is the
abstraction the exercise asks us to verify, so the tests call these *through* the agent.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from .config import Settings


@dataclass
class Deps:
    """Runtime dependencies injected into every tool call (PydanticAI `deps`)."""

    settings: Settings
    client: AsyncOpenAI
    http: httpx.AsyncClient


class BenchmarkResult(BaseModel):
    model: str
    ttft_ms: float = Field(description="Time to first streamed token, milliseconds (median of runs)")
    tokens_per_s: float = Field(description="Decode throughput for one user, output tokens per second (median)")
    output_tokens: int
    cached_prompt_tokens: int = Field(default=0, description="prompt_tokens_details.cached_tokens reported by the server")
    runs: int


# ----------------------------------------------------------------------------- list_models
async def list_models(ctx: RunContext[Deps], contains: str | None = None, limit: int = 20) -> list[str]:
    """List model ids currently served by Token Factory.

    Args:
        contains: optional case-insensitive substring filter, e.g. "qwen" or "deepseek".
        limit: maximum number of ids to return.
    """
    page = await ctx.deps.client.models.list()
    ids = [m.id for m in page.data]
    if contains:
        ids = [i for i in ids if contains.lower() in i.lower()]
    return ids[:limit]


# ----------------------------------------------------------------------------- benchmark_model
async def benchmark_model(
    ctx: RunContext[Deps],
    model: str,
    prompt: str = "Write a 120-word product update for an engineering team.",
    runs: int = 2,
    max_tokens: int = 160,
) -> BenchmarkResult:
    """Measure time-to-first-token and tokens/second for one model on one prompt, using streaming.

    Args:
        model: a model id from list_models.
        prompt: the partner's own prompt; benchmark on real traffic, not a leaderboard.
        runs: number of repeated runs (the second run also shows whether the prefix cache hit).
        max_tokens: cap on generated tokens per run.
    """
    runs = max(1, min(runs, 5))
    ttfts: list[float] = []
    tps: list[float] = []
    out_tokens = 0
    cached = 0
    # a stable, long-ish system prompt so the second run can hit Token Factory's prefix cache
    system = (
        "You are a concise technical writer for Nebius Token Factory partners. "
        "Follow the user's instructions exactly, keep a neutral tone, and prefer short sentences. "
    ) * 6
    for _ in range(runs):
        t0 = time.perf_counter()
        first: float | None = None
        n = 0
        stream = await ctx.deps.client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if chunk.usage:
                out_tokens = chunk.usage.completion_tokens or out_tokens
                details = getattr(chunk.usage, "prompt_tokens_details", None)
                if details is not None and getattr(details, "cached_tokens", None):
                    cached = details.cached_tokens
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                if first is None:
                    first = time.perf_counter() - t0
                n += 1
        total = time.perf_counter() - t0
        ttfts.append((first or total) * 1000)
        decode = max(1e-6, total - (first or 0.0))
        tps.append((out_tokens or n) / decode)
    ttfts.sort()
    tps.sort()
    return BenchmarkResult(
        model=model,
        ttft_ms=round(ttfts[len(ttfts) // 2], 1),
        tokens_per_s=round(tps[len(tps) // 2], 1),
        output_tokens=out_tokens,
        cached_prompt_tokens=cached,
        runs=runs,
    )


# ----------------------------------------------------------------------------- web_search (Tavily)
async def web_search(ctx: RunContext[Deps], query: str, max_results: int = 3) -> list[dict[str, str]]:
    """Search the live web with Tavily (Nebius's agentic search) and return title, url and a snippet per result.

    Args:
        query: what to search for.
        max_results: how many results to return (1-5).
    """
    key = ctx.deps.settings.tavily_api_key
    if not key:
        return [{"title": "Tavily not configured", "url": "", "content": "Set TAVILY_API_KEY to enable live web grounding."}]
    r = await ctx.deps.http.post(
        "https://api.tavily.com/search",
        json={"api_key": key, "query": query, "max_results": max(1, min(max_results, 5)), "include_answer": False},
        timeout=30,
    )
    r.raise_for_status()
    return [
        {"title": it.get("title", ""), "url": it.get("url", ""), "content": (it.get("content") or "")[:400]}
        for it in r.json().get("results", [])
    ]


# ----------------------------------------------------------------------------- run_python (sandbox)
async def run_python(ctx: RunContext[Deps], code: str, timeout_s: int = 20) -> dict[str, Any]:
    """Execute a short Python snippet in a sandbox and return stdout, stderr and the exit code.

    Uses a Daytona sandbox when DAYTONA_API_KEY is set; otherwise a local subprocess, only if
    ALLOW_LOCAL_EXEC=1. Keep snippets small: this is for calculations and quick checks, not builds.

    Args:
        code: Python source to run.
        timeout_s: wall-clock limit in seconds.
    """
    s = ctx.deps.settings
    if s.daytona_api_key:
        return await _run_in_daytona(s, code, timeout_s)
    if not s.allow_local_exec:
        return {"exit_code": -1, "stdout": "", "stderr": "Sandbox disabled: set DAYTONA_API_KEY or ALLOW_LOCAL_EXEC=1."}
    return await asyncio.to_thread(_run_local, code, timeout_s)


def _run_local(code: str, timeout_s: int) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        p = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True, timeout=timeout_s)
        return {"exit_code": p.returncode, "stdout": p.stdout[-4000:], "stderr": p.stderr[-4000:]}
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "stdout": "", "stderr": f"timed out after {timeout_s}s"}


async def _run_in_daytona(s: Settings, code: str, timeout_s: int) -> dict[str, Any]:
    """Daytona sandbox via its SDK (optional dependency: `pip install daytona`)."""
    try:
        from daytona import AsyncDaytona, DaytonaConfig  # type: ignore
    except ImportError:
        return {"exit_code": -1, "stdout": "", "stderr": "pip install daytona to use Daytona sandboxes"}
    async with AsyncDaytona(DaytonaConfig(api_key=s.daytona_api_key)) as d:
        sandbox = await d.create()
        try:
            resp = await sandbox.process.code_run(code, timeout=timeout_s)
            return {"exit_code": resp.exit_code, "stdout": (resp.result or "")[-4000:], "stderr": ""}
        finally:
            await sandbox.delete()


ALL_TOOLS = [list_models, benchmark_model, web_search, run_python]


def tool_schemas_preview(tools=ALL_TOOLS) -> str:
    """Debug helper: show the JSON schema PydanticAI will send for each tool."""
    from pydantic_ai.tools import Tool

    out = {}
    for fn in tools:
        t = Tool(fn, takes_ctx=True)
        out[t.name] = {"description": t.description, "parameters": t.function_schema.json_schema}
    return json.dumps(out, indent=2)
