"""Probe: a real coding-agent session inside the sandbox.

A PydanticAI agent on the Token Factory model gets a filesystem and a shell (this sandbox), and has
to: implement a small module, write pytest tests for it, run them until green, then render a chart
with matplotlib. Every tool call is recorded; the chart comes back base64-encoded.

Prints one line: PROBE_RESULT {...} with steps, test results, files and the chart.
"""

import asyncio
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
MODEL = os.environ.get("NEBIUS_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")
WORK = Path(os.environ.get("PROBE_WORKDIR", ".")).resolve()
MAX_STEPS = int(os.environ.get("PROBE_MAX_STEPS", "14"))
# `python` inside the shell must be the interpreter this probe runs under (pytest and matplotlib live there)
ENV = {**os.environ, "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}", "MPLBACKEND": "Agg"}

model = OpenAIChatModel(
    MODEL,
    provider=OpenAIProvider(base_url=BASE_URL, api_key=os.environ.get("NEBIUS_API_KEY", "mock")),
    profile=OpenAIModelProfile(openai_supports_strict_tool_definition=False, openai_chat_supports_max_completion_tokens=False),
)

TASK = """You are working in an empty project directory with Python 3.12, pytest and matplotlib installed.
Do these in order, using the tools (write files with write_file, run commands with run):
1. Write metrics.py with two functions: moving_average(xs, window) -> list[float] (simple trailing
   window, first window-1 values use the partial window) and percentile(xs, p) -> float (linear
   interpolation, p in 0..100). Raise ValueError on empty input or window < 1.
2. Write test_metrics.py with at least 5 pytest tests covering normal cases and the error cases.
3. Run `python -m pytest -q` and fix metrics.py or the tests until everything passes.
4. Write chart.py that builds a noisy series of 60 points, overlays its 5-point moving average,
   and saves chart.png (matplotlib, Agg backend, 900x450 px). Run it.
Then stop and summarise. Do not explain the code in prose; use the tools."""


class Session(BaseModel):
    summary: str = Field(description="two sentences: what was built and whether the tests pass")
    tests_passed: bool
    files_written: list[str]


steps: list[dict] = []
agent = Agent(
    model,
    output_type=Session,
    instructions="You are a careful coding agent. Prefer small, correct code. Always run the tests after writing them.",
    retries=2,
    model_settings=OpenAIChatModelSettings(openai_continuous_usage_stats=True, max_tokens=4000),
)


@agent.tool_plain
def write_file(path: str, content: str) -> str:
    """Write a UTF-8 text file at a relative path inside the project, creating parent folders."""
    if len(steps) >= MAX_STEPS:
        return "step budget exhausted; finish now"
    p = (WORK / path).resolve()
    if WORK not in p.parents and p != WORK:
        return "refused: path escapes the project"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    steps.append({"tool": "write_file", "path": path, "bytes": len(content.encode()), "ok": True})
    return f"wrote {path} ({len(content)} chars)"


@agent.tool_plain
def run(command: str) -> str:
    """Run a shell command in the project directory (60 s timeout). Returns exit code and the last 2000 chars of output."""
    if len(steps) >= MAX_STEPS:
        return "step budget exhausted; finish now"
    try:
        p = subprocess.run(command, shell=True, cwd=WORK, env=ENV, capture_output=True, text=True, timeout=60, check=False)
        out = (p.stdout + ("\n" + p.stderr if p.stderr else "")).strip()
        code = p.returncode
    except subprocess.TimeoutExpired:
        out, code = "timed out after 60 s", -1
    tail = out[-2000:]
    steps.append({"tool": "run", "command": command, "exit_code": code, "ok": code == 0, "tail": tail[-400:]})
    return f"exit {code}\n{tail}"


def _pytest_summary() -> dict:
    """Run the tests once more ourselves so the result does not depend on the model's self-report."""
    if not (WORK / "test_metrics.py").exists():
        return {"ran": False, "passed": 0, "failed": 0}
    p = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=WORK,
        env=ENV,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    text = p.stdout + p.stderr
    import re

    passed = sum(int(x) for x in re.findall(r"(\d+) passed", text))
    failed = sum(int(x) for x in re.findall(r"(\d+) failed", text)) + sum(int(x) for x in re.findall(r"(\d+) error", text))
    return {"ran": True, "passed": passed, "failed": failed, "exit_code": p.returncode, "tail": text.strip()[-600:]}


async def main() -> int:
    t0 = time.perf_counter()
    result = await agent.run(TASK)
    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    tests = _pytest_summary()
    files = sorted(
        str(p.relative_to(WORK))
        for p in WORK.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".png", ".txt", ".md")
    )
    chart = None
    cp = WORK / "chart.png"
    if cp.exists() and cp.stat().st_size < 400_000:
        chart = base64.b64encode(cp.read_bytes()).decode()
    u = result.usage
    out = {
        "framework": "coding_session",
        "model": MODEL,
        "tool_called": bool(steps),
        "steps": steps,
        "tests": tests,
        "files": files,
        "chart_png_b64": chart,
        "output": result.output.model_dump(),
        "usage": {"requests": u.requests, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens},
        "elapsed_ms": elapsed,
    }
    print("PROBE_RESULT " + json.dumps(out))
    return 0 if steps and tests.get("passed", 0) > 0 and tests.get("failed", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
