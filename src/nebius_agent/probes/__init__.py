"""
Framework probes: small, self-contained scripts the agent runs inside a sandbox to PROVE that a
framework's tool calling works against Token Factory, instead of assuming it does.

Each probe reads NEBIUS_BASE_URL / NEBIUS_API_KEY / NEBIUS_MODEL from the environment, defines a
tool, runs one agent turn, and prints a single line:

    PROBE_RESULT {"framework": ..., "tool_called": true, "tool_args": {...}, "output": ..., "usage": {...}, "elapsed_ms": ...}

Exit code 0 means the tool was really called through the framework's abstractions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).parent


@dataclass(frozen=True)
class Probe:
    key: str
    framework: str
    file: str
    install: str  # pip command run inside a fresh sandbox (local sandbox skips it)
    command: str
    notes: str

    @property
    def source(self) -> str:
        return (_HERE / self.file).read_text(encoding="utf-8")


PROBES: dict[str, Probe] = {
    "pydantic_ai": Probe(
        key="pydantic_ai",
        framework="PydanticAI",
        file="pydantic_ai_probe.py",
        install="pip install -q 'pydantic-ai-slim[openai]>=2.60'",
        command="python pydantic_ai_probe.py",
        notes="typed tool + typed output; tool-mode structured output",
    ),
    "openai_sdk": Probe(
        key="openai_sdk",
        framework="OpenAI SDK (raw chat.completions)",
        file="openai_sdk_probe.py",
        install="pip install -q 'openai>=1.50'",
        command="python openai_sdk_probe.py",
        notes="the wire-level baseline every framework sits on",
    ),
    "coding_session": Probe(
        key="coding_session",
        framework="Coding-agent session (PydanticAI, files + shell)",
        file="coding_agent_probe.py",
        install="pip install -q 'pydantic-ai-slim[openai]>=2.60' pytest matplotlib",
        command="python coding_agent_probe.py",
        notes="the agent writes a module and tests, runs pytest to green, renders a chart; everything happens inside the sandbox",
    ),
    "langgraph": Probe(
        key="langgraph",
        framework="LangGraph (create_react_agent)",
        file="langgraph_probe.py",
        install="pip install -q langgraph langchain-openai",
        command="python langgraph_probe.py",
        notes="ReAct agent through langchain-openai's ChatOpenAI pointed at Token Factory",
    ),
}


def parse_probe_result(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith("PROBE_RESULT "):
            try:
                return json.loads(line[len("PROBE_RESULT ") :])
            except json.JSONDecodeError:
                return None
    return None
