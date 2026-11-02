"""Probe: LangGraph ReAct agent via langchain-openai pointed at Token Factory."""

import json
import os
import time

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
MODEL = os.environ.get("NEBIUS_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")

calls: list[dict] = []


@tool
def get_weather(city: str) -> dict:
    """Current weather for a city."""
    calls.append({"city": city})
    return {"city": city, "temperature_c": 11.0, "sky": "overcast"}


def main() -> int:
    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=os.environ.get("NEBIUS_API_KEY", "mock"), temperature=0)
    agent = create_react_agent(llm, [get_weather])
    t0 = time.perf_counter()
    state = agent.invoke({"messages": [("user", "What is the weather in Amsterdam right now? Use get_weather.")]})
    final = state["messages"][-1].content
    out = {
        "framework": "langgraph",
        "model": MODEL,
        "tool_called": bool(calls),
        "tool_args": calls[0] if calls else None,
        "output": final if isinstance(final, str) else str(final),
        "usage": getattr(state["messages"][-1], "usage_metadata", None),
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    print("PROBE_RESULT " + json.dumps(out, default=str))
    return 0 if calls else 2


if __name__ == "__main__":
    raise SystemExit(main())
