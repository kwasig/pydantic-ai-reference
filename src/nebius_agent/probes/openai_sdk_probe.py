"""Probe: raw OpenAI SDK tool calling against Token Factory (the wire-level baseline)."""

import json
import os
import time

from openai import OpenAI

BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
MODEL = os.environ.get("NEBIUS_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")
client = OpenAI(base_url=BASE_URL, api_key=os.environ.get("NEBIUS_API_KEY", "mock"))

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def main() -> int:
    t0 = time.perf_counter()
    messages = [{"role": "user", "content": "What is the weather in Amsterdam right now? Use get_weather."}]
    r1 = client.chat.completions.create(model=MODEL, messages=messages, tools=[TOOL], max_tokens=200)
    calls = r1.choices[0].message.tool_calls or []
    args = json.loads(calls[0].function.arguments) if calls else None
    answer = None
    if calls:
        messages.append(r1.choices[0].message.model_dump(exclude_none=True))
        messages.append({"role": "tool", "tool_call_id": calls[0].id, "content": json.dumps({"city": "Amsterdam", "temperature_c": 11.0})})
        r2 = client.chat.completions.create(model=MODEL, messages=messages, tools=[TOOL], max_tokens=200)
        answer = r2.choices[0].message.content
    out = {
        "framework": "openai_sdk",
        "model": MODEL,
        "tool_called": bool(calls),
        "tool_args": args,
        "output": answer,
        "usage": r1.usage.model_dump() if r1.usage else None,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    print("PROBE_RESULT " + json.dumps(out))
    return 0 if calls else 2


if __name__ == "__main__":
    raise SystemExit(main())
