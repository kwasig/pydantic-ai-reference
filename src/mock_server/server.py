"""
A small, deterministic OpenAI-compatible server that imitates Nebius Token Factory.

Purpose: let the reference agent, the conformance kit and the test-suite run end-to-end
with no network and no API key (CI, laptops on planes, partner demos without credits).
It is NOT a model. It is a scripted responder that exercises the same wire protocol
Token Factory speaks: /v1/models, /v1/chat/completions with tools, tool_choice,
response_format (json_schema / json_object), streaming SSE with tool-call deltas, and
usage blocks that include prompt_tokens_details.cached_tokens.

Behaviour (chosen so that an agent loop terminates and every conformance check is reachable):
  * tools present, no tool results yet  -> emit one tool call (two if the prompt says "parallel")
  * tool results present                -> if an output tool (e.g. PydanticAI's `final_result`)
                                           exists, call it with schema-valid args; else answer in text
  * response_format json_schema/object   -> emit a schema-valid JSON object
  * stream=true                          -> SSE chunks, tool-call deltas split across chunks,
                                           usage in the final chunk
  * "cached" tokens                      -> a prompt prefix seen before (>= 64 "tokens") reports
                                           cached_tokens > 0, like Token Factory's prefix cache

Run:  uvicorn mock_server.server:app --port 8008
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="mock-token-factory", version="0.1")

MODELS = [
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "deepseek-ai/DeepSeek-V3-0324",
    "meta-llama/Llama-3.3-70B-Instruct",
    "zai-org/GLM-4.5",
    "MiniMaxAI/MiniMax-M2",
    "openai/gpt-oss-120b",
]

_SEEN_PREFIXES: set[str] = set()
CACHE_MIN_TOKENS = 64
STREAM_DELAY_S = float(os.getenv("MOCK_STREAM_DELAY_S", "0.004"))


# --------------------------------------------------------------------------- helpers
def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _messages_text(messages: list[dict[str, Any]]) -> str:
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list):
            out.append(" ".join(p.get("text", "") for p in c if isinstance(p, dict)))
    return "\n".join(out)


def _cached_tokens(messages: list[dict[str, Any]]) -> tuple[int, int]:
    """Return (prompt_tokens, cached_tokens) imitating automatic prefix caching."""
    text = _messages_text(messages)
    prompt_tokens = _approx_tokens(text)
    # the "prefix" is everything but the last user turn
    prefix = _messages_text(messages[:-1]) if len(messages) > 1 else ""
    prefix_tokens = _approx_tokens(prefix) if prefix else 0
    cached = 0
    if prefix_tokens >= CACHE_MIN_TOKENS:
        key = hashlib.sha256(prefix.encode()).hexdigest()
        if key in _SEEN_PREFIXES:
            cached = prefix_tokens
        _SEEN_PREFIXES.add(key)
    return prompt_tokens, cached


def _resolve_ref(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    if "$ref" in schema:
        ref = schema["$ref"]
        if ref.startswith("#/"):
            node: Any = root
            for part in ref[2:].split("/"):
                node = node[part]
            return _resolve_ref(node, root)
    return schema


def example_from_schema(schema: dict[str, Any], root: dict[str, Any] | None = None, hint: str = "", depth: int = 0) -> Any:
    """Produce a schema-valid example value. Good enough to satisfy Pydantic validation."""
    root = root or schema
    schema = _resolve_ref(schema, root)
    if depth > 6:
        return None
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    if schema.get("examples"):
        return schema["examples"][0]
    if "default" in schema and schema["default"] is not None:
        return schema["default"]
    for key in ("anyOf", "oneOf"):
        if key in schema:
            options = [_resolve_ref(o, root) for o in schema[key]]
            non_null = [o for o in options if o.get("type") != "null"]
            return example_from_schema((non_null or options)[0], root, hint, depth + 1)
    if "allOf" in schema:
        merged: dict[str, Any] = {}
        for o in schema["allOf"]:
            merged.update(_resolve_ref(o, root))
        return example_from_schema(merged, root, hint, depth + 1)
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), t[0])
    if t == "object" or "properties" in schema:
        props = schema.get("properties", {})
        required = set(schema.get("required", list(props.keys())))
        return {k: example_from_schema(v, root, f"{hint} {k}".strip(), depth + 1) for k, v in props.items() if k in required or depth == 0}
    if t == "array":
        item = example_from_schema(schema.get("items", {"type": "string"}), root, hint, depth + 1)
        n = max(1, int(schema.get("minItems", 1)))
        return [item] * n
    if t == "integer":
        return int(schema.get("minimum", 1))
    if t == "number":
        return float(schema.get("minimum", 1.0))
    if t == "boolean":
        return True
    if t == "string":
        fmt = schema.get("format")
        if fmt == "date-time":
            return "2026-11-01T00:00:00Z"
        desc = (schema.get("description") or "").lower()
        if "model" in hint.lower() or "model" in desc:
            return MODELS[0]
        if "query" in hint.lower() or "prompt" in hint.lower():
            return "nebius token factory"
        return f"mock {hint}".strip() or "mock"
    return "mock"


def _tool_by_name(tools: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for t in tools:
        if t.get("function", {}).get("name") == name:
            return t
    return None


def _is_output_tool(name: str) -> bool:
    return name.startswith("final_result") or name in {"final_result", "output"}


def _pick_tools(tools: list[dict[str, Any]], prompt: str, tool_choice: Any) -> list[dict[str, Any]]:
    fns = [t for t in tools if t.get("type") == "function"]
    if isinstance(tool_choice, dict):
        name = tool_choice.get("function", {}).get("name")
        t = _tool_by_name(fns, name)
        return [t] if t else fns[:1]
    plain = [t for t in fns if not _is_output_tool(t["function"]["name"])]
    if not plain:
        return fns[:1]
    low = prompt.lower()
    wanted = _ordered_wanted(plain, prompt)
    chosen = wanted[:1] or plain[:1]
    if "parallel" in low and len(plain) >= 2:
        chosen = plain[:2]
    return chosen


# the order a sensible agent would call things in; the mock walks this list for tools the prompt asks for
_ORDER = ["list_models", "run_probe", "run_conformance", "benchmark", "benchmark_model", "web_search", "run_python"]
_KEYWORDS = {
    "list_models": ("list", "models", "served", "which model"),
    "run_probe": ("probe", "prove", "verify", "framework", "integration", "pydantic", "langgraph", "crewai"),
    "run_conformance": ("conformance", "readiness", "agent-ready", "kit", "check by check", "rely on"),
    "benchmark": ("benchmark", "measure", "latency", "time-to-first-token", "ttft"),
    "benchmark_model": ("benchmark", "measure", "latency"),
    "web_search": ("search the web", "look up", "latest news"),
    "run_python": ("python", "run code", "calculate"),
}


def _ordered_wanted(plain: list[dict[str, Any]], prompt: str) -> list[dict[str, Any]]:
    by_name = {t["function"]["name"]: t for t in plain}
    ordered = [by_name[n] for n in _ORDER if n in by_name and _wanted(n, prompt)]
    ordered += [t for t in plain if t not in ordered and _wanted(t["function"]["name"], prompt)]
    return ordered


def _wanted(tool_name: str, prompt: str) -> bool:
    low = prompt.lower()
    if tool_name.lower() in low:
        return True
    return any(k in low for k in _KEYWORDS.get(tool_name, ()))


def _make_tool_calls(chosen: list[dict[str, Any]], prompt: str) -> list[dict[str, Any]]:
    calls = []
    for t in chosen:
        fn = t["function"]
        params = fn.get("parameters") or {"type": "object", "properties": {}}
        args = example_from_schema(params, params, hint=fn["name"])
        if not isinstance(args, dict):
            args = {}
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": fn["name"], "arguments": json.dumps(args)},
            }
        )
    return calls


def _final_text(messages: list[dict[str, Any]]) -> str:
    results = [m for m in messages if m.get("role") == "tool"]
    user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    q = user.get("content") if user and isinstance(user.get("content"), str) else "your request"
    if results:
        snippets = []
        for r in results:
            c = r.get("content")
            c = c if isinstance(c, str) else json.dumps(c)
            snippets.append(c[:200])
        return f"Mock Token Factory answer for: {q}\nI used {len(results)} tool result(s): " + " | ".join(snippets)
    return f"Mock Token Factory answer for: {q}. Streaming, tools and structured output all work here."


def _json_for_response_format(rf: dict[str, Any], messages: list[dict[str, Any]]) -> str:
    if rf.get("type") == "json_schema":
        schema = rf.get("json_schema", {}).get("schema") or rf.get("schema") or {}
        return json.dumps(example_from_schema(schema, schema))
    # json_object: no schema on the wire, but "prompted" output modes embed the schema in the
    # system prompt / instructions. Find it and honour it, like a well-behaved model would.
    schema = _schema_from_prompt(messages)
    if schema:
        return json.dumps(example_from_schema(schema, schema))
    return json.dumps({"answer": _final_text(messages), "ok": True})


def _schema_from_prompt(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for m in messages:
        if m.get("role") not in ("system", "developer", "user"):
            continue
        text = m.get("content") if isinstance(m.get("content"), str) else _messages_text([m])
        start = text.find("{")
        while start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict) and ("properties" in obj or "$defs" in obj):
                                return obj
                        except json.JSONDecodeError:
                            pass
                        break
            start = text.find("{", start + 1)
    return None


def _decide(body: dict[str, Any]) -> dict[str, Any]:
    """Return {'tool_calls': [...]} or {'content': str}."""
    messages: list[dict[str, Any]] = body.get("messages", [])
    tools = body.get("tools") or []
    tool_choice = body.get("tool_choice", "auto")
    rf = body.get("response_format")
    prompt = _messages_text([m for m in messages if m.get("role") == "user"])
    has_tool_results = any(m.get("role") == "tool" for m in messages)
    has_assistant_calls = any(m.get("role") == "assistant" and m.get("tool_calls") for m in messages)

    if rf and rf.get("type") in ("json_schema", "json_object") and not tools:
        return {"content": _json_for_response_format(rf, messages)}

    if tools and tool_choice != "none":
        fns = [t for t in tools if t.get("type") == "function"]
        output_tools = [t for t in fns if _is_output_tool(t["function"]["name"])]
        plain = [t for t in fns if not _is_output_tool(t["function"]["name"])]
        if not has_tool_results and not has_assistant_calls and plain:
            return {"tool_calls": _make_tool_calls(_pick_tools(tools, prompt, tool_choice), prompt)}
        # multi-step: keep calling tools the user asked for by name/keyword until each has run once
        called = {tc["function"]["name"] for m in messages if m.get("role") == "assistant" for tc in (m.get("tool_calls") or [])}
        pending = [t for t in _ordered_wanted(plain, prompt) if t["function"]["name"] not in called]
        if pending and len(called) < 4:
            return {"tool_calls": _make_tool_calls(pending[:1], prompt)}
        if output_tools:
            return {"tool_calls": _make_tool_calls(output_tools[:1], prompt)}
        if not plain and tool_choice == "required":
            return {"tool_calls": _make_tool_calls(fns[:1], prompt)}
        if not has_tool_results and plain and isinstance(tool_choice, dict):
            return {"tool_calls": _make_tool_calls(_pick_tools(tools, prompt, tool_choice), prompt)}
    if rf and rf.get("type") in ("json_schema", "json_object"):
        return {"content": _json_for_response_format(rf, messages)}
    return {"content": _final_text(messages)}


# --------------------------------------------------------------------------- routes
@app.get("/v1/models")
@app.get("/models")
async def list_models() -> JSONResponse:
    now = int(time.time())
    return JSONResponse({"object": "list", "data": [{"id": m, "object": "model", "created": now, "owned_by": "mock"} for m in MODELS]})


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", MODELS[0])
    messages = body.get("messages", [])
    prompt_tokens, cached = _cached_tokens(messages)
    decision = _decide(body)
    created = int(time.time())
    cid = f"chatcmpl-{uuid.uuid4().hex[:16]}"

    if body.get("stream"):
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        return StreamingResponse(
            _stream(cid, created, model, decision, prompt_tokens, cached, include_usage),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Mock-Token-Factory": "1"},
        )

    if "tool_calls" in decision:
        message = {"role": "assistant", "content": None, "tool_calls": decision["tool_calls"]}
        finish = "tool_calls"
        completion_tokens = sum(_approx_tokens(tc["function"]["arguments"]) for tc in decision["tool_calls"])
    else:
        message = {"role": "assistant", "content": decision["content"]}
        finish = "stop"
        completion_tokens = _approx_tokens(decision["content"])

    return JSONResponse(
        {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish, "logprobs": None}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "prompt_tokens_details": {"cached_tokens": cached},
            },
        }
    )


async def _stream(
    cid: str, created: int, model: str, decision: dict[str, Any], prompt_tokens: int, cached: int, include_usage: bool
) -> AsyncIterator[bytes]:
    def chunk(delta: dict[str, Any], finish: str | None = None, usage: dict[str, Any] | None = None) -> bytes:
        payload: dict[str, Any] = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}],
        }
        if usage is not None:
            payload["usage"] = usage
        return f"data: {json.dumps(payload)}\n\n".encode()

    yield chunk({"role": "assistant", "content": ""})
    completion_tokens = 0
    if "tool_calls" in decision:
        for i, tc in enumerate(decision["tool_calls"]):
            args = tc["function"]["arguments"]
            completion_tokens += _approx_tokens(args)
            # first delta carries id + name, following deltas carry argument fragments
            yield chunk(
                {
                    "tool_calls": [
                        {"index": i, "id": tc["id"], "type": "function", "function": {"name": tc["function"]["name"], "arguments": ""}}
                    ]
                }
            )
            step = max(1, len(args) // 4)
            for j in range(0, len(args), step):
                yield chunk({"tool_calls": [{"index": i, "function": {"arguments": args[j : j + step]}}]})
        finish = "tool_calls"
    else:
        text = decision["content"]
        completion_tokens = _approx_tokens(text)
        words = text.split(" ")
        for k in range(0, len(words), 3):
            piece = " ".join(words[k : k + 3])
            if k + 3 < len(words):
                piece += " "
            await asyncio.sleep(STREAM_DELAY_S)  # a little decode latency so TTFT / tokens-per-s are measurable
            yield chunk({"content": piece})
        finish = "stop"
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": cached},
    }
    yield chunk({}, finish=finish, usage=usage if include_usage else None)
    if include_usage:
        yield chunk({}, finish=None, usage=usage)
    yield b"data: [DONE]\n\n"
