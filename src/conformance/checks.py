"""
The reusable part of this repo.

Every agent framework (PydanticAI, LangGraph, CrewAI, OpenAI Agents SDK, Mastra...) sits on
the same handful of wire-level capabilities. Before we debug a framework integration we
want to know whether the *endpoint + model* actually supports each one. These checks use
the raw OpenAI client, so they are framework-independent, and each returns PASS / PARTIAL /
FAIL with evidence. Run them against any model on Token Factory (or the mock) with:

    nebius-agent verify --model <id> --json reports/<id>.json

The report is what we hand the next partner on day one, and what we attach to a
field-to-product ticket when something does not work.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

PASS, PARTIAL, FAIL, SKIP = "PASS", "PARTIAL", "FAIL", "SKIP"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    ms: float = 0.0


Check = Callable[[AsyncOpenAI, str], Awaitable[CheckResult]]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}, "unit": {"type": "string", "enum": ["c", "f"]}},
            "required": ["city"],
        },
    },
}
TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current local time for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
RECOMMENDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "model": {"type": "string"},
        "score": {"type": "integer", "minimum": 0, "maximum": 10},
        "reasons": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
    "required": ["model", "score", "reasons"],
    "additionalProperties": False,
}


async def _timed(name: str, fn: Callable[[], Awaitable[CheckResult]]) -> CheckResult:
    t0 = time.perf_counter()
    try:
        r = await fn()
    except Exception as e:  # noqa: BLE001 - we want every failure as a row, not a crash
        r = CheckResult(name, FAIL, f"{type(e).__name__}: {str(e)[:300]}")
    r.name = name
    r.ms = round((time.perf_counter() - t0) * 1000, 1)
    return r


# ----------------------------------------------------------------------------- reasoning models
# Reasoning models (Kimi-K3, DeepSeek-V4, MiniMax-M3, Qwen3 thinking...) generate `reasoning_content`
# before `content`. A check that asks for one word with max_tokens=20 then hits `finish_reason=length`
# with empty content, which looks like a failed chat but is a budget problem. Every check therefore goes
# through `_complete` / `_stream`: when the first attempt is starved this way, retry once with room to
# think and record it, so the report says "reasoning model: budget for N tokens" rather than FAIL.
REASONING_RETRY_TOKENS = 2000


def _reasoning_tokens(usage: Any) -> int:
    det = getattr(usage, "completion_tokens_details", None)
    return int(getattr(det, "reasoning_tokens", 0) or 0) if det is not None else 0


def _reasoning_text(obj: Any) -> str:
    return getattr(obj, "reasoning_content", None) or (getattr(obj, "model_extra", None) or {}).get("reasoning_content") or ""


def _starved(choice: Any) -> bool:
    msg = choice.message
    return choice.finish_reason == "length" and not (msg.content or "").strip() and not (msg.tool_calls or [])


def _note(note: dict[str, Any]) -> str:
    if not note.get("reasoning_model"):
        return ""
    if "retry_max_tokens" in note:
        return (
            f"; reasoning model: max_tokens={note['first_max_tokens']} was spent on reasoning (no content), "
            f"passed at max_tokens={note['retry_max_tokens']} with {note['reasoning_tokens']} reasoning tokens"
        )
    return f"; reasoning model: {note['reasoning_tokens']} reasoning tokens before the answer"


async def _complete(client: AsyncOpenAI, model: str, **kw: Any) -> tuple[Any, dict[str, Any]]:
    """One chat completion; if a reasoning model used the whole budget before answering, retry with room."""
    r = await client.chat.completions.create(model=model, **kw)
    ch = r.choices[0]
    note: dict[str, Any] = {}
    rt = _reasoning_tokens(r.usage) or len(_reasoning_text(ch.message).split())
    if _starved(ch):
        first = kw.get("max_tokens", 0)
        kw2 = {**kw, "max_tokens": max(REASONING_RETRY_TOKENS, first * 10)}
        r = await client.chat.completions.create(model=model, **kw2)
        rt2 = _reasoning_tokens(r.usage) or len(_reasoning_text(r.choices[0].message).split()) or rt
        note = {"reasoning_model": True, "first_max_tokens": first, "retry_max_tokens": kw2["max_tokens"], "reasoning_tokens": rt2}
    elif rt:
        note = {"reasoning_model": True, "reasoning_tokens": rt}
    return r, note


# ----------------------------------------------------------------------------- checks
async def chat_basic(client: AsyncOpenAI, model: str) -> CheckResult:
    async def go():
        r, note = await _complete(client, model, messages=[{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=20)
        text = (r.choices[0].message.content or "").strip()
        ok = bool(text)
        usage = r.usage.model_dump() if r.usage else {}
        return CheckResult(
            "",
            PASS if ok else FAIL,
            f"content={text[:40]!r}" + _note(note),
            {"usage": usage, "finish_reason": r.choices[0].finish_reason, **note},
        )

    return await _timed("chat.basic", go)


async def usage_reported(client: AsyncOpenAI, model: str) -> CheckResult:
    async def go():
        r, _ = await _complete(client, model, messages=[{"role": "user", "content": "Say hi"}], max_tokens=10)
        u = r.usage
        ok = u is not None and (u.prompt_tokens or 0) > 0 and (u.completion_tokens or 0) > 0
        return CheckResult(
            "", PASS if ok else FAIL, f"prompt={getattr(u, 'prompt_tokens', None)} completion={getattr(u, 'completion_tokens', None)}"
        )

    return await _timed("usage.reported", go)


async def tool_call_emitted(client: AsyncOpenAI, model: str) -> CheckResult:
    async def go():
        r, note = await _complete(
            client,
            model,
            messages=[{"role": "user", "content": "What is the weather in Amsterdam right now? Use the get_weather tool."}],
            tools=[WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=200,
        )
        msg = r.choices[0].message
        calls = msg.tool_calls or []
        if not calls:
            return CheckResult(
                "",
                FAIL,
                f"no tool_calls with tool_choice=auto; model answered in text: {(msg.content or '')[:70]!r}" + _note(note),
                {"content": (msg.content or "")[:200], "finish_reason": r.choices[0].finish_reason, **note},
            )
        c = calls[0]
        try:
            args = json.loads(c.function.arguments or "{}")
        except json.JSONDecodeError as e:
            return CheckResult("", FAIL, f"arguments not valid JSON: {e}", {"arguments": c.function.arguments})
        ok = c.function.name == "get_weather" and isinstance(args.get("city"), str)
        return CheckResult(
            "",
            PASS if ok else PARTIAL,
            f"{c.function.name}({args})" + _note(note),
            {"finish_reason": r.choices[0].finish_reason, "id": c.id, **note},
        )

    return await _timed("tools.call_emitted", go)


async def tool_result_roundtrip(client: AsyncOpenAI, model: str) -> CheckResult:
    async def go():
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "What is the weather in Amsterdam? Use the get_weather tool, then answer in one sentence."}
        ]
        r1, note = await _complete(client, model, messages=messages, tools=[WEATHER_TOOL], max_tokens=200)
        calls = r1.choices[0].message.tool_calls or []
        if not calls:
            return CheckResult("", FAIL, "first turn produced no tool call")
        messages.append(r1.choices[0].message.model_dump(exclude_none=True))
        messages.append(
            {"role": "tool", "tool_call_id": calls[0].id, "content": json.dumps({"city": "Amsterdam", "temp_c": 11, "sky": "overcast"})}
        )
        r2, note2 = await _complete(client, model, messages=messages, tools=[WEATHER_TOOL], max_tokens=200)
        note = note2 or note
        text = r2.choices[0].message.content or ""
        again = r2.choices[0].message.tool_calls or []
        if again and not text:
            return CheckResult("", PARTIAL, "model called the tool again instead of answering", {"calls": len(again)})
        ok = bool(text)
        return CheckResult("", PASS if ok else FAIL, f"final={text[:80]!r}" + _note(note), note)

    return await _timed("tools.result_roundtrip", go)


async def parallel_tool_calls(client: AsyncOpenAI, model: str) -> CheckResult:
    async def go():
        r, note = await _complete(
            client,
            model,
            messages=[{"role": "user", "content": "In parallel, get the weather AND the time in Amsterdam using both tools."}],
            tools=[WEATHER_TOOL, TIME_TOOL],
            max_tokens=300,
        )
        calls = r.choices[0].message.tool_calls or []
        names = sorted({c.function.name for c in calls})
        if len(calls) >= 2 and names == ["get_time", "get_weather"]:
            return CheckResult("", PASS, f"{len(calls)} calls in one turn: {names}" + _note(note), note)
        if calls:
            return CheckResult("", PARTIAL, f"only {len(calls)} call(s) in one turn: {names}; sequential calling still works")
        return CheckResult("", FAIL, "no tool calls")

    return await _timed("tools.parallel", go)


async def tool_choice_required(client: AsyncOpenAI, model: str) -> CheckResult:
    async def go():
        r, note = await _complete(
            client,
            model,
            messages=[{"role": "user", "content": "Hello there."}],
            tools=[WEATHER_TOOL],
            tool_choice="required",
            max_tokens=200,
        )
        msg = r.choices[0].message
        calls = msg.tool_calls or []
        detail = f"{len(calls)} call(s) with tool_choice=required"
        if not calls:
            detail += f"; finish={r.choices[0].finish_reason}, content={(msg.content or '')[:60]!r} (tool_choice ignored)"
        return CheckResult("", PASS if calls else FAIL, detail + _note(note), note)

    return await _timed("tools.choice_required", go)


async def tool_choice_named(client: AsyncOpenAI, model: str) -> CheckResult:
    async def go():
        r, note = await _complete(
            client,
            model,
            messages=[{"role": "user", "content": "Tell me about Amsterdam."}],
            tools=[WEATHER_TOOL, TIME_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_time"}},
            max_tokens=200,
        )
        msg = r.choices[0].message
        calls = msg.tool_calls or []
        ok = bool(calls) and calls[0].function.name == "get_time"
        detail = f"called {[c.function.name for c in calls]}"
        if not calls:
            detail += f"; finish={r.choices[0].finish_reason}, content={(msg.content or '')[:60]!r} (named tool_choice ignored)"
        return CheckResult("", PASS if ok else (PARTIAL if calls else FAIL), detail + _note(note), note)

    return await _timed("tools.choice_named", go)


async def structured_json_schema(client: AsyncOpenAI, model: str) -> CheckResult:
    async def go():
        r, note = await _complete(
            client,
            model,
            messages=[{"role": "user", "content": "Recommend one open model for a coding agent and score it 0-10 with reasons."}],
            response_format={"type": "json_schema", "json_schema": {"name": "recommendation", "schema": RECOMMENDATION_SCHEMA}},
            max_tokens=300,
        )
        text = (r.choices[0].message.content or "").strip()
        if not text:
            return CheckResult("", FAIL, f"empty content, finish={r.choices[0].finish_reason}" + _note(note), note)
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            return CheckResult("", FAIL, f"content is not JSON ({e.msg}): {text[:80]!r}" + _note(note), {"content": text[:300], **note})
        ok = (
            isinstance(obj.get("model"), str)
            and isinstance(obj.get("score"), int)
            and isinstance(obj.get("reasons"), list)
            and obj["reasons"]
        )
        return CheckResult("", PASS if ok else PARTIAL, f"keys={sorted(obj)}" + _note(note), {"object": obj, **note})

    return await _timed("structured.json_schema", go)


async def structured_json_object(client: AsyncOpenAI, model: str) -> CheckResult:
    async def go():
        r, note = await _complete(
            client,
            model,
            messages=[
                {
                    "role": "user",
                    "content": "Return a JSON object with keys 'answer' (string) and 'ok' (boolean) about whether Amsterdam is in Europe.",
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=200,
        )
        text = (r.choices[0].message.content or "").strip()
        if not text:
            return CheckResult("", FAIL, f"empty content, finish={r.choices[0].finish_reason}" + _note(note), note)
        obj = json.loads(text)
        return CheckResult("", PASS if isinstance(obj, dict) else FAIL, f"keys={sorted(obj)}" + _note(note), note)

    return await _timed("structured.json_object", go)


async def streaming_text(client: AsyncOpenAI, model: str) -> CheckResult:
    async def attempt(max_tokens: int) -> dict[str, Any]:
        chunks = reasoning_chunks = usage_chunks = 0
        text = ""
        finish = None
        usage = None
        usage_values: list[int] = []
        t0 = time.perf_counter()
        first = None
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Count from one to twenty in words."}],
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for ch in stream:
            if ch.usage:
                usage = ch.usage.model_dump()
                usage_chunks += 1
                usage_values.append(int(ch.usage.completion_tokens or 0))
            if ch.choices:
                d = ch.choices[0].delta
                if d and _reasoning_text(d):
                    reasoning_chunks += 1
                if d and d.content:
                    if first is None:
                        first = (time.perf_counter() - t0) * 1000
                    chunks += 1
                    text += d.content
                if ch.choices[0].finish_reason:
                    finish = ch.choices[0].finish_reason
        cumulative = usage_chunks > 1 and usage_values == sorted(usage_values)
        return {
            "chunks": chunks,
            "reasoning_chunks": reasoning_chunks,
            "text": text,
            "finish": finish,
            "first": first,
            "usage": usage,
            "usage_chunks": usage_chunks,
            "cumulative": cumulative,
            "max_tokens": max_tokens,
        }

    async def go():
        a = await attempt(120)
        note = ""
        if a["finish"] == "length" and (a["chunks"] == 0 or a["reasoning_chunks"]):
            # reasoning model spent (most of) the budget thinking: give it room and report that
            b = await attempt(REASONING_RETRY_TOKENS)
            note = (
                f"; reasoning model: max_tokens=120 was spent on {a['reasoning_chunks']} reasoning deltas "
                f"({a['chunks']} content chunks before the limit), "
                f"retried at max_tokens={REASONING_RETRY_TOKENS}: {b['reasoning_chunks']} reasoning deltas before the first content delta"
            )
            a = b
        elif a["reasoning_chunks"]:
            note = f"; reasoning model: {a['reasoning_chunks']} reasoning deltas before the first content delta"
        status = PASS if a["chunks"] > 1 and a["finish"] == "stop" else (PARTIAL if a["text"] else FAIL)
        # vLLM-style servers may send cumulative usage in every chunk; frameworks that add per-chunk
        # usage then over-count. Report it so the integration sets the right flag.
        usage_note = (
            "final chunk only"
            if a["usage_chunks"] == 1
            else (f"cumulative in {a['usage_chunks']} chunks" if a["cumulative"] else f"in {a['usage_chunks']} chunks")
        )
        return CheckResult(
            "",
            status,
            f"{a['chunks']} content chunks, finish={a['finish']}, ttft={a['first'] and round(a['first'], 1)}ms, usage {usage_note}" + note,
            {
                "usage": a["usage"],
                "usage_chunks": a["usage_chunks"],
                "usage_cumulative": a["cumulative"],
                "reasoning_chunks": a["reasoning_chunks"],
            },
        )

    return await _timed("streaming.text", go)


async def streaming_tool_call_deltas(client: AsyncOpenAI, model: str) -> CheckResult:
    async def attempt(max_tokens: int):
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What is the weather in Amsterdam? Use get_weather."}],
            tools=[WEATHER_TOOL],
            max_tokens=max_tokens,
            stream=True,
        )
        acc: dict[int, dict[str, str]] = {}
        finish = None
        reasoning_chunks = 0
        async for ch in stream:
            if not ch.choices:
                continue
            d = ch.choices[0].delta
            if d and _reasoning_text(d):
                reasoning_chunks += 1
            for tc in d.tool_calls or []:
                slot = acc.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
            if ch.choices[0].finish_reason:
                finish = ch.choices[0].finish_reason
        return acc, finish, reasoning_chunks

    async def go():
        acc, finish, rc = await attempt(200)
        note = ""
        if not acc and finish == "length":
            acc, finish, rc2 = await attempt(REASONING_RETRY_TOKENS)
            note = f"; reasoning model: max_tokens=200 was spent on {rc} reasoning deltas, retried at {REASONING_RETRY_TOKENS} ({rc2} reasoning deltas)"
        elif rc:
            note = f"; reasoning model: {rc} reasoning deltas before the call"
        if not acc:
            return CheckResult("", FAIL, f"no tool-call deltas in stream, finish={finish}" + note)
        call = acc[0]
        try:
            args = json.loads(call["arguments"])
        except json.JSONDecodeError as e:
            return CheckResult("", FAIL, f"reassembled arguments invalid JSON: {e}", {"arguments": call["arguments"]})
        ok = call["name"] == "get_weather" and "city" in args and finish == "tool_calls"
        return CheckResult("", PASS if ok else PARTIAL, f"{call['name']}({args}) finish={finish}" + note)

    return await _timed("streaming.tool_call_deltas", go)


async def streaming_parallel_tool_call_deltas(client: AsyncOpenAI, model: str) -> CheckResult:
    """Two tool calls in one streamed turn: every argument delta must land on its own tool-call index.
    Found live on Token Factory: with several calls in flight, a parser can hand the tail of call N's
    arguments to call N+1, so frameworks that reassemble by `index` get invalid JSON for both."""

    async def attempt(max_tokens: int):
        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Call get_weather for Amsterdam and get_time for Europe/Amsterdam, both now, in one turn."}
            ],
            tools=[WEATHER_TOOL, TIME_TOOL],
            max_tokens=max_tokens,
            stream=True,
        )
        acc: dict[int, dict[str, str]] = {}
        finish = None
        async for ch in stream:
            if not ch.choices:
                continue
            for tc in ch.choices[0].delta.tool_calls or []:
                slot = acc.setdefault(tc.index, {"name": "", "arguments": ""})
                if tc.function and tc.function.name:
                    slot["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
            if ch.choices[0].finish_reason:
                finish = ch.choices[0].finish_reason
        return acc, finish

    async def go():
        acc, finish = await attempt(300)
        note = ""
        if not acc and finish == "length":
            acc, finish = await attempt(REASONING_RETRY_TOKENS)
            note = f"; reasoning model: retried at max_tokens={REASONING_RETRY_TOKENS}"
        if len(acc) < 2:
            return CheckResult(
                "",
                PARTIAL,
                f"model emitted {len(acc)} call(s) in one streamed turn (finish={finish}); cannot judge delta routing" + note,
                {"calls": acc},
            )
        bad = []
        for i, c in sorted(acc.items()):
            try:
                json.loads(c["arguments"])
            except json.JSONDecodeError as e:
                bad.append(f"index {i} {c['name']}: {e.msg} in {c['arguments']!r}")
        if bad:
            return CheckResult("", FAIL, "argument deltas landed on the wrong tool-call index: " + "; ".join(bad)[:300], {"calls": acc})
        return CheckResult(
            "", PASS, f"{len(acc)} parallel calls, all argument streams valid JSON: {[c['name'] for c in acc.values()]}" + note
        )

    return await _timed("streaming.parallel_tool_call_deltas", go)


async def cached_tokens_reported(client: AsyncOpenAI, model: str) -> CheckResult:
    async def go():
        system = ("You are a careful assistant for Nebius Token Factory partners. Answer briefly and precisely. ") * 40  # ~1k tokens
        messages = [{"role": "system", "content": system}, {"role": "user", "content": "Reply with one word: cached"}]
        r1 = await client.chat.completions.create(model=model, messages=messages, max_tokens=5)
        r2 = await client.chat.completions.create(model=model, messages=messages, max_tokens=5)  # cache check: usage is all we need

        def cached(r):
            d = getattr(r.usage, "prompt_tokens_details", None)
            return None if d is None else (getattr(d, "cached_tokens", None) or 0)

        c1, c2 = cached(r1), cached(r2)
        if c2 is None:
            return CheckResult(
                "", PARTIAL, "no prompt_tokens_details in usage (prefix cache not observable)", {"prompt_tokens": r2.usage.prompt_tokens}
            )
        if c2 > 0:
            return CheckResult("", PASS, f"second call cached_tokens={c2} of prompt_tokens={r2.usage.prompt_tokens} (first={c1})")
        return CheckResult(
            "", PARTIAL, "prompt_tokens_details present but cached_tokens=0 on repeat (cache miss or model not cache-enabled)"
        )

    return await _timed("cache.cached_tokens", go)


ALL_CHECKS: list[Check] = [
    chat_basic,
    usage_reported,
    tool_call_emitted,
    tool_result_roundtrip,
    parallel_tool_calls,
    tool_choice_required,
    tool_choice_named,
    structured_json_schema,
    structured_json_object,
    streaming_text,
    streaming_tool_call_deltas,
    streaming_parallel_tool_call_deltas,
    cached_tokens_reported,
]
