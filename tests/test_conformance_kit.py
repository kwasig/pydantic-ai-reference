"""The conformance kit must be green against the mock; against Token Factory it is the partner report."""

from __future__ import annotations

from openai import AsyncOpenAI

from conformance.checks import ALL_CHECKS, FAIL, PASS
from mock_server.server import example_from_schema


async def test_every_check_passes_on_mock(settings):
    client = AsyncOpenAI(base_url=settings.base_url, api_key="mock")
    results = [await c(client, settings.model) for c in ALL_CHECKS]
    failures = [r for r in results if r.status == FAIL]
    assert not failures, [(r.name, r.detail) for r in failures]
    assert sum(1 for r in results if r.status == PASS) == len(ALL_CHECKS)


def test_example_from_schema_handles_refs_enums_and_arrays():
    schema = {
        "$defs": {"Item": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}},
        "type": "object",
        "properties": {
            "kind": {"enum": ["a", "b"]},
            "items": {"type": "array", "items": {"$ref": "#/$defs/Item"}, "minItems": 2},
            "maybe": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "flag": {"type": "boolean"},
        },
        "required": ["kind", "items", "flag"],
    }
    ex = example_from_schema(schema)
    assert ex["kind"] == "a" and ex["flag"] is True
    assert len(ex["items"]) == 2 and ex["items"][0] == {"n": 1}
    assert isinstance(ex["maybe"], str)


class _FakeReasoningModel:
    """A chat.completions stand-in that spends a small budget on reasoning and only answers with room to think."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

        class _Completions:
            async def create(inner, **kw):  # noqa: N805
                self.calls.append(kw)
                from types import SimpleNamespace as NS

                starved = kw["max_tokens"] < 100
                msg = NS(content=None if starved else "ready", tool_calls=None, reasoning_content="thinking..." * 3, model_extra={})
                usage = NS(
                    prompt_tokens=12,
                    completion_tokens=kw["max_tokens"] if starved else 240,
                    completion_tokens_details=NS(reasoning_tokens=kw["max_tokens"] if starved else 231),
                    model_dump=lambda: {},
                )
                return NS(choices=[NS(message=msg, finish_reason="length" if starved else "stop")], usage=usage)

        self.chat = type("Chat", (), {"completions": _Completions()})()


async def test_reasoning_model_is_reported_not_failed():
    """Kimi-K3 spends max_tokens=20 on reasoning_content: the kit must retry with room and say so, not report FAIL."""
    from conformance.checks import chat_basic

    client = _FakeReasoningModel()
    r = await chat_basic(client, "fake/reasoner")
    assert r.status == PASS
    assert [c["max_tokens"] for c in client.calls] == [20, 2000]
    assert "reasoning model" in r.detail and "231 reasoning tokens" in r.detail
    assert r.evidence["reasoning_model"] is True
