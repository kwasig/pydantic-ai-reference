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
