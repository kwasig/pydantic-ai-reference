"""The HTTP service wraps the agent; this is what runs on Nebius Managed Kubernetes."""

from __future__ import annotations

import json

import httpx
import pytest

from nebius_agent import config


@pytest.fixture
async def service(settings, monkeypatch):
    # the service reads Settings from the environment; point it at the mock
    monkeypatch.setenv("NEBIUS_BASE_URL", settings.base_url)
    monkeypatch.setenv("NEBIUS_API_KEY", "mock")
    monkeypatch.setenv("NEBIUS_MODEL", settings.model)
    monkeypatch.setenv("TRACE_EXPORTER", "none")
    config.get_settings.cache_clear()
    from nebius_agent.server import app

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://svc") as client:
            yield client
    config.get_settings.cache_clear()


async def test_healthz(service):
    r = await service.get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok"


async def test_recommend_returns_structured_output(service):
    r = await service.post("/recommend", json={"prompt": "Pick a model for a coding agent"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["recommended_model"] and isinstance(body["latency_budget_met"], bool)


async def test_recommend_stream_emits_sse(service):
    events = []
    async with service.stream("POST", "/recommend/stream", json={"prompt": "What can you do?"}) as r:
        assert r.status_code == 200
        async for line in r.aiter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    assert any("delta" in e for e in events)
    assert events[-1].get("done") is True
