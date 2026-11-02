"""The Studio: start a run through the API, follow its SSE stream, check the agent really did things."""

from __future__ import annotations

import json

import httpx
import pytest

from nebius_agent import config


@pytest.fixture
async def studio(settings, monkeypatch, tmp_path):
    monkeypatch.setenv("NEBIUS_BASE_URL", settings.base_url)
    monkeypatch.setenv("NEBIUS_API_KEY", "mock")
    monkeypatch.setenv("NEBIUS_MODEL", settings.model)
    monkeypatch.setenv("ALLOW_LOCAL_EXEC", "1")
    monkeypatch.setenv("DAYTONA_API_KEY", "")
    monkeypatch.setenv("CARTESIA_API_KEY", "")
    monkeypatch.setenv("TRACE_EXPORTER", "memory")
    monkeypatch.setenv("STUDIO_AUDIO_DIR", str(tmp_path / "audio"))
    monkeypatch.setenv("STUDIO_REPLAY_DIR", str(tmp_path / "replays"))
    monkeypatch.setenv("STUDIO_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    config.get_settings.cache_clear()
    import importlib

    import nebius_agent.studio as studio_mod

    importlib.reload(studio_mod)
    app = studio_mod.app
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://studio", timeout=120) as client:
            yield client
    config.get_settings.cache_clear()


async def test_status_and_scenarios(studio):
    st = (await studio.get("/api/status")).json()
    assert st["token_factory"]["reachable"] is True and st["token_factory"]["mock"] is True
    assert st["daytona"]["fallback_local"] is True
    sc = (await studio.get("/api/scenarios")).json()
    assert any(s["id"] == "voice-cartesia" for s in sc)
    assert (await studio.get("/")).status_code == 200


async def test_full_run_streams_events_and_verifies_integration(studio):
    run_id = (
        await studio.post(
            "/api/runs",
            json={
                "prompt": "Prove the pydantic_ai probe works in the sandbox, run the conformance kit on the model, and benchmark it on a support reply within a 300 ms budget.",
                "intake": {"workload": "voice", "framework": "pydantic_ai", "budget_ms": 300, "monthly_tokens_m": 500},
            },
        )
    ).json()["run_id"]
    kinds: list[str] = []
    payloads: dict[str, list[dict]] = {}
    async with studio.stream("GET", f"/api/runs/{run_id}/events") as r:
        assert r.status_code == 200
        async for line in r.aiter_lines():
            if line.startswith("data: "):
                ev = json.loads(line[6:])
                kinds.append(ev["kind"])
                payloads.setdefault(ev["kind"], []).append(ev)
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    assert "tool_call" in kinds and "tool_result" in kinds
    sandbox_done = [e for e in payloads["sandbox"] if e["status"] == "done"]
    assert sandbox_done and sandbox_done[0]["tool_called"] is True, "the probe must prove the tool call inside the sandbox"
    assert any(e["status"] == "done" and e["passed"] == 13 for e in payloads["conformance"])
    assert any(e["status"] == "done" and e["ttft_ms"] > 0 for e in payloads["benchmark"])
    report = payloads["report"][0]["report"]
    assert report["integration_verified"] is True and report["spoken_summary"]
    assert "OpenAIChatModel" in report["integration_snippet"] and report["framework_key"] == "pydantic_ai"
    assert report["budget_ms"] == 300 and report["budget_met_measured"] is True
    assert report["readme_md"].startswith("# ") and "## Integration" in report["readme_md"]
    assert report["cost"] is None  # pricing.json is empty on purpose
    voice = [e for e in payloads["voice"] if e["status"] == "done"][0]
    assert voice["mode"] in ("browser", "cartesia")
    assert payloads["trace"][0]["count"] >= 3
    final = (await studio.get(f"/api/runs/{run_id}")).json()
    assert final["status"] == "done" and final["report"]["recommended_model"]


async def test_export_and_replay_round_trip(studio):
    run_id = (await studio.post("/api/runs", json={"scenario": "model-readiness"})).json()["run_id"]
    async with studio.stream("GET", f"/api/runs/{run_id}/events") as r:
        async for line in r.aiter_lines():
            if line.startswith("event: run_end"):
                break
    doc = (await studio.get(f"/api/runs/{run_id}/export", params={"save_as": "readiness"})).json()
    assert doc["saved"].endswith("readiness.json") and doc["report"] and len(doc["events"]) > 5
    assert any(x["name"] == "readiness" for x in (await studio.get("/api/runs")).json()["replays"])

    replay_id = (await studio.post("/api/runs", json={"replay": "readiness"})).json()["run_id"]
    kinds = []
    async with studio.stream("GET", f"/api/runs/{replay_id}/events") as r:
        async for line in r.aiter_lines():
            if line.startswith("event: "):
                kinds.append(line[7:])
            if line.startswith("event: run_end"):
                break
    assert kinds[0] == "run_start" and "report" in kinds and kinds[-1] == "run_end"
    assert (await studio.get(f"/api/runs/{replay_id}")).json()["report"]["title"] == doc["report"]["title"]


async def test_coding_session_runs_in_the_sandbox_and_returns_a_chart(studio):
    run_id = (await studio.post("/api/runs", json={"scenario": "coding-daytona"})).json()["run_id"]
    sessions = []
    async with studio.stream("GET", f"/api/runs/{run_id}/events") as r:
        async for line in r.aiter_lines():
            if line.startswith("data: "):
                ev = json.loads(line[6:])
                if ev["kind"] == "sandbox" and ev.get("status") == "done" and ev.get("probe") == "coding_session":
                    sessions.append(ev)
            if line.startswith("event: run_end"):
                break
    assert sessions, "the coding-agent session did not run"
    sess = sessions[0]
    assert sess["tool_called"] and len(sess["steps"]) >= 4
    assert sess["tests_passed"] and sess["tests"]["passed"] >= 5
    assert "chart.png" in sess["files"] and sess["chart_url"]
    png = await studio.get(sess["chart_url"])
    assert png.status_code == 200 and png.content[:8] == b"\x89PNG\r\n\x1a\n"
