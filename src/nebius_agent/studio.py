"""
Nebius FDE Studio: the web UI and run manager around the Studio agent.

    nebius-agent studio            # http://127.0.0.1:8090

POST /api/runs                      start a run  {"prompt" | "scenario", "model"?}  -> {"run_id"}
GET  /api/runs/{id}/events          Server-Sent Events: everything the agent does, as it happens
GET  /api/runs/{id}                 the run's final state (report, usage, trace summary)
GET  /api/status                    which services are live (Token Factory, Daytona, Cartesia, Tavily, Langfuse)
GET  /api/models                    models served by the configured endpoint
GET  /api/scenarios                 the preset scenarios
GET  /audio/{file}                  Cartesia output
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .config import Settings, get_settings
from .dedicated import DedicatedClient, dedicated_allowed
from .provider import make_client
from .studio_agent import SCENARIOS, build_studio_agent, build_studio_deps, integration_snippet
from .tracing import langfuse_configured, langfuse_host, langfuse_trace_url, memory_spans
from .voice import synthesize

STATIC = Path(__file__).parent / "static"
# Stateless mode (serverless hosts such as Vercel): one request runs the scout and streams it; media is
# embedded inline instead of served from disk; nothing is written outside /tmp; saving recordings is off.
STATELESS = os.getenv("STUDIO_STATELESS", "").lower() in ("1", "true", "yes")
_REPO = Path(__file__).resolve().parents[2]
_SCRATCH = Path(os.getenv("STUDIO_SCRATCH", "/tmp/nebius-scout")) if STATELESS else Path(".")
AUDIO_DIR = Path(os.getenv("STUDIO_AUDIO_DIR", str(_SCRATCH / "audio") if STATELESS else "audio"))
ARTIFACT_DIR = Path(os.getenv("STUDIO_ARTIFACT_DIR", str(_SCRATCH / "artifacts") if STATELESS else "artifacts"))


class Intake(BaseModel):
    """What the partner tells us. The prompt is composed from it on the client; we keep it for the README."""

    workload: str | None = None
    framework: str | None = None  # pydantic_ai | langgraph | openai_sdk
    budget_ms: int | None = None
    monthly_tokens_m: float | None = None
    partner: str | None = None


class StartRun(BaseModel):
    prompt: str | None = None
    scenario: str | None = None
    model: str | None = None
    intake: Intake | None = None
    replay: str | None = None  # name of a saved run under demo/runs to replay instead of executing


@dataclass
class Run:
    id: str
    prompt: str
    model: str
    intake: dict[str, Any] = field(default_factory=dict)
    started: float = field(default_factory=time.time)
    status: str = "running"  # running | done | error
    events: list[dict[str, Any]] = field(default_factory=list)
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    report: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    error: str | None = None
    audio: dict[str, Any] | None = None
    spans: list[dict[str, Any]] = field(default_factory=list)
    trace_id: str | None = None
    langfuse: dict[str, Any] | None = None
    artifacts: list[str] = field(default_factory=list)

    def emit(self, kind: str, data: dict[str, Any] | None = None) -> None:
        ev = {"seq": len(self.events), "t": round(time.time() - self.started, 3), "kind": kind, **(data or {})}
        self.events.append(ev)
        self.queue.put_nowait(ev)


RUNS: dict[str, Run] = {}
REPLAY_DIR = Path(
    os.getenv("STUDIO_REPLAY_DIR")
    or (str(_REPO / "demo" / "runs") if (_REPO / "demo" / "runs").exists() and not Path("demo/runs").exists() else "demo/runs")
)


def list_replays() -> list[dict[str, Any]]:
    out = []
    for p in sorted(REPLAY_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            out.append(
                {
                    "name": p.stem,
                    "title": d.get("title") or p.stem,
                    "model": d.get("model"),
                    "recorded_at": d.get("recorded_at"),
                    "events": len(d.get("events", [])),
                }
            )
        except Exception:  # noqa: BLE001, S112 - a corrupt saved run just does not appear in the list
            continue
    return out


async def _replay(run: Run, name: str, speed: float = 2.0, max_gap: float = 1.5) -> None:
    """Re-emit a saved run's events with (compressed) original timing. Used for demos without network."""
    p = REPLAY_DIR / f"{Path(name).stem}.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    run.report, run.usage, run.audio, run.spans = d.get("report"), d.get("usage"), d.get("audio"), d.get("spans") or []
    last = 0.0
    for ev in d["events"]:
        gap = min(max(0.0, float(ev.get("t", 0)) - last) / speed, max_gap)
        last = float(ev.get("t", 0))
        if gap:
            await asyncio.sleep(gap)
        data = {k: v for k, v in ev.items() if k not in ("seq", "t")}
        kind = data.pop("kind")
        if kind == "run_start":
            data["replay"] = name
            if d.get("services"):
                data["services"] = d["services"]
        run.emit(kind, data)
    run.status = "done"


def _span_dict(span) -> dict[str, Any]:
    ctx = span.get_span_context()
    parent = span.parent.span_id if span.parent else None
    attrs = dict(span.attributes or {})
    keep = {
        k: v
        for k, v in attrs.items()
        if k.startswith("gen_ai.") and not k.endswith(("messages", "content", "input.messages", "output.messages"))
    }
    return {
        "name": span.name,
        "trace_id": f"{ctx.trace_id:032x}",
        "span_id": f"{ctx.span_id:016x}",
        "parent_span_id": f"{parent:016x}" if parent else None,
        "start_ns": span.start_time,
        "duration_ms": round(((span.end_time or 0) - (span.start_time or 0)) / 1e6, 1),
        "attributes": keep,
    }


async def _execute(run: Run, settings: Settings) -> None:
    # The scout itself runs on the verified default model; `run.model` is the model under evaluation.
    try:
        exporter = "memory+otlp" if langfuse_configured() else "memory"
        agent = build_studio_agent(settings, settings.model, instrument=True, trace_exporter=exporter)
        deps = build_studio_deps(settings, emit=run.emit, eval_model=run.model)

        def save_artifact(name: str, data: bytes) -> str:
            if STATELESS:  # another instance may answer the next request: ship the bytes inline
                return "data:image/png;base64," + base64.b64encode(data).decode()
            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            fname = f"{run.id}-{Path(name).name}"
            (ARTIFACT_DIR / fname).write_bytes(data)
            run.artifacts.append(fname)
            return f"/artifacts/{fname}"

        deps.save_artifact = save_artifact
    except Exception as e:  # noqa: BLE001 - setup failures must still end the stream
        run.status, run.error = "error", f"{type(e).__name__}: {str(e)[:500]}"
        run.emit("error", {"message": run.error})
        run.emit("run_end", {"status": run.status, "elapsed_s": 0})
        return
    run.emit(
        "run_start",
        {"prompt": run.prompt, "model": run.model, "endpoint": settings.base_url, "sandbox": deps.sandbox.name, "mock": settings.is_mock},
    )
    try:
        # The orchestrator deliberately does NOT stream its own model responses: Token Factory's streamed
        # tool-call deltas can land on the wrong tool-call index when the model emits several calls at
        # once (see the kit's streaming.parallel_tool_call_deltas check). Non-streamed requests are
        # unaffected; the partner-facing legs (benchmarks, probes) still stream.
        from pydantic_ai import Agent as _Agent
        from pydantic_ai.usage import UsageLimits

        async with agent.iter(run.prompt, deps=deps, usage_limits=UsageLimits(request_limit=14)) as agent_run:
            async for node in agent_run:
                if _Agent.is_model_request_node(node):
                    continue  # executed by the iterator, without streaming
                if _Agent.is_call_tools_node(node):
                    for part in node.model_response.parts:
                        if type(part).__name__ == "TextPart" and getattr(part, "content", ""):
                            run.emit("text", {"delta": part.content})
                    async with node.stream(agent_run.ctx) as tool_events:
                        async for ev in tool_events:
                            k = ev.event_kind
                            if k == "function_tool_call":
                                run.emit(
                                    "tool_call",
                                    {"tool": ev.part.tool_name, "args": _safe_json(ev.part.args), "call_id": ev.part.tool_call_id},
                                )
                            elif k == "function_tool_result":
                                run.emit(
                                    "tool_result",
                                    {
                                        "tool": ev.part.tool_name,
                                        "call_id": ev.part.tool_call_id,
                                        "result": _trim(_safe_json(ev.part.content)),
                                    },
                                )
                            elif k == "final_result":
                                run.emit("final_result_started", {"tool": ev.tool_name})
            result = agent_run.result
        report = result.output.model_dump()
        u = result.usage
        run.report = report
        run.usage = {
            "requests": u.requests,
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_read_tokens": getattr(u, "cache_read_tokens", 0) or 0,
        }
        # deliverables derived deterministically from what was verified, not from the model's prose
        verified = [p for p in report.get("probes", []) if p.get("tool_called")]
        fw = run.intake.get("framework") or (_framework_key(verified[0]["framework"]) if verified else "pydantic_ai")
        report["integration_snippet"] = integration_snippet(fw, report["recommended_model"], settings.base_url)
        report["framework_key"] = fw
        report["budget_ms"] = run.intake.get("budget_ms")
        best = next((m for m in report.get("measurements", []) if m["model"] == report["recommended_model"]), None)
        report["budget_met_measured"] = (best["ttft_ms"] <= run.intake["budget_ms"]) if (best and run.intake.get("budget_ms")) else None
        report["cost"] = _cost_estimate(report["recommended_model"], run.intake.get("monthly_tokens_m"))
        report["readme_md"] = _partner_readme(run, report, settings)
        run.emit("report", {"report": report, "usage": run.usage})
        # speak the summary
        run.emit("voice", {"status": "start", "text": report["spoken_summary"]})
        voice = await synthesize(report["spoken_summary"], settings, out_dir=str(AUDIO_DIR))
        audio_url = f"/audio/{Path(voice.path).name}" if voice.path else None
        if STATELESS and voice.path and Path(voice.path).exists():
            audio_url = "data:audio/mpeg;base64," + base64.b64encode(Path(voice.path).read_bytes()).decode()
        run.audio = {
            "mode": voice.mode,
            "url": audio_url,
            "text": voice.text,
            "model": voice.model,
            "elapsed_ms": voice.elapsed_ms,
            "error": voice.error,
        }
        run.emit("voice", {"status": "done", **run.audio})
        run.status = "done"
    except Exception as e:  # noqa: BLE001
        run.status = "error"
        run.error = f"{type(e).__name__}: {str(e)[:500]}"
        run.emit("error", {"message": run.error})
    finally:
        try:
            run.spans = [
                _span_dict(s)
                for s in memory_spans()
                if s.start_time
                and s.start_time / 1e9 >= run.started - 1
                and s.name.split(" ")[0] in ("invoke_agent", "chat", "execute_tool", "running", "agent")
            ]
        except Exception:  # noqa: BLE001
            run.spans = []
        root = next((sp for sp in run.spans if not sp.get("parent_span_id")), None)
        run.trace_id = root["trace_id"] if root else None
        project = getattr(app.state, "langfuse_project", None)
        url = None
        if run.trace_id and langfuse_configured():
            url = f"{langfuse_host()}/project/{project}/traces/{run.trace_id}" if project else langfuse_trace_url(run.trace_id)
        lf = {"exported": langfuse_configured(), "host": langfuse_host() if langfuse_configured() else None, "url": url}
        run.langfuse = lf
        run.emit("trace", {"spans": run.spans, "count": len(run.spans), "trace_id": run.trace_id, "langfuse": lf})
        run.emit("run_end", {"status": run.status, "elapsed_s": round(time.time() - run.started, 2)})
        await deps.http.aclose()


def _framework_key(label: str) -> str:
    low = label.lower()
    if "langgraph" in low:
        return "langgraph"
    if "openai" in low:
        return "openai_sdk"
    return "pydantic_ai"


def load_pricing() -> dict[str, dict[str, float]]:
    """Optional per-model prices, USD per 1M tokens: pricing.json next to the repo root or PRICING_FILE.
    Empty by default on purpose: fill it from your account's pricing page; we never invent prices."""
    path = Path(os.getenv("PRICING_FILE", "pricing.json"))
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if isinstance(v, dict) and "input_per_m" in v and "output_per_m" in v}
    except (OSError, json.JSONDecodeError):
        return {}


def _cost_estimate(model: str, monthly_tokens_m: float | None) -> dict[str, Any] | None:
    prices = load_pricing().get(model)
    if not prices or not monthly_tokens_m:
        return None
    # assume a 3:1 input:output split unless the partner tells us otherwise
    inp, out = monthly_tokens_m * 0.75, monthly_tokens_m * 0.25
    monthly = inp * prices["input_per_m"] + out * prices["output_per_m"]
    return {
        "monthly_usd": round(monthly, 2),
        "input_per_m": prices["input_per_m"],
        "output_per_m": prices["output_per_m"],
        "assumed_split": "75% input / 25% output",
        "monthly_tokens_m": monthly_tokens_m,
    }


def _partner_readme(run: Run, r: dict[str, Any], s: Settings) -> str:
    probes = (
        "\n".join(
            f"- {p['framework']} on {p['backend']}: {'tool call verified' if p['tool_called'] else 'NOT verified'} — {p['summary']}"
            for p in r.get("probes", [])
        )
        or "- no probe run"
    )
    meas = (
        "\n".join(
            f"| {m['model']} | {m['ttft_ms']} | {m['tokens_per_s']} | {m['cached_prompt_tokens']} |" for m in r.get("measurements", [])
        )
        or "| – | – | – | – |"
    )
    conf = r.get("conformance")
    conf_line = f"{conf['passed']} pass · {conf['partial']} partial · {conf['failed']} fail ({conf['notes']})" if conf else "not run"
    budget = (
        f"{r['budget_ms']} ms TTFT budget: {'met' if r.get('budget_met_measured') else 'not met on the shared endpoint' if r.get('budget_met_measured') is False else 'not measured'}"
        if r.get("budget_ms")
        else "no latency budget given"
    )
    cost = r.get("cost")
    cost_line = (
        f"≈ ${cost['monthly_usd']:,.0f} / month at {cost['monthly_tokens_m']}M tokens ({cost['assumed_split']})"
        if cost
        else "add prices to pricing.json for an estimate"
    )
    return f"""# {r["title"]}

Generated by Nebius Integration Scout · {datetime.now(UTC).strftime("%Y-%m-%d")} · endpoint {s.base_url}

**Verdict:** {"integration verified" if r["integration_verified"] else "not verified"} · recommended model `{r["recommended_model"]}` · workload {r["workload"]} · {budget}

## What was proven
{probes}

Conformance kit: {conf_line}

## Measured on your prompt
| model | TTFT ms | tokens/s | prefix-cache hit (tokens) |
|---|---|---|---|
{meas}

## Findings
{chr(10).join("- " + f for f in r["findings"])}

## Next steps
{chr(10).join("- " + f for f in r["next_steps"])}

## Cost
{cost_line}

## Integration
```python
{r["integration_snippet"]}
```

## Your ask
> {run.prompt}
"""


def _safe_json(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    if hasattr(v, "model_dump"):
        return v.model_dump()
    try:
        json.dumps(v)
        return v
    except TypeError:
        return str(v)


def _trim(v: Any, limit: int = 4000) -> Any:
    s = json.dumps(v, default=str)
    return v if len(s) <= limit else {"truncated": True, "preview": s[:limit]}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = app.state.settings = get_settings()
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    if dedicated_allowed() and settings.api_key and not settings.is_mock:
        # never leave a scout-* endpoint billing after a crash
        try:
            c = DedicatedClient(settings.api_key)
            swept = await c.sweep(log=print)
            await c.aclose()
            if swept:
                print(f"dedicated: swept {len(swept)} leftover endpoint(s): {', '.join(swept)}")
        except Exception as e:  # noqa: BLE001
            print(f"dedicated: sweep failed: {e}")
    yield


app = FastAPI(title="Nebius FDE Studio", version="0.2.0", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))


@app.get("/static/{name}")
async def static_file(name: str) -> FileResponse:
    p = STATIC / Path(name).name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/api/scenarios")
async def scenarios() -> list[dict[str, str]]:
    return SCENARIOS


@app.get("/api/status")
async def status() -> dict[str, Any]:
    s: Settings = app.state.settings
    tf = {"configured": bool(s.api_key) or s.is_mock, "endpoint": s.base_url, "model": s.model, "mock": s.is_mock, "reachable": None}
    try:
        client = make_client(s)
        page = await asyncio.wait_for(client.models.list(), timeout=8)
        tf["reachable"] = True
        tf["models"] = len(page.data)
    except Exception as e:  # noqa: BLE001
        tf["reachable"] = False
        tf["error"] = f"{type(e).__name__}: {str(e)[:160]}"
    return {
        "token_factory": tf,
        "daytona": {"configured": bool(s.daytona_api_key), "fallback_local": s.allow_local_exec},
        "cartesia": {"configured": bool(os.getenv("CARTESIA_API_KEY"))},
        "tavily": {"configured": bool(s.tavily_api_key)},
        "langfuse": {"configured": langfuse_configured(), "host": langfuse_host()},
        "dedicated": {"allowed": dedicated_allowed() and not STATELESS},
        "stateless": STATELESS,
        "trace_exporter": s.trace_exporter,
    }


@app.get("/api/langfuse/metrics")
async def langfuse_metrics(hours: int = 72, limit: int = 100) -> dict[str, Any]:
    """Read back what Langfuse holds for this service: recent traces, per-trace latency, model calls.
    This is the other half of the integration: spans go out over OTLP, numbers come back over the
    public API (v2 observations, grouped by trace here)."""
    if not langfuse_configured():
        return {"configured": False}
    auth = (os.environ["LANGFUSE_PUBLIC_KEY"].strip('"'), os.environ["LANGFUSE_SECRET_KEY"].strip('"'))
    since = (datetime.now(tz=UTC) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    async with httpx.AsyncClient(base_url=langfuse_host(), auth=auth, timeout=20) as c:
        r = await c.get("/api/public/v2/observations", params={"fromStartTime": since, "limit": limit})
        if r.status_code != 200:
            return {"configured": True, "host": langfuse_host(), "error": f"{r.status_code}: {r.text[:160]}"}
        obs = r.json().get("data", [])
    project = next((o.get("projectId") for o in obs if o.get("projectId")), None)
    if project:
        app.state.langfuse_project = project
    by_trace: dict[str, dict[str, Any]] = {}
    for o in obs:
        t = by_trace.setdefault(
            o["traceId"],
            {
                "id": o["traceId"],
                "start": o["startTime"],
                "end": o.get("endTime") or o["startTime"],
                "generations": 0,
                "models": set(),
                "gen_latency_s": 0.0,
                "name": None,
            },
        )
        t["start"] = min(t["start"], o["startTime"])
        t["end"] = max(t["end"], o.get("endTime") or o["startTime"])
        if o.get("isRootObservation"):
            t["name"] = o.get("name")
        if o.get("type") == "GENERATION":
            t["generations"] += 1
            t["gen_latency_s"] += float(o.get("latency") or 0)
            if (o.get("name") or "").startswith("chat "):
                t["models"].add(o["name"][5:])
    traces = []
    for t in sorted(by_trace.values(), key=lambda x: x["start"], reverse=True):
        if not t["generations"]:  # HTTP/SDK spans without a model call are not agent runs
            continue
        try:
            lat = (datetime.fromisoformat(t["end"]) - datetime.fromisoformat(t["start"])).total_seconds()
        except ValueError:
            lat = None
        traces.append(
            {
                "id": t["id"],
                "name": t["name"] or "nebius-fde-studio",
                "start": t["start"],
                "latency_s": round(lat, 2) if lat is not None else None,
                "generations": t["generations"],
                "gen_latency_s": round(t["gen_latency_s"], 2),
                "models": sorted(t["models"]),
                "url": f"{langfuse_host()}/project/{project}/traces/{t['id']}" if project else langfuse_trace_url(t["id"]),
            }
        )
    lat = sorted(x["latency_s"] for x in traces if x["latency_s"] is not None)
    gens = sum(x["generations"] for x in traces)
    return {
        "configured": True,
        "host": langfuse_host(),
        "project": project,
        "hours": hours,
        "observations": len(obs),
        "count": len(traces),
        "model_calls": gens,
        "avg_latency_s": round(sum(lat) / len(lat), 2) if lat else None,
        "p95_latency_s": lat[max(0, int(len(lat) * 0.95) - 1)] if lat else None,
        "traces": traces[:12],
    }


@app.get("/api/dedicated")
async def dedicated_info() -> dict[str, Any]:
    """What the control plane says: templates we could deploy and endpoints that exist. Read-only."""
    s: Settings = app.state.settings
    if not s.api_key or s.is_mock:
        return {"allowed": dedicated_allowed(), "templates": [], "endpoints": [], "note": "no Token Factory key"}
    c = DedicatedClient(s.api_key)
    try:
        try:
            templates = await c.templates()
        except Exception as e:  # noqa: BLE001
            templates = [{"error": f"{type(e).__name__}: {str(e)[:200]}"}]
        try:
            endpoints = await c.list()
        except Exception as e:  # noqa: BLE001
            endpoints = [{"error": f"{type(e).__name__}: {str(e)[:200]}"}]
    finally:
        await c.aclose()
    return {"allowed": dedicated_allowed(), "templates": templates, "endpoints": endpoints}


@app.get("/api/dedicated/plan")
async def dedicated_plan(model: str) -> dict[str, Any]:
    """Dry run: what provision_dedicated_endpoint would create for this model. No side effects."""
    s: Settings = app.state.settings
    if not s.api_key or s.is_mock:
        return {"model": model, "template": None}
    c = DedicatedClient(s.api_key)
    try:
        tpl = await c.find_template(model)
        return {"model": model, "template": (tpl or {}).get("name"), "config": DedicatedClient.cheapest_config(tpl) if tpl else None}
    finally:
        await c.aclose()


@app.get("/api/pricing")
async def pricing() -> dict[str, Any]:
    return {"models": load_pricing(), "source": os.getenv("PRICING_FILE", "pricing.json")}


@app.get("/api/models")
async def models() -> dict[str, Any]:
    s: Settings = app.state.settings
    try:
        page = await asyncio.wait_for(make_client(s).models.list(), timeout=10)
        return {"models": [m.id for m in page.data], "default": s.model}
    except Exception as e:  # noqa: BLE001
        return {"models": [s.model], "default": s.model, "error": str(e)[:200]}


@app.post("/api/runs")
async def start_run(req: StartRun) -> dict[str, str]:
    s: Settings = app.state.settings
    if req.replay:
        if not (REPLAY_DIR / f"{Path(req.replay).stem}.json").exists():
            raise HTTPException(404, "no such saved run")
        run = Run(id=uuid.uuid4().hex[:10], prompt="(replay)", model=req.model or s.model, intake={"replay": req.replay})
        RUNS[run.id] = run
        asyncio.create_task(_replay(run, req.replay))
        return {"run_id": run.id}
    prompt = req.prompt
    if req.scenario:
        sc = next((x for x in SCENARIOS if x["id"] == req.scenario), None)
        if not sc:
            raise HTTPException(404, "unknown scenario")
        prompt = sc["prompt"]
    if not prompt:
        raise HTTPException(400, "prompt or scenario required")
    run = Run(
        id=uuid.uuid4().hex[:10],
        prompt=prompt,
        model=req.model or s.model,
        intake=req.intake.model_dump(exclude_none=True) if req.intake else {},
    )
    RUNS[run.id] = run
    asyncio.create_task(_execute(run, s))
    return {"run_id": run.id}


@app.get("/api/runs")
async def list_runs() -> dict[str, Any]:
    return {
        "runs": [
            {"id": r.id, "status": r.status, "model": r.model, "started": r.started, "title": (r.report or {}).get("title")}
            for r in RUNS.values()
        ],
        "replays": list_replays(),
    }


@app.get("/api/runs/{run_id}/export")
async def export_run(run_id: str, save_as: str | None = None) -> dict[str, Any]:
    """Everything needed to replay this run later: events, report, usage, spans, and the audio file name.
    With ?save_as=<name> it is also written to demo/runs/<name>.json (audio copied alongside)."""
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "no such run")
    doc = {
        "title": (run.report or {}).get("title"),
        "model": run.model,
        "prompt": run.prompt,
        "intake": run.intake,
        "recorded_at": datetime.fromtimestamp(run.started, tz=UTC).isoformat(),
        "events": run.events,
        "report": run.report,
        "usage": run.usage,
        "audio": run.audio,
        "spans": run.spans,
        "services": await status(),
    }
    if save_as and STATELESS:
        doc["saved"] = None
        doc["note"] = "saving recordings is disabled on a stateless host"
    elif save_as:
        REPLAY_DIR.mkdir(parents=True, exist_ok=True)
        name = Path(save_as).stem
        (REPLAY_DIR / f"{name}.json").write_text(json.dumps(doc, default=str, indent=1), encoding="utf-8")
        audio_url = (run.audio or {}).get("url") or ""
        audio_path = AUDIO_DIR / Path(audio_url).name if audio_url else None
        if audio_path and audio_path.exists():
            shutil.copy(audio_path, REPLAY_DIR / audio_path.name)
        for fname in run.artifacts:
            if (ARTIFACT_DIR / fname).exists():
                shutil.copy(ARTIFACT_DIR / fname, REPLAY_DIR / fname)
        doc["saved"] = str(REPLAY_DIR / f"{name}.json")
    return doc


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "no such run")
    return {
        "id": run.id,
        "status": run.status,
        "prompt": run.prompt,
        "model": run.model,
        "report": run.report,
        "usage": run.usage,
        "audio": run.audio,
        "error": run.error,
        "events": len(run.events),
        "spans": run.spans,
    }


@app.post("/api/runs/stream")
async def run_and_stream(req: StartRun) -> StreamingResponse:
    """Start a run (or a replay) and stream its events in the same response. This is the only path that
    works on a serverless host, where the next request may hit a different instance."""
    s: Settings = app.state.settings
    if req.replay:
        if not (REPLAY_DIR / f"{Path(req.replay).stem}.json").exists():
            raise HTTPException(404, "no such saved run")
        run = Run(id=uuid.uuid4().hex[:10], prompt="(replay)", model=req.model or s.model, intake={"replay": req.replay})
        task = asyncio.create_task(_replay(run, req.replay))
    else:
        prompt = req.prompt
        if req.scenario:
            sc = next((x for x in SCENARIOS if x["id"] == req.scenario), None)
            if not sc:
                raise HTTPException(404, "unknown scenario")
            prompt = sc["prompt"]
        if not prompt:
            raise HTTPException(400, "prompt or scenario required")
        run = Run(
            id=uuid.uuid4().hex[:10],
            prompt=prompt,
            model=req.model or s.model,
            intake=req.intake.model_dump(exclude_none=True) if req.intake else {},
        )
        task = asyncio.create_task(_execute(run, s))
    RUNS[run.id] = run

    async def gen() -> AsyncIterator[bytes]:
        idx = 0
        yield f": run {run.id}\n\n".encode()
        while True:
            while idx < len(run.events):
                ev = run.events[idx]
                idx += 1
                yield f"event: {ev['kind']}\ndata: {json.dumps(ev, default=str)}\n\n".encode()
                if ev["kind"] == "run_end":
                    return
            try:
                await asyncio.wait_for(run.queue.get(), timeout=10)
            except TimeoutError:
                yield b": keepalive\n\n"
            if task.done() and idx >= len(run.events):
                return

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/runs/{run_id}/events")
async def run_events(run_id: str, since: int = 0) -> StreamingResponse:
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "no such run")

    async def gen() -> AsyncIterator[bytes]:
        # replay what already happened, then follow live
        idx = since
        while True:
            while idx < len(run.events):
                ev = run.events[idx]
                idx += 1
                yield f"event: {ev['kind']}\ndata: {json.dumps(ev, default=str)}\n\n".encode()
                if ev["kind"] == "run_end":
                    return
            try:
                await asyncio.wait_for(run.queue.get(), timeout=15)
            except TimeoutError:
                yield b": keepalive\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/artifacts/{name}")
async def artifact(name: str) -> FileResponse:
    p = ARTIFACT_DIR / Path(name).name
    if not p.exists():
        p = REPLAY_DIR / Path(name).name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/audio/{name}")
async def audio(name: str) -> FileResponse:
    p = AUDIO_DIR / Path(name).name
    if not p.exists():
        p = REPLAY_DIR / Path(name).name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="audio/mpeg")
