"""
The agent as a service: how the runtime actually runs on Nebius.

    uvicorn nebius_agent.server:app --port 8080

POST /recommend          {"prompt": "...", "model": "optional"} -> ModelRecommendation (JSON)
POST /recommend/stream   same, free-text answer streamed as SSE
GET  /healthz            liveness for Kubernetes

Deployment shape (see deploy/): this container runs on Nebius Managed Kubernetes, talks to a
Token Factory *dedicated endpoint* for the model, uses Tavily for search, Daytona or ConTree
for code execution, and exports OpenTelemetry spans to Langfuse (self-hosted on the same
cluster or Langfuse Cloud). The agent process is stateless; scale it with replicas.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .agent import ModelRecommendation, build_agent, build_deps
from .config import get_settings


class RecommendRequest(BaseModel):
    prompt: str
    model: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    app.state.settings = s
    app.state.deps = build_deps(s)
    app.state.agent = build_agent(settings=s, output_mode="tool")
    app.state.text_agent = build_agent(settings=s, output_mode="text", instrument=False)
    yield
    await app.state.deps.http.aclose()


app = FastAPI(title="nebius-latency-scout", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "model": app.state.settings.model, "endpoint": app.state.settings.base_url}


@app.post("/recommend", response_model=ModelRecommendation)
async def recommend(req: RecommendRequest) -> ModelRecommendation:
    result = await app.state.agent.run(req.prompt, deps=app.state.deps, model=req.model)
    return result.output


@app.post("/recommend/stream")
async def recommend_stream(req: RecommendRequest) -> StreamingResponse:
    async def gen() -> AsyncIterator[bytes]:
        async with app.state.text_agent.run_stream(req.prompt, deps=app.state.deps, model=req.model) as result:
            async for delta in result.stream_text(delta=True, debounce_by=None):
                yield f"data: {json.dumps({'delta': delta})}\n\n".encode()
            u = result.usage
            yield f"data: {json.dumps({'done': True, 'input_tokens': u.input_tokens, 'output_tokens': u.output_tokens})}\n\n".encode()

    return StreamingResponse(gen(), media_type="text/event-stream")
