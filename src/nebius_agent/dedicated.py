"""
Token Factory dedicated endpoints: provision, wait for readiness, benchmark, tear down.

Control plane (https://api.tokenfactory.nebius.com):
    GET    /v0/dedicated_endpoints/templates   what can be deployed
    POST   /v0/dedicated_endpoints             create -> {endpoint_id, routing_key}
    GET    /v0/dedicated_endpoints             list (status + readiness live here)
    PATCH  /v0/dedicated_endpoints/{id}        scaling / enabled
    DELETE /v0/dedicated_endpoints/{id}

Billing starts when the endpoint reports `ready`, not while it provisions, and stops when it has no
replicas. The scout therefore: names everything `scout-...`, deletes in a `finally`, and sweeps
leftovers on startup. Nothing here runs unless ALLOW_DEDICATED=1.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

CONTROL_PLANE = os.getenv("NEBIUS_CONTROL_PLANE", "https://api.tokenfactory.nebius.com").rstrip("/")
SCOUT_PREFIX = "scout-"


def dedicated_allowed() -> bool:
    return os.getenv("ALLOW_DEDICATED", "").lower() in ("1", "true", "yes")


@dataclass
class DedicatedEndpoint:
    endpoint_id: str
    routing_key: str
    name: str
    model_name: str
    gpu_type: str
    gpu_count: int
    region: str
    base_url: str
    created_at: float = field(default_factory=time.time)
    ready_at: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def billable_minutes(self) -> float:
        if not self.ready_at:
            return 0.0
        return round((time.time() - self.ready_at) / 60, 1)


def _pick(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _flatten_status(item: dict[str, Any]) -> tuple[str, str]:
    """Return (lifecycle, readiness) as lower-case strings from whatever shape the API uses."""
    status = _pick(item, "status", "state", "lifecycle", default={})
    if isinstance(status, dict):
        life = str(_pick(status, "lifecycle", "state", "phase", "status", default="")).lower()
        ready = str(_pick(status, "readiness", "ready", "readiness_status", default="")).lower()
    else:
        life = str(status).lower()
        ready = str(_pick(item, "readiness", "readiness_status", "ready", default="")).lower()
    return life, ready


class DedicatedClient:
    def __init__(self, api_key: str, control_plane: str = CONTROL_PLANE) -> None:
        self.api_key = api_key
        self.control_plane = control_plane
        self._h = httpx.AsyncClient(base_url=control_plane, headers={"Authorization": f"Bearer {api_key}"}, timeout=60)

    async def aclose(self) -> None:
        await self._h.aclose()

    async def templates(self) -> list[dict[str, Any]]:
        r = await self._h.get("/v0/dedicated_endpoints/templates")
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else _pick(data, "templates", "items", "data", default=[])

    async def list(self) -> list[dict[str, Any]]:
        r = await self._h.get("/v0/dedicated_endpoints")
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else _pick(data, "endpoints", "items", "data", default=[])

    async def find_template(self, model_name: str) -> dict[str, Any] | None:
        tpls = await self.templates()
        want = model_name.lower()
        exact = [t for t in tpls if str(_pick(t, "model_name", "name", "model", default="")).lower() == want]
        if exact:
            return exact[0]
        loose = [t for t in tpls if want.split("/")[-1] in str(_pick(t, "model_name", "name", "model", default="")).lower()]
        return loose[0] if loose else None

    @staticmethod
    def cheapest_config(tpl: dict[str, Any], flavor_name: str | None = None) -> dict[str, Any]:
        """Pick the smallest GPU configuration a template allows (fewest GPUs, then cheaper GPU class)."""
        flavors = tpl.get("flavors") or {}
        if isinstance(flavors, list):
            flavors = {str(_pick(f, "name", "flavor_name", default=i)): f for i, f in enumerate(flavors)}
        fname = flavor_name if flavor_name in flavors else ("base" if "base" in flavors else (next(iter(flavors), "base")))
        flavor = flavors.get(fname) or {}
        gpus = ((flavor.get("available_configurations") or {}).get("gpu_configurations")) or {}
        rank = {"gpu-l40s-d": 0, "gpu-h100-sxm": 1, "gpu-h200-sxm": 2, "gpu-b200-sxm": 3, "gpu-b300-sxm": 4}
        best = None
        for gpu_type, cfg in gpus.items():
            counts = sorted(cfg.get("allowed_gpu_counts") or [1])
            regions = cfg.get("allowed_regions") or []
            key = (counts[0], rank.get(gpu_type, 9))
            if best is None or key < best[0]:
                best = (key, gpu_type, counts[0], regions[0] if regions else None)
        if best is None:
            return {"flavor_name": fname, "gpu_type": "gpu-h100-sxm", "gpu_count": 1, "region": None}
        return {"flavor_name": fname, "gpu_type": best[1], "gpu_count": best[2], "region": best[3]}

    async def create(
        self,
        model_name: str,
        *,
        gpu_type: str | None = None,
        gpu_count: int | None = None,
        region: str | None = None,
        flavor_name: str | None = None,
        name: str | None = None,
    ) -> DedicatedEndpoint:
        tpl = await self.find_template(model_name) or {}
        cfg = self.cheapest_config(tpl, flavor_name)
        body = {
            "name": name or f"{SCOUT_PREFIX}{model_name.split('/')[-1].lower()[:24]}-{int(time.time()) % 100000}",
            "description": "Integration Scout: temporary endpoint for a latency measurement; auto-deleted",
            "model_name": _pick(tpl, "model_name", "name", default=model_name),
            "flavor_name": cfg["flavor_name"],
            "gpu_type": gpu_type or cfg["gpu_type"],
            "gpu_count": int(gpu_count or cfg["gpu_count"]),
            "region": region or cfg["region"] or os.getenv("NEBIUS_DEDICATED_REGION", "us-central1"),
            "scaling": {"min_replicas": 1, "max_replicas": 1},
        }
        r = await self._h.post("/v0/dedicated_endpoints", json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"create failed {r.status_code}: {r.text[:300]}")
        d = r.json()
        region = body["region"]
        base_url = os.getenv("NEBIUS_DEDICATED_BASE_URL") or f"https://api.tokenfactory.{region}.nebius.com/v1"
        return DedicatedEndpoint(
            endpoint_id=str(_pick(d, "endpoint_id", "id")),
            routing_key=str(_pick(d, "routing_key", "model", default="")),
            name=body["name"],
            model_name=body["model_name"],
            gpu_type=body["gpu_type"],
            gpu_count=body["gpu_count"],
            region=region,
            base_url=base_url,
            raw=d,
        )

    async def status(self, endpoint_id: str) -> tuple[str, str, dict[str, Any]]:
        for item in await self.list():
            if str(_pick(item, "endpoint_id", "id")) == endpoint_id:
                life, ready = _flatten_status(item)
                return life, ready, item
        return "missing", "", {}

    async def wait_ready(self, ep: DedicatedEndpoint, timeout_s: int = 900, poll_s: int = 10, log=None) -> bool:
        """Poll until readiness says ready, then confirm with one real chat call (404 while routing is not live)."""
        t0 = time.time()
        last = ""
        seen_raw = False
        while time.time() - t0 < timeout_s:
            life, ready, raw = await self.status(ep.endpoint_id)
            if raw and not seen_raw and log:
                seen_raw = True
                log(
                    "dedicated: status fields "
                    + ", ".join(f"{k}={str(v)[:40]}" for k, v in raw.items() if k not in ("name", "description"))
                )
            if (life, ready) != last and log:
                log(f"dedicated: {ep.name} lifecycle={life or '?'} readiness={ready or '?'} ({int(time.time() - t0)} s)")
                last = (life, ready)
            if ready.startswith("ready") or (not ready and life in ("running", "ready")):
                if await self._probe(ep):
                    ep.ready_at = time.time()
                    return True
            if life in ("error", "failed"):
                return False
            await asyncio.sleep(poll_s)
        return False

    async def _probe(self, ep: DedicatedEndpoint) -> bool:
        try:
            async with httpx.AsyncClient(base_url=ep.base_url, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=30) as h:
                r = await h.post(
                    "/chat/completions", json={"model": ep.routing_key, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}
                )
                return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def delete(self, endpoint_id: str) -> bool:
        r = await self._h.delete(f"/v0/dedicated_endpoints/{endpoint_id}")
        if r.status_code in (200, 202, 204, 404):
            return True
        # fall back to stopping it so it stops billing even if delete is refused
        r2 = await self._h.patch(f"/v0/dedicated_endpoints/{endpoint_id}", json={"enabled": False})
        return r2.status_code < 400

    async def sweep(self, older_than_s: int = 0, log=None) -> list[str]:
        """Delete every endpoint the scout created. Called on Studio startup and after every run."""
        gone = []
        for item in await self.list():
            name = str(_pick(item, "name", default=""))
            if name.startswith(SCOUT_PREFIX):
                eid = str(_pick(item, "endpoint_id", "id"))
                if await self.delete(eid):
                    gone.append(name)
                    if log:
                        log(f"dedicated: swept leftover endpoint {name}")
        return gone
