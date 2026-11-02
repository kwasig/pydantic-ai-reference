"""Start the mock Token Factory once per test session and build Settings pointed at it."""

from __future__ import annotations

import socket
import threading
import time

import httpx
import pytest
import uvicorn

from nebius_agent.config import Settings


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def mock_base_url() -> str:
    from mock_server.server import app

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(f"{url}/healthz", timeout=1).status_code == 200:
                break
        except Exception:
            time.sleep(0.05)
    else:  # pragma: no cover
        raise RuntimeError("mock server did not start")
    yield f"{url}/v1/"
    server.should_exit = True


@pytest.fixture
def settings(mock_base_url: str, tmp_path) -> Settings:
    return Settings(
        base_url=mock_base_url,
        api_key="mock",
        model="Qwen/Qwen3-235B-A22B-Instruct-2507",
        trace_exporter="memory",
        trace_file=str(tmp_path / "spans.jsonl"),
        allow_local_exec=True,
    )
