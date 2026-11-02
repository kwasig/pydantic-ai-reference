"""
Where the agent runs code it did not write itself: a Daytona sandbox in production, a local
temp dir in development. Same interface either way, so the agent's tools do not care.

    result = await get_sandbox(settings).run(files={"probe.py": "..."}, command="python probe.py", env={...})
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from .config import Settings

LogFn = Callable[[str], None]


@dataclass
class SandboxResult:
    backend: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    sandbox_id: str | None = None
    files: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class Sandbox(Protocol):
    name: str

    async def run(
        self, files: dict[str, str], command: str, env: dict[str, str] | None = None, timeout_s: int = 180, log: LogFn | None = None
    ) -> SandboxResult: ...


# ----------------------------------------------------------------------------- Daytona
class DaytonaSandbox:
    """Ephemeral Daytona sandbox per run: create → upload files → exec → delete."""

    name = "daytona"

    def __init__(self, api_key: str, image: str | None = None, keep: bool = False) -> None:
        self.api_key = api_key
        self.image = image or os.getenv("DAYTONA_IMAGE", "python:3.12-slim")
        self.keep = keep

    async def run(
        self, files: dict[str, str], command: str, env: dict[str, str] | None = None, timeout_s: int = 180, log: LogFn | None = None
    ) -> SandboxResult:
        from daytona import AsyncDaytona, CreateSandboxFromImageParams, DaytonaConfig

        log = log or (lambda _: None)
        t0 = time.perf_counter()
        async with AsyncDaytona(DaytonaConfig(api_key=self.api_key)) as client:
            log(f"daytona: creating sandbox from image {self.image}")
            sandbox = await client.create(
                CreateSandboxFromImageParams(
                    image=self.image, env_vars=env or {}, ephemeral=not self.keep, labels={"app": "nebius-fde-studio"}
                ),
                timeout=120,
            )
            sid = getattr(sandbox, "id", None)
            log(f"daytona: sandbox {sid} ready in {(time.perf_counter() - t0) * 1000:.0f} ms")
            try:
                work = await sandbox.get_work_dir() if hasattr(sandbox, "get_work_dir") else "/home/daytona"
                for name, content in files.items():
                    await sandbox.fs.upload_file(content.encode(), f"{work}/{name}")
                    log(f"daytona: uploaded {name} ({len(content)} bytes)")
                log(f"daytona: $ {command}")
                resp = await sandbox.process.exec(command, cwd=work, env=env, timeout=timeout_s)
                out = getattr(resp, "result", "") or ""
                code = int(getattr(resp, "exit_code", 0) or 0)
                for line in out.splitlines()[-40:]:
                    log(line)
                return SandboxResult(
                    "daytona", code, out, "", round((time.perf_counter() - t0) * 1000, 1), sandbox_id=sid, files=list(files)
                )
            finally:
                if not self.keep:
                    try:
                        await sandbox.delete()
                        log(f"daytona: sandbox {sid} deleted")
                    except Exception as e:  # noqa: BLE001
                        log(f"daytona: delete failed: {e}")


# ----------------------------------------------------------------------------- local
class LocalSandbox:
    """Temp directory + subprocess. Development only; opt in with ALLOW_LOCAL_EXEC=1."""

    name = "local"

    async def run(
        self, files: dict[str, str], command: str, env: dict[str, str] | None = None, timeout_s: int = 180, log: LogFn | None = None
    ) -> SandboxResult:
        log = log or (lambda _: None)
        return await asyncio.to_thread(self._run_sync, files, command, env or {}, timeout_s, log)

    def _run_sync(self, files: dict[str, str], command: str, env: dict[str, str], timeout_s: int, log: LogFn) -> SandboxResult:
        t0 = time.perf_counter()
        work = Path(tempfile.mkdtemp(prefix="nebius-sandbox-"))
        try:
            for name, content in files.items():
                p = work / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")
                log(f"local: wrote {name} ({len(content)} bytes)")
            # run with the same interpreter so the framework under test is importable
            cmd = command.replace("python ", f"{sys.executable} ", 1) if command.startswith("python ") else command
            full_env = {**os.environ, **env, "PYTHONDONTWRITEBYTECODE": "1", "PYDANTIC_AI_NO_BANNER": "1"}
            log(f"local: $ {command}")
            try:
                p = subprocess.run(cmd, shell=True, cwd=work, env=full_env, capture_output=True, text=True, timeout=timeout_s, check=False)
                out, err, code = p.stdout, p.stderr, p.returncode
            except subprocess.TimeoutExpired:
                out, err, code = "", f"timed out after {timeout_s}s", -1
            for line in (out + ("\n" + err if err else "")).splitlines()[-40:]:
                log(line)
            return SandboxResult(
                "local",
                code,
                out[-20000:],
                err[-8000:],
                round((time.perf_counter() - t0) * 1000, 1),
                sandbox_id=str(work),
                files=list(files),
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)


class DisabledSandbox:
    name = "disabled"

    async def run(self, files, command, env=None, timeout_s=180, log=None) -> SandboxResult:  # type: ignore[override]
        return SandboxResult(
            "disabled", -1, "", "No sandbox configured. Set DAYTONA_API_KEY (production) or ALLOW_LOCAL_EXEC=1 (development).", 0.0
        )


def get_sandbox(settings: Settings) -> Sandbox:
    if settings.daytona_api_key:
        return DaytonaSandbox(settings.daytona_api_key)
    if settings.allow_local_exec:
        return LocalSandbox()
    return DisabledSandbox()
