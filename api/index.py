"""Vercel entry point: exposes the Studio's FastAPI app as a serverless function (stateless mode)."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("STUDIO_STATELESS", "1")
os.environ.setdefault("TRACE_EXPORTER", "memory")
os.environ.setdefault("ALLOW_LOCAL_EXEC", "0")

from nebius_agent.studio import app  # noqa: E402

__all__ = ["app"]
