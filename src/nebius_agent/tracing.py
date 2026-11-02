"""
OpenTelemetry setup for the agent.

PydanticAI instruments itself with OpenTelemetry using the GenAI semantic conventions
(spans like `agent run`, `chat <model>`, `running tool`). We only choose where spans go:

  TRACE_EXPORTER=file     write spans as JSON lines to TRACE_FILE (default traces/spans.jsonl)
  TRACE_EXPORTER=console  pretty-print spans to stderr
  TRACE_EXPORTER=otlp     OTLP/HTTP to OTEL_EXPORTER_OTLP_ENDPOINT (Langfuse, Phoenix, Grafana, LangSmith...)
  TRACE_EXPORTER=none     disable

Langfuse shortcut: set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and (optionally) LANGFUSE_HOST
and we build the OTLP endpoint and Basic-auth header for you.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
from collections.abc import Sequence
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from .config import Settings, get_settings


class JsonlFileSpanExporter(SpanExporter):
    """One span per line, so a partner can `jq` a trace without any backend."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            for span in spans:
                f.write(json.dumps(_span_to_dict(span), ensure_ascii=False) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:  # pragma: no cover - nothing to do
        pass


def _span_to_dict(span: ReadableSpan) -> dict:
    ctx = span.get_span_context()
    parent = span.parent.span_id if span.parent else None
    return {
        "name": span.name,
        "trace_id": f"{ctx.trace_id:032x}",
        "span_id": f"{ctx.span_id:016x}",
        "parent_span_id": f"{parent:016x}" if parent else None,
        "start_ns": span.start_time,
        "end_ns": span.end_time,
        "duration_ms": round(((span.end_time or 0) - (span.start_time or 0)) / 1e6, 3),
        "status": span.status.status_code.name,
        "attributes": dict(span.attributes or {}),
        "events": [{"name": e.name, "attributes": dict(e.attributes or {})} for e in span.events],
    }


def _langfuse_otlp_env() -> None:
    """Translate LANGFUSE_* into the standard OTEL_* variables."""
    pk, sk = os.getenv("LANGFUSE_PUBLIC_KEY"), os.getenv("LANGFUSE_SECRET_KEY")
    if pk and sk and not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")
        auth = base64.b64encode(f"{pk}:{sk}".encode()).decode()
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{host}/api/public/otel"
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {auth}"
        os.environ.setdefault("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")


_memory_exporter: InMemorySpanExporter | None = None


def configure_tracing(settings: Settings | None = None, exporter: str | None = None) -> TracerProvider:
    """Build a TracerProvider and make it global. Safe to call once per process."""
    global _memory_exporter
    s = settings or get_settings()
    kind = (exporter or s.trace_exporter).lower()
    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: s.service_name, "nebius.base_url": s.base_url}))

    if kind == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter(out=sys.stderr)))
    elif kind == "file":
        provider.add_span_processor(SimpleSpanProcessor(JsonlFileSpanExporter(s.trace_file)))
    elif kind == "otlp":
        _langfuse_otlp_env()
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    elif kind == "memory":
        _memory_exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(_memory_exporter))
    elif kind == "none":
        pass
    else:
        raise ValueError(f"unknown TRACE_EXPORTER={kind!r}; use file|console|otlp|memory|none")

    trace.set_tracer_provider(provider)
    return provider


def memory_spans() -> list[ReadableSpan]:
    """Spans captured when TRACE_EXPORTER=memory (used by the tests)."""
    return list(_memory_exporter.get_finished_spans()) if _memory_exporter else []


def read_trace_file(path: str | None = None) -> list[dict]:
    p = Path(path or get_settings().trace_file)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
