"""
Command line for the reference agent.

  nebius-agent models                 list what Token Factory serves
  nebius-agent run "prompt"           one agent run, structured output, trace written
  nebius-agent stream "prompt"        same, streaming text as it arrives
  nebius-agent verify                 conformance kit against the configured endpoint
  nebius-agent schemas                show the tool JSON schemas sent to the model
  nebius-agent trace                  summarise the last trace file
"""

from __future__ import annotations

import asyncio
import json
import os

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .agent import build_agent, build_deps
from .config import get_settings
from .tracing import read_trace_file

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
console = Console()


@app.command()
def models(contains: str | None = typer.Option(None, help="substring filter")) -> None:
    """List models served by the configured endpoint."""
    from .provider import make_client

    async def _run():
        client = make_client()
        page = await client.models.list()
        ids = [m.id for m in page.data]
        return [i for i in ids if not contains or contains.lower() in i.lower()]

    ids = asyncio.run(_run())
    s = get_settings()
    console.print(f"[bold]{len(ids)} models[/bold] at {s.base_url}")
    for i in ids:
        console.print(f"  {i}")


@app.command()
def run(
    prompt: str = typer.Argument(..., help="What the partner wants, e.g. 'pick a model for a voice agent'"),
    model: str | None = typer.Option(None, help="override NEBIUS_MODEL"),
    output_mode: str = typer.Option("tool", help="tool | native | prompted | text"),
    exporter: str | None = typer.Option(None, help="file | console | otlp | none (default TRACE_EXPORTER)"),
) -> None:
    """Run the agent once and print the structured recommendation."""
    agent = build_agent(model, output_mode=output_mode, trace_exporter=exporter)  # type: ignore[arg-type]
    deps = build_deps()

    async def _run():
        result = await agent.run(prompt, deps=deps)
        return result

    result = asyncio.run(_run())
    out = result.output
    if hasattr(out, "model_dump"):
        console.print(Panel(json.dumps(out.model_dump(), indent=2), title="ModelRecommendation", border_style="blue"))
    else:
        console.print(Panel(str(out), title="output", border_style="blue"))
    u = result.usage
    console.print(
        f"usage: requests={u.requests} input={u.input_tokens} output={u.output_tokens} cache_read={getattr(u, 'cache_read_tokens', 0)}"
    )
    s = get_settings()
    if (exporter or s.trace_exporter) == "file":
        console.print(f"trace: {s.trace_file}  (run `nebius-agent trace` to summarise)")


@app.command()
def stream(
    prompt: str = typer.Argument(...),
    model: str | None = typer.Option(None),
    exporter: str | None = typer.Option(None),
) -> None:
    """Stream a free-text answer token by token (text output mode)."""
    agent = build_agent(model, output_mode="text", trace_exporter=exporter)
    deps = build_deps()

    async def _run():
        async with agent.run_stream(prompt, deps=deps) as result:
            async for delta in result.stream_text(delta=True):
                console.print(delta, end="", soft_wrap=True)
        console.print()
        u = result.usage
        console.print(f"[dim]usage: requests={u.requests} input={u.input_tokens} output={u.output_tokens}[/dim]")

    asyncio.run(_run())


@app.command()
def verify(
    model: str | None = typer.Option(None, help="override NEBIUS_MODEL"),
    json_out: str | None = typer.Option(None, "--json", help="write the report to this path"),
) -> None:
    """Run the framework-agnostic conformance kit against the configured endpoint."""
    from conformance.run import run_conformance

    report = asyncio.run(run_conformance(model))
    if json_out:
        os.makedirs(os.path.dirname(json_out) or ".", exist_ok=True)
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        console.print(f"report written to {json_out}")
    raise typer.Exit(code=0 if report["summary"]["fail"] == 0 else 1)


@app.command()
def studio(
    port: int = typer.Option(8090, help="port for the Studio UI"),
    host: str = typer.Option("127.0.0.1"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="open the UI in your browser"),
) -> None:
    """Start the Nebius FDE Studio web UI (live agent runs with sandbox, benchmarks, voice and traces)."""
    import threading
    import webbrowser

    import uvicorn

    s = get_settings()
    console.print(f"[bold]Nebius FDE Studio[/bold] → http://{host}:{port}   endpoint={s.base_url} model={s.model}")
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    uvicorn.run("nebius_agent.studio:app", host=host, port=port, log_level="warning")


@app.command()
def schemas() -> None:
    """Print the JSON schemas PydanticAI derives from the tool functions."""
    from .tools import tool_schemas_preview

    console.print(tool_schemas_preview())


@app.command()
def trace(path: str | None = typer.Option(None, help="trace file (default TRACE_FILE)")) -> None:
    """Summarise spans in the JSONL trace file: one row per span, nested by parent."""
    spans = read_trace_file(path)
    if not spans:
        console.print("no spans found; run the agent with TRACE_EXPORTER=file first")
        raise typer.Exit(1)
    by_id = {s["span_id"]: s for s in spans}
    depth = {}

    def d(sid: str) -> int:
        if sid in depth:
            return depth[sid]
        p = by_id[sid]["parent_span_id"]
        depth[sid] = 0 if not p or p not in by_id else d(p) + 1
        return depth[sid]

    t = Table(title=f"{len(spans)} spans, {len({s['trace_id'] for s in spans})} trace(s)")
    t.add_column("span")
    t.add_column("ms", justify="right")
    t.add_column("gen_ai attributes")
    for s in sorted(spans, key=lambda x: x["start_ns"]):
        a = s["attributes"]
        keys = [
            k
            for k in a
            if k.startswith("gen_ai.") and k.endswith(("model", "name", "input_tokens", "output_tokens", "operation.name", "system"))
        ]
        attrs = ", ".join(f"{k.split('gen_ai.')[1]}={a[k]}" for k in keys[:5])
        t.add_row("  " * d(s["span_id"]) + s["name"], f"{s['duration_ms']:.1f}", attrs)
    console.print(t)


if __name__ == "__main__":  # pragma: no cover
    app()
