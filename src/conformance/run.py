"""Run every conformance check and print / return a report."""

from __future__ import annotations

import asyncio
import platform
import sys
from datetime import UTC, datetime
from typing import Any

from rich.console import Console
from rich.table import Table

from nebius_agent.config import get_settings
from nebius_agent.provider import make_client

from .checks import ALL_CHECKS, FAIL, PARTIAL, PASS, CheckResult

console = Console()


async def run_conformance(model: str | None = None, checks=ALL_CHECKS) -> dict[str, Any]:
    s = get_settings()
    model = model or s.model
    client = make_client(s)
    results: list[CheckResult] = []
    for check in checks:
        r = await check(client, model)
        results.append(r)
        console.print(f"[{_color(r.status)}]{r.status:7}[/] {r.name:30} {r.ms:8.1f} ms  {r.detail}")
    summary = {k: sum(1 for r in results if r.status == k) for k in (PASS, PARTIAL, FAIL)}
    report = {
        "endpoint": s.base_url,
        "model": model,
        "ran_at": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "summary": {"pass": summary[PASS], "partial": summary[PARTIAL], "fail": summary[FAIL], "total": len(results)},
        "results": [r.__dict__ for r in results],
    }
    _print_table(report)
    return report


def _color(status: str) -> str:
    return {PASS: "green", PARTIAL: "yellow", FAIL: "red"}.get(status, "white")


def _print_table(report: dict[str, Any]) -> None:
    t = Table(title=f"Token Factory conformance: {report['model']} @ {report['endpoint']}")
    t.add_column("check")
    t.add_column("status")
    t.add_column("ms", justify="right")
    t.add_column("detail")
    for r in report["results"]:
        t.add_row(r["name"], f"[{_color(r['status'])}]{r['status']}[/]", f"{r['ms']:.0f}", r["detail"][:90])
    console.print(t)
    s = report["summary"]
    console.print(f"[bold]{s['pass']} pass, {s['partial']} partial, {s['fail']} fail[/bold] of {s['total']}")


def markdown_table(report: dict[str, Any]) -> str:
    rows = ["| Check | Status | Detail |", "|---|---|---|"]
    for r in report["results"]:
        rows.append(f"| `{r['name']}` | {r['status']} | {r['detail'].replace('|', '/')[:100]} |")
    return "\n".join(rows)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(run_conformance(sys.argv[1] if len(sys.argv) > 1 else None))
