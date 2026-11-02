#!/usr/bin/env bash
# Panel demo, ~8 minutes. Run against Token Factory (NEBIUS_API_KEY set) or the mock (make mock).
set -euo pipefail
. .venv/bin/activate
: "${NEBIUS_BASE_URL:=https://api.tokenfactory.nebius.com/v1/}"
export NEBIUS_BASE_URL TRACE_EXPORTER=file TRACE_FILE=traces/demo.jsonl
rm -f traces/demo.jsonl

step() { echo; echo "────────────────────────────────────────────────────────────"; echo "▶ $*"; echo; }

step "1. What does Token Factory serve?"
nebius-agent models | head -12

step "2. Is the endpoint + model actually agent-ready? (framework-agnostic conformance kit)"
nebius-agent verify --json reports/demo.json || true

step "3. The reference agent: tool calls -> typed recommendation (PydanticAI on Token Factory)"
nebius-agent run "A partner is building a real-time voice agent. Which served model should they use? Benchmark two candidates on a short support reply."

step "4. Streaming, token by token"
nebius-agent stream "In three sentences, what did you measure and what would you tell the partner?"

step "5. The trace (OpenTelemetry, GenAI semantic conventions)"
nebius-agent trace

echo
echo "Live-change ideas for the panel:"
echo "  • swap the model:            nebius-agent run '...' --model deepseek-ai/DeepSeek-V3-0324"
echo "  • switch output mode:        --output-mode native   (response_format json_schema instead of a tool call)"
echo "  • ship traces to Langfuse:   LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... nebius-agent run '...' --exporter otlp"
echo "  • add a tool live:           edit src/nebius_agent/tools.py, add to ALL_TOOLS, rerun; `nebius-agent schemas` shows what the model sees"
