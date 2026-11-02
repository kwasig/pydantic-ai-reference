#!/usr/bin/env bash
# One command on your own machine: install, verify against the real services, start the Studio.
#   1. put NEBIUS_API_KEY (and optionally CARTESIA_API_KEY, DAYTONA_API_KEY, TAVILY_API_KEY) in .env
#   2. bash scripts/run_live.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f .env ]; then cp .env.example .env; echo "created .env — add NEBIUS_API_KEY and run again"; exit 1; fi
set -a; . ./.env; set +a
: "${NEBIUS_API_KEY:?NEBIUS_API_KEY missing in .env (create one at https://tokenfactory.nebius.com)}"
export NEBIUS_BASE_URL="${NEBIUS_BASE_URL:-https://api.tokenfactory.nebius.com/v1/}"

if [ ! -d .venv ]; then python3 -m venv .venv; fi
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q -e ".[dev,daytona]" cartesia

echo; echo "▶ 1/4  models served at $NEBIUS_BASE_URL"
nebius-agent models | head -25
if [ -z "${NEBIUS_MODEL:-}" ]; then
  echo "   NEBIUS_MODEL not set in .env; pick a tool-calling model from the list above and add it, e.g. NEBIUS_MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507"
fi

echo; echo "▶ 2/4  conformance kit on ${NEBIUS_MODEL:-the default model}"
mkdir -p reports
TRACE_EXPORTER=none nebius-agent verify --json "reports/$(echo "${NEBIUS_MODEL:-default}" | tr '/' '_').json" || echo "   (some checks did not pass — that is information, keep going)"

echo; echo "▶ 3/4  one agent run with a trace"
TRACE_EXPORTER=file TRACE_FILE=traces/live.jsonl nebius-agent run "A partner is building a voice agent. Which served model should they use? Benchmark two candidates on a short support reply." || true
TRACE_FILE=traces/live.jsonl nebius-agent trace || true

echo; echo "▶ 4/4  Studio UI  (Daytona: ${DAYTONA_API_KEY:+on}${DAYTONA_API_KEY:-off}, Cartesia: ${CARTESIA_API_KEY:+on}${CARTESIA_API_KEY:-off}, Tavily: ${TAVILY_API_KEY:+on}${TAVILY_API_KEY:-off})"
export TRACE_EXPORTER=memory
nebius-agent studio --port 8090
