#!/usr/bin/env bash
# One command on your own machine: install, verify against the real services, start the Studio.
#   1. put NEBIUS_API_KEY (and optionally CARTESIA_API_KEY, DAYTONA_API_KEY, TAVILY_API_KEY) in .env
#   2. bash scripts/run_live.sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p logs reports traces
exec > >(tee -a logs/live.log) 2>&1
echo "=== run_live $(date) ==="

if [ ! -f .env ]; then cp .env.example .env; echo "created .env — add NEBIUS_API_KEY and run again"; exit 1; fi
set -a; . ./.env; set +a
: "${NEBIUS_API_KEY:?NEBIUS_API_KEY missing in .env (create one at https://tokenfactory.nebius.com)}"
export NEBIUS_BASE_URL="${NEBIUS_BASE_URL:-https://api.tokenfactory.nebius.com/v1/}"

# Python >= 3.11 is required. Try the usual places, then fall back to uv, which downloads one.
PY=""
for c in python3.13 python3.12 python3.11 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then PY="$c"; break; fi
done
if [ ! -d .venv ]; then
  if [ -n "$PY" ]; then
    "$PY" -m venv .venv
  else
    if ! command -v uv >/dev/null 2>&1; then
      echo "no Python >= 3.11 found; installing uv to fetch one"
      curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    fi
    uv venv --python 3.12 .venv
  fi
fi
. .venv/bin/activate
python -m pip install -q --upgrade pip 2>/dev/null || true
python -m pip install -q -e ".[dev,daytona]" cartesia

echo; echo "▶ 1/4  models served at $NEBIUS_BASE_URL"
nebius-agent models > reports/models.txt || true
head -40 reports/models.txt
# Pick a served tool-calling model if the configured one is not in the list.
if ! grep -q "${NEBIUS_MODEL:-Qwen/Qwen3-235B-A22B-Instruct-2507}" reports/models.txt; then
  for cand in Qwen/Qwen3-235B-A22B-Instruct-2507 Qwen/Qwen3-30B-A3B-Instruct-2507 openai/gpt-oss-120b Qwen/Qwen3.5-397B-A17B moonshotai/Kimi-K2.6 zai-org/GLM-5.1 deepseek-ai/DeepSeek-V4-Flash-0731; do
    if grep -q "$cand" reports/models.txt; then export NEBIUS_MODEL="$cand"; echo "   configured model not served; using $NEBIUS_MODEL"; break; fi
  done
fi
export NEBIUS_MODEL="${NEBIUS_MODEL:-Qwen/Qwen3-235B-A22B-Instruct-2507}"
echo "   model: $NEBIUS_MODEL"

echo; echo "▶ 2/4  conformance kit on ${NEBIUS_MODEL:-the default model}"
mkdir -p reports
TRACE_EXPORTER=none nebius-agent verify --json "reports/$(echo "${NEBIUS_MODEL:-default}" | tr '/' '_').json" || echo "   (some checks did not pass — that is information, keep going)"

echo; echo "▶ 3/4  one agent run with a trace"
TRACE_EXPORTER=file TRACE_FILE=traces/live.jsonl nebius-agent run "A partner is building a voice agent. Which served model should they use? Benchmark two candidates on a short support reply." || true
TRACE_FILE=traces/live.jsonl nebius-agent trace || true

on_off() { if [ -n "${1:-}" ]; then echo on; else echo off; fi; }
echo; echo "▶ 4/4  Studio UI  (Daytona: $(on_off "${DAYTONA_API_KEY:-}"), Cartesia: $(on_off "${CARTESIA_API_KEY:-}"), Tavily: $(on_off "${TAVILY_API_KEY:-}"))"
export TRACE_EXPORTER=memory
nebius-agent studio --port 8090
