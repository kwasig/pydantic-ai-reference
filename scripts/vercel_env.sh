#!/usr/bin/env bash
# Push the keys from .env to a Vercel project as production environment variables.
# Run from the repo root on a machine that has `vercel` (npx is fine) and is linked to the project
# (`npx vercel link`). Keys never leave your machine except to Vercel.
set -eu
cd "$(dirname "$0")/.."
[ -f .env ] || { echo ".env missing"; exit 1; }
set -a; . ./.env; set +a
push() { # name value
  [ -n "${2:-}" ] || return 0
  printf '%s' "$2" | npx -y vercel env add "$1" production --force ${VERCEL_TOKEN:+--token "$VERCEL_TOKEN"} >/dev/null && echo "  set $1"
}
echo "▶ pushing environment to Vercel (production)"
push NEBIUS_API_KEY "${NEBIUS_API_KEY:-}"
push NEBIUS_BASE_URL "${NEBIUS_BASE_URL:-https://api.tokenfactory.nebius.com/v1/}"
push NEBIUS_MODEL "${NEBIUS_MODEL:-Qwen/Qwen3-235B-A22B-Instruct-2507}"
push DAYTONA_API_KEY "${DAYTONA_API_KEY:-}"
push CARTESIA_API_KEY "${CARTESIA_API_KEY:-}"
push TAVILY_API_KEY "${TAVILY_API_KEY:-}"
push LANGFUSE_PUBLIC_KEY "${LANGFUSE_PUBLIC_KEY:-}"
push LANGFUSE_SECRET_KEY "${LANGFUSE_SECRET_KEY:-}"
push LANGFUSE_BASE_URL "${LANGFUSE_BASE_URL:-}"
push STUDIO_STATELESS "1"
push TRACE_EXPORTER "memory+otlp"
# dedicated-endpoint provisioning holds a request for ~7.5 minutes: fits Vercel Pro (maxDuration 800), not Hobby (300)
push ALLOW_DEDICATED "${VERCEL_ALLOW_DEDICATED:-0}"
echo "done. Deploy with: npx vercel --prod"
