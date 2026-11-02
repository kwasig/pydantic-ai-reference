# PydanticAI on Nebius Token Factory — reference agent + conformance kit

A production-shaped reference integration that makes [PydanticAI](https://ai.pydantic.dev) work well on
[Nebius Token Factory](https://tokenfactory.nebius.com), and a reusable kit that makes the *next*
framework integration cheap.

What you get:

| | |
|---|---|
| **Reference agent** | `nebius-agent run "..."` — a "latency scout" that lists served models, benchmarks candidates on the partner's own prompt, optionally grounds with Tavily, and returns a typed `ModelRecommendation`. Tool calling, structured output (3 modes), streaming. |
| **Traces** | OpenTelemetry spans using the GenAI semantic conventions, out of the box. Ship them to a JSONL file, the console, or any OTLP backend (Langfuse shortcut included). |
| **Conformance kit** | `nebius-agent verify` — 12 framework-agnostic checks (tool calls, parallel calls, `tool_choice`, `json_schema` output, streaming tool-call deltas, prefix-cache reporting…) that tell you whether an *endpoint + model* is agent-ready before you touch a framework. This is the reusable piece for partner #2…#10. |
| **Studio** | `nebius-agent studio` — a web UI where the agent runs framework probes in a Daytona sandbox, the conformance kit, benchmarks and Tavily search live, returns a typed report and speaks it through Cartesia. |
| **Runtime** | The agent as a FastAPI service, Dockerfile, and a Kubernetes manifest for Nebius Managed Kubernetes (`deploy/`). |
| **Offline mode** | A deterministic mock of Token Factory so tests, CI and demos run with no key and no network. |

Verified end to end (17 tests, 12 conformance checks, a headless-browser run of the Studio) against the mock. Against Token Factory
itself the same commands produce the partner-facing report; see [Status](#status) for what has
and has not been run there.

---

## 1. Five-minute start

```bash
git clone https://github.com/kwasig/pydantic-ai-reference.git && cd pydantic-ai-reference
make install                       # python3 -m venv .venv && pip install -e ".[dev]"
cp .env.example .env               # add NEBIUS_API_KEY from https://tokenfactory.nebius.com
. .venv/bin/activate

nebius-agent models                # what Token Factory serves right now
nebius-agent verify                # is this endpoint + model agent-ready?  (writes reports/*.json)
nebius-agent run "A partner is building a voice agent. Which served model should they use? Benchmark two candidates."
nebius-agent stream "Summarise what you measured in two sentences."
nebius-agent trace                 # the OpenTelemetry spans from the last run
```

No key yet? Run everything against the mock:

```bash
make mock                                        # terminal 1: mock Token Factory on :8008
export NEBIUS_BASE_URL=http://127.0.0.1:8008/v1/ NEBIUS_API_KEY=mock   # terminal 2
nebius-agent verify && nebius-agent run "Pick a model for a coding agent; benchmark one."
make test                                        # 15 tests, starts its own mock
```


## 1b. Integration Scout: the Studio UI

```bash
make studio          # http://127.0.0.1:8090  (reads .env: NEBIUS_API_KEY, DAYTONA_API_KEY, CARTESIA_API_KEY, TAVILY_API_KEY)
make studio-mock     # same UI against the mock + local sandbox, no keys
```

The four example scenarios target the newest models Token Factory serves (MiniMax-M3.1-Flash, Kimi-K2.7-Code, DeepSeek-V4.1-Flash, Kimi-K3) and compare them with the verified default, `Qwen/Qwen3-235B-A22B-Instruct-2507`, which is also the model the scout itself runs on. The "Model to evaluate" selector changes what is probed, benchmarked and checked, not the orchestrator.

![Integration Scout: describe the workload, pick a scenario, run](docs/img/scout-compose.png)

![A recorded live run replayed: Daytona sandbox probe with the tool call verified, the 13-check kit, two benchmarks, and the plan with Cartesia's spoken verdict](docs/img/scout-sandbox.png)

![The scout provisions a dedicated 1× H200 endpoint for Qwen3-30B, waits for readiness, warms it, benchmarks (111 ms TTFT vs 268 ms shared) and tears it down after 0.1 billable minutes](docs/img/scout-dedicated.png)

![The coding-agent session, recorded live: Kimi-K3 in a Daytona sandbox writes metrics.py and tests, runs pytest (6 pass), renders chart.png; the chart comes back into the plan](docs/img/scout-session.png)

The Studio ("Nebius Integration Scout") is the FDE motion as a product for a partner: describe the workload (type, framework, TTFT budget, volume) or pick an example, and the
agent **proves** things rather than describing them:

| Step | What actually happens | Where |
|---|---|---|
| `run_probe` | writes a framework probe (PydanticAI, LangGraph or raw OpenAI SDK) into a **Daytona sandbox**, installs the framework, runs it against Token Factory, and reports whether the tool call *really* happened and with which arguments | `probes/`, `sandbox.py` |
| `run_coding_session` | a real coding-agent session inside the sandbox: a PydanticAI agent on the model gets `write_file` and `run` tools, implements a module, writes pytest tests, runs them to green and renders a matplotlib chart. The UI shows every tool call, an *independent* pytest result (the probe re-runs the tests itself, it does not trust the model's self-report), the files, and the chart copied back out of the sandbox. This is what a long-lived sandbox with a filesystem and a shell buys over a one-shot execution API | `probes/coding_agent_probe.py` |
| `run_conformance` | the 13-check kit, each check streaming into the UI as PASS / PARTIAL / FAIL | `conformance/` |
| `benchmark` | TTFT and tokens/s on the partner's own prompt, charted live. The clock starts at the first streamed token, reasoning or content, and the result carries decode rate, end-to-end rate, chunk counts, reasoning tokens and a `stream_note`; a buffered ("bursty") stream or a reasoning model is reported as such rather than as a flattering number | `tools.py` |
| `provision_dedicated_endpoint` | optional, off by default (`ALLOW_DEDICATED=1`): when nothing meets the partner's latency budget on the shared endpoint, the scout provisions a temporary dedicated Token Factory endpoint for the recommended model (smallest allowed GPU config from the template catalogue), waits for `readiness=ready`, benchmarks the partner's prompt on it, and deletes it. Named `scout-*`, deleted in a `finally`, hard time cap, and the Studio sweeps leftovers on startup. Billing only starts at `ready` | `dedicated.py` |
| `web_search` | live grounding via Tavily when the ask needs facts | `tools.py` |
| report | a typed `FdeReport`: verdict, model, findings with numbers, next steps; plus deliverables derived from what was verified, not from prose: a copy-paste integration snippet for the partner's framework, a downloadable `INTEGRATION.md`, the TTFT budget check, and a cost estimate when `pricing.json` has prices | `studio_agent.py`, `studio.py` |
| voice | the report's 60-word `spoken_summary`, synthesised by **Cartesia Sonic** and played in the page (browser speech if no key) | `voice.py` |
| trace | the OpenTelemetry spans for the run, as a tree, with an "open in Langfuse" link when keys are set | `tracing.py` |
| Langfuse panel | the other direction of the observability partnership: the Studio reads recent traces, model-call counts and p95 run latency *back* from Langfuse's public API and shows them next to the run | `studio.py` `/api/langfuse/metrics` |
| replay | any finished run can be exported (`/api/runs/{id}/export?save_as=name`) and replayed from `demo/runs/` with its original timing and audio, so the demo survives a dead Wi-Fi | `studio.py` |

Everything streams over Server-Sent Events (`/api/runs/{id}/events`), so the panel can watch a
sandbox install packages line by line. Without keys the Studio degrades honestly: the status pills
say "mock", "local fallback" and "browser fallback" so nobody mistakes a stand-in for the real thing.

## 2. The integration in 25 lines

Token Factory is OpenAI-compatible, so PydanticAI needs one provider line. Everything else
(tool schemas, validation, retries, structured output, streaming, tracing) is the framework.
See [`examples/partner_quickstart.py`](examples/partner_quickstart.py):

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIChatModel(
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    provider=OpenAIProvider(base_url="https://api.tokenfactory.nebius.com/v1/", api_key=NEBIUS_API_KEY),
)
agent = Agent(model, output_type=CityReport, instructions="Use the tool, then answer.")

@agent.tool_plain
def get_weather(city: str) -> dict:
    """Current weather for a city."""
    ...

Agent.instrument_all()           # OpenTelemetry, GenAI semantic conventions
result = await agent.run("What's the weather in Amsterdam?")
result.output                    # -> CityReport(city='Amsterdam', ...)
```

The repo's [`provider.py`](src/nebius_agent/provider.py) adds a **model profile** for open models on
Token Factory (no `strict: true` on tools, `max_tokens` not `max_completion_tokens`, DeepSeek-style
`reasoning_content`, prefix-cache aware). Partners copy that profile; the conformance kit is what
keeps it honest.

## 3. What the reference agent does

```
user: "A partner is building a voice agent. Which model? Benchmark two candidates."
  ├─ chat  Qwen/...            → tool call  list_models(contains=None, limit=20)
  ├─ execute_tool list_models  → ["Qwen/...", "deepseek-ai/...", ...]
  ├─ chat  Qwen/...            → tool call  benchmark_model(model=..., prompt=..., runs=2)
  ├─ execute_tool benchmark_model → {ttft_ms: 212.4, tokens_per_s: 88.1, cached_prompt_tokens: 230, ...}
  └─ chat  Qwen/...            → final_result(ModelRecommendation{workload="voice", recommended_model=..., ...})
```

Tools (all in [`tools.py`](src/nebius_agent/tools.py); PydanticAI derives the JSON schema from the signature + docstring, `nebius-agent schemas` shows what the model receives):

| Tool | What it does | Needs |
|---|---|---|
| `list_models` | `GET /v1/models`, optional substring filter | – |
| `benchmark_model` | streams a prompt N times; median TTFT, tokens/s, and `cached_tokens` on the repeat (prefix cache) | – |
| `web_search` | live grounding via [Tavily](https://tavily.com) (Nebius) | `TAVILY_API_KEY` |
| `run_python` | code execution in a [Daytona](https://daytona.io) sandbox, or a local subprocess in dev | `DAYTONA_API_KEY` or `ALLOW_LOCAL_EXEC=1` |

Structured output, three ways (`--output-mode`):

| mode | wire mechanism | when to use |
|---|---|---|
| `tool` (default) | the model calls a `final_result` tool whose schema is `ModelRecommendation` | most portable across open models; needs tool calling only |
| `native` | `response_format: {type: json_schema}` | fastest, no extra tool turn; needs server-side guided decoding (checked by `structured.json_schema`) |
| `prompted` | schema in the instructions + `response_format: json_object` | fallback for models with neither |

Streaming: `nebius-agent stream` uses `agent.run_stream()`; the service exposes it as SSE at `POST /recommend/stream`.

## 4. Traces

PydanticAI instruments itself with OpenTelemetry. [`tracing.py`](src/nebius_agent/tracing.py) only chooses the exporter:

```bash
TRACE_EXPORTER=file     nebius-agent run "..."   # traces/spans.jsonl, one span per line (default)
TRACE_EXPORTER=console  nebius-agent run "..."
TRACE_EXPORTER=otlp OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318/v1/traces nebius-agent run "..."
LANGFUSE_PUBLIC_KEY=pk-lf-... LANGFUSE_SECRET_KEY=sk-lf-... nebius-agent run "..." --exporter otlp
```

A run produces `invoke_agent` → `chat <model>` → `execute_tool <name>` spans with `gen_ai.*`
attributes (operation, request model, input/output tokens, tool name). A sample is checked in at
[`traces/sample-spans.jsonl`](traces/sample-spans.jsonl); `nebius-agent trace` renders it:

```
invoke_agent token-factory-latency-scout   170.0 ms
  chat Qwen/Qwen3-235B-A22B-Instruct-2507   39.7 ms   operation.name=chat, request.model=...
  execute_tool list_models                   6.7 ms   tool.name=list_models
  chat Qwen/...                              6.4 ms
  execute_tool benchmark_model              92.2 ms   tool.name=benchmark_model
  chat Qwen/...                             10.0 ms
```

## 5. The conformance kit (the reusable part)

Every agent framework sits on the same wire-level capabilities. Before debugging a framework,
find out whether the endpoint + model actually has them:

```bash
nebius-agent verify --model deepseek-ai/DeepSeek-V3-0324 --json reports/deepseek.json
```

| check | asks |
|---|---|
| `chat.basic`, `usage.reported` | does it answer, does it report tokens |
| `tools.call_emitted` | a tool call with JSON-valid, schema-shaped arguments |
| `tools.result_roundtrip` | after a `role: tool` message the model answers instead of looping |
| `tools.parallel` | two tool calls in one turn (PARTIAL if only sequential) |
| `tools.choice_required`, `tools.choice_named` | `tool_choice: required` and `{function: {name}}` are honoured |
| `structured.json_schema`, `structured.json_object` | `response_format` guided decoding |
| `streaming.text` | several content chunks, `finish_reason`, usage in the final chunk |
| `streaming.tool_call_deltas` | tool-call deltas reassemble into valid JSON, `finish_reason=tool_calls` |
| `cache.cached_tokens` | repeat of a ~1k-token prefix reports `prompt_tokens_details.cached_tokens > 0` |

Each returns PASS / PARTIAL / FAIL with evidence, printed and written as JSON. The JSON report is
the artefact we hand a partner on day one ("here is what your model does on Nebius") and attach
to a field-to-product ticket when it does not. Adding a framework later means re-running the
*same* kit plus one `tests/test_<framework>_through_framework.py`.

## 6. Running it on Nebius infrastructure

```
            partner app / harness
                    │ HTTP
     ┌──────────────▼───────────────┐      OTLP      ┌───────────────────┐
     │ latency-scout (this repo)    │───────────────▶│ Langfuse          │
     │ FastAPI + PydanticAI         │                │ self-hosted on the│
     │ Nebius Managed Kubernetes    │                │ same cluster      │
     └───┬───────────┬──────────┬───┘                └───────────────────┘
         │ chat      │ search   │ code
 ┌───────▼──────┐ ┌──▼──────┐ ┌─▼────────────────┐
 │ Token Factory│ │ Tavily  │ │ Daytona / ConTree│
 │ dedicated    │ │ (Nebius)│ │ sandbox          │
 │ endpoint     │ └─────────┘ └──────────────────┘
 └──────────────┘
```

* **Model**: a Token Factory *dedicated endpoint* for the partner's model, so latency and throughput are an SLA rather than shared-pool luck. Same base URL shape, just a different `NEBIUS_BASE_URL`.
* **Agent runtime**: this container (`Dockerfile`), stateless, on Nebius Managed Kubernetes with an HPA (`deploy/k8s.yaml`). Conversation state belongs in the partner's store, not in the pod.
* **Tools**: Tavily for the web; Daytona or Token Factory's ConTree sandboxes for code.
* **Traces**: OTLP to Langfuse running in the same cluster (or Langfuse Cloud), so traces, evals and datasets live next to the inference they describe. That is the loop that feeds post-training on Token Factory.

`uvicorn nebius_agent.server:app --port 8080` runs the same service locally; `tests/test_service.py` covers it.

## 7. Why PydanticAI (and not LangGraph, CrewAI, or the OpenAI Agents SDK)

* LangChain/LangGraph already has a first-party Nebius integration (`langchain-nebius`, May 2026). The marginal value of a second one is low; the FDE job is to make **#2…#10** cheap, so the kit matters more than the framework.
* PydanticAI is typed end to end (tools *and* outputs are Pydantic models), which is exactly where open models on OpenAI-compatible servers break: schema shape, `strict` flags, `tool_choice`, `json_schema` support. It surfaces those problems as validation errors instead of silent garbage.
* It emits OpenTelemetry with the GenAI semantic conventions natively, so "industry-standard tracing" is a flag, not a plugin.
* CrewAI and the OpenAI Agents SDK are next; the Agents SDK's default tracing goes to OpenAI's platform, which is its own partner conversation.

## 8. Status — verified live on 2 Nov 2026

| Verified against | tests | conformance | notes |
|---|---|---|---|
| mock Token Factory (`src/mock_server`) | 18/18 | 13/13 PASS | deterministic; runs in CI with no secrets; Studio driven end to end in headless Chromium |
| **Token Factory** `api.tokenfactory.nebius.com` · `Qwen/Qwen3-235B-A22B-Instruct-2507` | — | **13/13 PASS** | tools, parallel calls, forced choice, json_schema, streaming deltas, prefix cache 688/700 cached; `reports/Qwen_Qwen3-235B-A22B-Instruct-2507.json` |
| Token Factory · `Qwen/Qwen3-30B-A3B-Instruct-2507` | — | 12/13 · 1 PARTIAL | 304 ms TTFT, 231 tok/s on the shared endpoint; prefix cache not observable in that run |
| Token Factory · `deepseek-ai/DeepSeek-V4.1-Flash-1017` | — | 12/13 · 1 FAIL | `structured.json_schema` returns prose; 1,788 tok/s, 638 ms TTFT |
| Token Factory · `moonshotai/Kimi-K2.7-Code` | — | 9/13 · 2 FAIL | empty `chat.basic` content, no streamed text deltas; 838 tok/s |
| Token Factory · `moonshotai/Kimi-K3` | — | 9/13 · 3 FAIL | `chat.basic`, `tools.choice_named`, `structured.json_schema`; yet it completed the full coding session in Daytona (code, 6 tests green, chart) with 352 ms TTFT and ~210 tok/s end to end |
| Token Factory · `moonshotai/Kimi-K2.7-Code` (benchmark) | — | — | reasoning model: on a 160-token budget every token was reasoning and no content arrived; 32 s to the first visible token. The benchmark now says so instead of reporting 3,400 tok/s |
| Token Factory · `MiniMaxAI/MiniMax-M3.1-Flash` | — | probe ✔ | PydanticAI tool call verified; **3.2 s TTFT** on the shared endpoint (10× a voice budget) |
| **Token Factory dedicated endpoint** · `Qwen3-30B-A3B-Instruct-2507` on 1× H200 | — | ✔ | provisioned by the scout (template catalogue → smallest allowed config), `ready` in 5 min 24 s, cold first request 1,279 ms, **warm TTFT 111 ms vs 268 ms shared**, 157 tok/s end to end, deleted after 0.1 billable minutes |
| **Daytona** | — | ✔ | ephemeral `python:3.12-slim` sandboxes created, probe uploaded + executed, deleted; PydanticAI, LangGraph and OpenAI-SDK probes all reported `tool_called=true`; the coding session (Kimi-K3) wrote code + tests, 6 green, rendered a chart |
| **Cartesia** | — | ✔ | `sonic-3` speaks every verdict (≈2.7–3.7 s per 60 words) |
| **Langfuse** (`us.cloud.langfuse.com`) | — | ✔ | every Studio run exported over OTLP; the Studio reads traces/latency/model-call counts back through `/api/public/v2/observations` |
| **Tavily** | — | ✔ | live rate-limit / pricing lookups inside the migration scenario |

Every row above came from the Studio driving the real services; the four scenario runs are saved under `demo/runs/` and can be replayed from the UI without any network.

### Gotchas found live (and what the repo does about them)

These are the kind of findings an FDE hands back to the platform team. Each one is now a check, a setting, or a documented workaround.

1. **Streamed usage is cumulative.** Token Factory sends `usage` in *every* streamed chunk with running totals (vLLM `continuous_usage_stats` behaviour). PydanticAI adds per-chunk usage unless told otherwise, which inflated one run to 635k input tokens. Fix: `OpenAIChatModelSettings(openai_continuous_usage_stats=True)` in `provider.py`; the kit's `streaming.text` check now reports `usage cumulative in N chunks`.
2. **Parallel tool-call deltas can land on the wrong index.** With five tool calls in one streamed turn, the tail of call N's JSON arguments arrived on call N+1 (`…"max_tokens": 16` / `0}{"query": …`), so both calls failed validation. New check `streaming.parallel_tool_call_deltas`; the Studio's orchestrator runs its own model requests non-streamed (`agent.iter`) and keeps streaming on the partner-facing legs.
3. **Several new models are not agent-ready yet.** Kimi-K3 and Kimi-K2.7-Code return empty `content` on plain chat and ignore `tool_choice`; DeepSeek-V4.1-Flash ignores `json_schema`. The kit makes this a table instead of a support ticket.
4. **Cartesia SDK 4.x** returns a coroutine from `tts.bytes()`; `voice.py` accepts bytes, iterators or awaitables.
5. **Dedicated endpoints answer cold.** The first request on a freshly `ready` replica took 1.3–5.4 s to first token; after five requests the same endpoint served 111 ms. The scout measures cold once, warms, then reports the warm number and says so. The control plane's live shape also differs from the docs (`id` not `endpoint_id`; status under `deployment.status` / `deployment.readiness`; `routing_key` is `dedicated/<model>-<suffix>`), which `dedicated.py` now follows.

### Daytona next to Nebius Sandboxes

Token Factory now has its own Sandboxes (beta): one *operation* per run, fresh compute from an image, optional filesystem snapshot. Daytona stays the partner pick here because a coding agent wants what the beta does not offer yet: long-lived sandboxes with `fs.upload_file` / `process.exec` round-trips, git and LSP inside the box, preview URLs and snapshots. The sandbox interface in `sandbox.py` is one class per backend, so a Nebius Sandboxes backend is a 40-line addition when the API leaves beta.

## 9. Repo map

```
src/nebius_agent/   config.py provider.py tools.py agent.py tracing.py cli.py server.py
                    studio.py studio_agent.py sandbox.py voice.py probes/ static/   # the Studio
src/conformance/    checks.py run.py            # framework-agnostic kit
src/mock_server/    server.py                   # deterministic OpenAI-compatible fake
tests/              15 tests through PydanticAI's abstractions + the kit + the service
examples/           partner_quickstart.py
deploy/             k8s.yaml      Dockerfile    scripts/demo.sh
docs/               ARCHITECTURE.md  PARTNER_PLAYBOOK.md  RECOMMENDATION.md
BUILD_LOG.md        how this was built (tools, models, decisions, what changed through testing)
```

License: Apache-2.0.
