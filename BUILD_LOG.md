# Build log — how this was made

The exercise asks for a record of how the work was built. This is it, written as the work
happened. The raw record is the Claude session transcript (exported alongside this repo).

## Tooling choice, and why

* **Coding tool: Claude (Claude Code harness, claude.ai).** I wanted an agent that can run the
  code it writes: install packages, start the mock server, run pytest, read tracebacks and fix
  them, rather than paste suggestions. Claude Code gives a shell, file tools and a task list in
  one loop, and the transcript is a complete audit trail of every command and edit.
* **Model: `claude-fable-5-1`.** The strongest model I have access to for long, multi-file
  engineering sessions where the agent has to hold a plan, inspect an unfamiliar library version
  and recover from test failures. The exercise needs judgment about *what* to build as much as
  code; I did not want to trade that for a cheaper model.
* **What I did myself:** the framework choice, the partner choices, the architecture, the
  conformance check list, the model-profile decisions, and reviewing every file. What the agent
  did: typed the code, ran it, fixed what broke.

## Timeline (Nov 1–2, 2026, one session)

| Step | What happened | Decision / what changed |
|---|---|---|
| 1 | Re-read the brief. Checked Nebius Token Factory docs: LangChain, CrewAI and PydanticAI already have doc pages; LangChain has a first-party partnership (May 2026). | Framework = **PydanticAI**: typed tools + outputs expose exactly where open models break, native OpenTelemetry, and it is not the one Nebius already owns. The reusable piece (conformance kit) matters more than the framework. |
| 2 | `pip index versions pydantic-ai` → 2.64.0, newer than I knew. Inspected the installed API (`OpenAIChatModel`, `OpenAIProvider`, `Agent.instrument_all`, `Instrumentation` capability, `ToolOutput`/`NativeOutput`/`PromptedOutput`, `run_stream`). | Wrote against the real 2.64 surface, not memory. Two deprecations surfaced later (`openai_supports_tool_choice_required` → `supports_forced_tool_choice`; `result.usage()` → `.usage`) and were fixed. |
| 3 | No Token Factory key in the build sandbox and no outbound network. | Built a **deterministic OpenAI-compatible mock** (`src/mock_server`) that honours tool schemas, `response_format`, streaming deltas and `cached_tokens`. Everything is testable offline; the real endpoint is one env var away. |
| 4 | Wrote `provider.py` with a Token Factory model profile. First attempt used a non-existent profile key (`openai_supports_sampling_settings`). | Read `OpenAIModelProfile.__annotations__`; kept only real keys: no `strict`, `max_tokens`, `reasoning_content`, auto-cache, tool-mode structured output by default. |
| 5 | Wrote tools, agent, tracing, CLI. First `nebius-agent run` against the mock worked: tool call → `final_result` → validated `ModelRecommendation`, trace written. | Kept `final_result` tool mode as default; `native` and `prompted` as switches. |
| 6 | Package layout: editable install with `where=["src", "."]` did not expose `conformance`. | Moved `conformance` and `mock_server` under `src/`. |
| 7 | Conformance kit: 12 checks with PASS/PARTIAL/FAIL and evidence. 12/12 on the mock. | PARTIAL exists so "sequential tool calls only" and "cache field present but 0" are reported, not failed. |
| 8 | Tests through PydanticAI. Two failures: (a) streaming returned one delta; (b) `prompted` output mode failed validation. | (a) PydanticAI debounces `stream_text` by 100 ms and the mock is faster than that → `debounce_by=None` in the test and a 4 ms per-chunk delay in the mock so TTFT/tokens-per-s are measurable. (b) prompted mode sends the schema in the instructions with `json_object`; the mock now finds the schema in the prompt and honours it, like a real model would. |
| 9 | Added the FastAPI service, Dockerfile, Kubernetes manifest, service tests (SSE streaming included). | This is the answer to "how would the runtime run on Nebius infrastructure". |
| 10 | Mock only ever called the first tool, so demos didn't show the multi-step loop. | Mock now keeps calling tools the prompt asks for by keyword until each has run once: `list_models` → `benchmark_model` → `final_result`. Sample trace regenerated. |
| 11 | README, architecture, partner playbook, recommendation, CI (offline matrix + weekly Token Factory job with a secret). | — |

Final state: 15 tests pass, 12/12 conformance checks pass against the mock.

## Not done here, to do before the panel

* Run `make verify-real` and `nebius-agent run ...` against Token Factory with the credits
  Alix sent; paste the report into the README Status table. Expect PARTIAL on `tools.parallel`
  and `cache.cached_tokens` for some models; that is useful information, not a failure of the kit.
* Pick `NEBIUS_MODEL` from `nebius-agent models` (a tool-calling model; Qwen3 or DeepSeek first).
* Optional: `TAVILY_API_KEY` for live grounding, Langfuse keys for the OTLP demo.

## Session 2 (Nov 1–2, 2026): the Studio

Feedback after the first pass: "all telemetry, no app". Fair. The second session built the agent
as a product.

| Step | What happened | Decision / what changed |
|---|---|---|
| 12 | Checked outbound network from the build sandbox: Token Factory, Daytona, Cartesia, Tavily and Langfuse are all refused by the gateway. | Everything here is verified against the mock; `scripts/run_live.sh` runs the same thing against the real services on the candidate's machine. Said so plainly in the README Status table. |
| 13 | Installed the Daytona and Cartesia SDKs from PyPI and inspected them (`AsyncDaytona.create(CreateSandboxFromImageParams)`, `sandbox.fs.upload_file`, `sandbox.process.exec`; `AsyncCartesia.tts.bytes(model_id, transcript, voice, output_format)`). | `sandbox.py`: one interface, Daytona in production, local subprocess in dev, disabled otherwise; `voice.py`: Cartesia with a browser-speech fallback that is labelled as such. |
| 14 | Wrote framework probes (PydanticAI, raw OpenAI SDK, LangGraph): self-contained scripts that run inside the sandbox, define a tool, run one turn and print `PROBE_RESULT {...}` with `tool_called`. | This is how the Studio *proves* an integration instead of asserting it. |
| 15 | Studio agent: `run_probe`, `run_conformance`, `benchmark`, `web_search`, typed `FdeReport` with a 60-word `spoken_summary`. Tools emit progress events into the UI. | First run against the mock only called `benchmark`: the mock picked tools by prompt keywords and had none for probe/conformance. Taught the mock a sensible order (list → probe → kit → benchmark) and keywords. |
| 16 | Daytona call from the build sandbox failed with the proxy's 403 and crashed the run. | Tools must never crash a run: `run_probe` now catches sandbox errors, falls back to the local sandbox when allowed, and reports the fallback in the result. Tests pin `daytona_api_key=""` so `.env` cannot leak into CI. |
| 17 | FastAPI Studio: run manager, SSE event stream with replay, status endpoint that pings the endpoint, audio serving; the UI (scenarios, live timeline, sandbox terminal, conformance grid, benchmark chart, report, voice, trace tree). | |
| 18 | Drove the UI in headless Chromium with Playwright. First attempt: the page never started a run. | The sandbox blocks CDNs, so `Chart` was undefined and `drawChart()` threw before the run began. Vendored Chart.js into `static/` from the npm registry and guarded the chart code. Also stopped auto-scrolling the whole window. Screenshots in `docs/img/`. |
| 19 | `tests/test_studio.py`: starts a run through the API, follows the SSE stream, asserts the probe reported `tool_called=true`, 12/12 checks, a benchmark, a report, a voice result and spans. | 17 tests green. |

Not done here: the live run. `make live` on a machine with the keys does: models → conformance
report → one traced agent run → the Studio with Daytona and Cartesia on.

| 20 | Feedback: "make it Nebius-like, clean, and a real use case." | Reframed the Studio as **Integration Scout**: a partner describes the workload (type, framework, TTFT budget, volume) and gets an integration plan: verdict, model, measured numbers against the budget, the copy-paste snippet for their framework, a downloadable INTEGRATION.md, and a cost estimate if prices are configured. Light, clean UI with four brand tokens at the top of the stylesheet to swap for official values. Deliverables are derived from what was verified (which probe passed, which model was measured), never from the model's prose. Two more tests on the deliverables; 17 green. |

## Session 3 (Nov 1–2, 2026): live against the real services

The build sandbox cannot reach Token Factory, Daytona, Cartesia, Tavily or Langfuse, so the
candidate linked his Mac; Claude drove Finder/Terminal and the in-app browser there to run and
test the Studio against the real services, and edited code in the cloud workspace in between.

| Step | What happened | Decision / what changed |
|---|---|---|
| 21 | Branding: pulled computed styles from nebius.com in a real browser (not a blog): ink `#052B42`, lime `#E0FF4F` (the logo block), blue `#3277FF`, violet `#614EFA`, grey `#F3F4F5`; Inter body, Space Mono code. | Rebuilt the Studio tokens around those; navy header with a lime block, lime "verified" badges. No Nebius logo file is used, only the palette. |
| 22 | First `make live` on the Mac exited after listing models. | `nebius-agent models \| head` + `pipefail` = SIGPIPE. Script now writes `reports/models.txt`, logs to `logs/live.log`, picks a served model if the configured one is missing, and no longer echoes key values. Added `Start Scout.command` so the launcher is a double-click. |
| 23 | Live: 30 models served; **12/12 checks PASS** on `Qwen/Qwen3-235B-A22B-Instruct-2507`; traced CLI run (`list_models`, 2× `benchmark_model`, 2,992 cached prompt tokens); Studio run with a real Daytona sandbox (`python:3.12-slim`, ready in 15 s, probe `tool_called=true`), two benchmarks, a verdict. | Two defects surfaced: Cartesia 4.x returns a coroutine from `tts.bytes()` (voice fell back to browser speech) and the agent skipped the kit. Fixed `voice.py` to accept bytes / iterators / awaitables; instructions now require probe + kit + benchmark. |
| 24 | Usage read **635,225 input tokens for 5 requests**. | Token Factory streams cumulative `usage` in every chunk; PydanticAI sums unless `openai_continuous_usage_stats=True`. Set it in `provider.py` (`token_factory_model_settings()`), and the kit's `streaming.text` now reports `usage cumulative in N chunks`. After the fix: 17,200 in / 871 out / 13,888 cached. |
| 25 | Parallel tool-call argument deltas landed on the wrong call index in one streamed turn, failing validation for both calls. | New check `streaming.parallel_tool_call_deltas` (13 checks). The orchestrator's own model requests run non-streamed; streaming stays on the partner-facing legs. |
| 26 | Pointed the scenarios at the newest served models (MiniMax-M3.1-Flash, Kimi-K2.7-Code, DeepSeek-V4.1-Flash-1017, Kimi-K3) with the verified default as the comparator; the "model to evaluate" selector no longer changes the scout's own model. | Live results in the README status table: Kimi-K3 8/13, Kimi-K2.7-Code 9/13, DeepSeek-V4.1-Flash 12/13 (json_schema ignored), MiniMax-M3.1-Flash 3.2 s TTFT. The kit turned "is it agent-ready?" into a table. |
| 27 | Langfuse keys arrived (`us.cloud.langfuse.com`). | `TRACE_EXPORTER=memory+otlp`: spans go to the UI and to Langfuse; the trace card links to the Langfuse trace; a Langfuse panel reads traces, model-call counts and p95 run latency *back* through the public API. Tavily key added; the migration scenario does a live rate-limit/pricing lookup. |
| 28 | Run export + replay: `/api/runs/{id}/export?save_as=name` writes `demo/runs/<name>.json` (+ audio); `POST /api/runs {"replay": name}` re-emits it with original timing. | All four scenarios recorded live and committed under `demo/runs/` so the panel demo survives a dead network. Test added; 18 green. |
| 29 | Secrets: `.env`, `logs/`, `audio/` ignored; the log was scrubbed after step 22 printed key values; the repo was grepped for every key prefix before the commit. | Only `.env.example` placeholders remain. |
| 30 | Studio UI v2. | A two-stage layout (one composer with scenario tiles, then run + plan), the five right-hand boxes folded into one plan with tabs, sentence-case headings, per-probe sandbox steps keyed by probe id, replay labels from the scenario, the launcher picks a free port. Deck and one-pager moved to the same palette. All four scenarios re-recorded live on the new UI; screenshots in `docs/img/` are replays of those recordings. |
| 31 | Coding-agent session in the sandbox. | New probe `coding_agent_probe.py` and tool `run_coding_session`: the agent on the model gets `write_file`/`run`, builds a module + tests, runs pytest to green, renders a chart; the probe re-runs pytest independently and ships the PNG back. The mock plays a scripted session so the offline demo produces real files and a chart. Coding scenario moved to Kimi-K3 with Kimi-K2.7-Code as comparator. Run live in Daytona: 5 tool calls, 6 tests green, chart in 24 s. |
| 32 | A benchmark reported 3,425 tok/s for Kimi-K2.7-Code. | Reasoning models stream `reasoning_content` before `content`; the clock started at the first content delta, so mostly-reasoning completions divided by a tiny tail. The clock now starts at the first token of either kind; results carry decode and end-to-end rates, chunk counts, reasoning tokens and a `stream_note` for buffered streams. Kimi-K2.7-Code on a 160-token budget: all reasoning, 32 s to first visible token. |
| 33 | "Why can't the scout move the workload to a dedicated endpoint itself?" | It can: `dedicated.py` wraps the Token Factory control plane (templates → create → poll `deployment.readiness` → delete), `provision_dedicated_endpoint` runs only with `ALLOW_DEDICATED=1`, names endpoints `scout-*`, deletes in `finally`, hard-caps the wait, and the Studio sweeps leftovers on startup. First live attempt: the poller matched the wrong keys (docs say `endpoint_id`, API returns `id`; status lives under `deployment`), so a run waited on an endpoint it could not see; fixed from the live list shape, the sweep cleaned up. Second attempt measured the dedicated endpoint cold (5.4 s). Third: measure cold, warm with five requests, measure again: Qwen3-30B 111 ms TTFT dedicated vs 268 ms shared, 0.1 billable minutes. Opt-in; documented in `.env.example`. |
| 34 | "Deploy the same app on Vercel; keep local working." | One flag, `STUDIO_STATELESS=1` (`api/index.py`): a single request runs the scout and streams it, audio and charts travel inline as data URLs, exports are off. Vercel's Python builder installs from `pyproject.toml`, so the Studio extras became base dependencies; `rewrites` broke the routing and were replaced by the FastAPI preset. Dedicated provisioning turned out to be a plan question, not a hosting one: the run holds one request for ~7 minutes, over Hobby's 300 s and under Pro's 800 s. The candidate upgraded to Pro; `maxDuration: 800`. |
| 35 | "Dedicated off" on the hosted app read as a setting; "should that be a switch?" | It is now a per-run switch in the composer, enabled when the server allows provisioning and off by default, so a run only pays for a GPU when the person asks. Tested on the hosted app from the cloud workspace: coding scenario 169 s; voice with the switch on 426 s (endpoint ready in 323 s, 77 ms warm vs 302 ms shared, deleted, 0.3 billable minutes, nothing left behind). One defect from that test: the trace card had no Langfuse link because the hosted runtime wraps the request in an outer span, so the agent span is no longer parentless; the trace id now comes from the agent span itself. |
