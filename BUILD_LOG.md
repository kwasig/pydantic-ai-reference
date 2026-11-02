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
