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
