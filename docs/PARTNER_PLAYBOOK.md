# Partner integration playbook — how to use this kit for framework #2…#10

The point of this repo is not PydanticAI. It is a repeatable way to say "framework X works on
Token Factory" and mean it.

## Day 0 — before touching the framework (30 min)

```bash
nebius-agent verify --model <the model the partner wants> --json reports/<model>.json
```

Read the report. Three outcomes:

| Report | What it means | What you do |
|---|---|---|
| all PASS | endpoint + model are agent-ready | go to Day 1 |
| PARTIAL on `tools.parallel` / `cache.cached_tokens` | fine for most agents; note it in the partner README | go to Day 1, set expectations |
| FAIL on `tools.call_emitted` / `structured.json_schema` | the model, not the framework, is the problem | pick another model or file a field-to-product ticket with the JSON report attached |

## Day 1 — the framework (half a day)

1. Copy `src/nebius_agent/provider.py` and adapt the one provider call to the framework's OpenAI-compatible client. Keep the model profile.
2. Write `tests/test_<framework>_through_framework.py` with the same five tests as [`test_agent_through_framework.py`](../tests/test_agent_through_framework.py): tool reaches your Python, structured output validates, streaming yields >1 delta, spans follow GenAI semconv, schemas are well-formed.
3. Run against the mock first (`make test`), then against Token Factory.
4. Wire tracing. If the framework has OTel, point it at `tracing.configure_tracing()`. If it has its own tracer (OpenAI Agents SDK, LangSmith), add an exporter/processor and say so in the README.

## Day 2 — package for the partner

* README with: the 25-line snippet, the conformance table for the model(s) they'll use, the env vars, and the Status table.
* A `partner_quickstart.py` they can run in 60 seconds.
* The conformance JSON report checked in under `reports/`.
* CI: offline job on every PR; weekly job against Token Factory with a secret, so drift shows up before the partner sees it.

## What goes back to Product

Every FAIL and every PARTIAL is a ticket with: model, endpoint, the check name, the raw
request/response from the report, and which partner it blocks. The kit makes field-to-product a
copy-paste, not a meeting.

## Scaling rule

Tier 1 partners (engineering time): Day 0 + Day 1 + Day 2 by an FDE, joint README.
Tier 2 partners (kit + office hours): they run `verify`, we review the report, they write the tests.
Tier 3 (self-serve): docs page generated from the kit's markdown table.
