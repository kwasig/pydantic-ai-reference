# Ecosystem recommendation: three partners, two quarters, four FDEs

*To the Nebius exec team · from Kwasi Ankomah · Nov 2026 · one page*

**Recommendation.** Over the next two quarters, land **Langfuse** (observability), **Cartesia** (voice) and **Daytona** (agent sandboxes), in that order, and staff four FDE hires so the same kit lands ten partners by the end of Q2. The choice is deliberate: one partner proves where Token Factory's premium inference wins (voice), one gives us distribution into coding agents (sandboxes), and one instruments everything so field learning becomes post-training data on our own platform (observability). None duplicates something we already bought (Tavily, ConTree, Clarifai).

## The three tests we applied

1. **Premium tokens.** Does the partner put Token Factory in a path where latency *is* the product?
2. **Reach we don't have.** Does it bring developers who would not otherwise pick Nebius?
3. **The loop.** Does it feed traces → evals → datasets → post-training on Token Factory?

## The partners

| # | Partner | Why now | What Nebius offers | What we get | Lands |
|---|---|---|---|---|---|
| 1 | **Langfuse** — open-source LLM observability, OTel-native, self-hostable | Cheapest, fastest, and it instruments partners 2 and 3. Every reference agent we ship already emits OTel (see repo). | "Langfuse on Nebius": a one-command self-host on Managed Kubernetes next to the inference it traces; Token Factory as a default model in Langfuse playground/evals; Builder Program credits and co-marketing | Traces on partner traffic → eval datasets → post-training with the Eigen team. The flywheel that makes open models on Nebius better than the same weights elsewhere | Q1, weeks 1–6 |
| 2 | **Cartesia** — real-time voice (Sonic TTS, Ink STT, Line agent platform) | Voice has a ~200 ms conversational budget; the LLM gets a slice. It is the clearest case where fast inference wins, and where we have to prove it on their traffic, not a leaderboard | Dedicated Token Factory endpoints with a TTFT SLA for the LLM leg (Inferize scaling absorbs call spikes); Nebius GPU capacity to host Cartesia's own models; a joint "voice agent on Nebius" reference | Premium real-time tokens and the public proof point that Token Factory is the inference for voice. Hume on SambaNova is the precedent | Start Q1 week 3, live by end Q2 |
| 3 | **Daytona** — open-source sandboxes for agents | Coding agents are loops: every token is many tokens. Daytona has the developer base; we have the fast decode | Daytona's sandbox fleet on Nebius compute; Token Factory as the default model in Daytona's agent templates; joint coding-agent reference (this repo's `run_python` already supports Daytona and ConTree) | Coding-agent token volume and a route into OSS harnesses (OpenHands, OpenCode) that already use Daytona | Q2 |

**Why this order.** Langfuse first because it takes weeks and instruments everything after it. Cartesia starts almost immediately because it is the longest cycle and the biggest revenue proof. Daytona in Q2, after we have agreed how it sits next to ConTree.

## Four hires, and how they get us to ten

| Hire | Owns | Lands | Sets up |
|---|---|---|---|
| **FDE-1 Agents & frameworks** | the conformance kit and reference agents; frameworks #2–#5 (LangGraph tests, CrewAI, OpenAI Agents SDK, Mastra) | Daytona (with FDE-4) | Tier-2 self-serve: partners run `verify`, we review |
| **FDE-2 Real-time & voice** | TTFT engineering, dedicated endpoints, work with the Inferize team | Cartesia | the premium real-time tier playbook for the next voice partner |
| **FDE-3 Observability & evals** | Langfuse on Nebius, OTel conventions, traces → datasets → post-training with Eigen | Langfuse | the loop every later partner plugs into |
| **FDE-4 Platform & developer relations** | partner tiering, Builder Program office hours, docs generated from the kit, reference repos | Daytona (with FDE-1) | the machinery that makes partners 4–10 a kit-and-office-hours job, not an engineering project |

Tiering is the scaling rule: Tier 1 gets an FDE (the three above); Tier 2 gets the kit, a review and office hours; Tier 3 gets docs generated from the kit's reports. Ten partners by end of Q2 means three Tier 1, four Tier 2, three Tier 3.

## The trade-off we are making

We are choosing **premium-inference proof and developer distribution over enterprise breadth.** No enterprise ISV (Palantir-style) and no gateway (OpenRouter, LiteLLM) in the first three; those stay with core engineering and sales for two quarters. And we are choosing Daytona's distribution over exclusivity for ConTree: ConTree stays the first-party default in our docs; Daytona is a supported backend for developers already there.

## One concrete ask

**A standing partner capacity reserve owned by FDE: two dedicated Token Factory endpoints and a $150k credits pool**, so a partner integration never waits on a sales order, and a named Token Factory product counterpart who takes the kit's FAIL reports as roadmap input.

## How we will know it worked (end of Q2)

Three partner references live with public READMEs and conformance reports · conformance reports for the top ten served models · Cartesia p50 TTFT under 300 ms on a dedicated endpoint · share of partner traffic traced into Langfuse · tokens per month arriving through partner integrations, reported monthly.
