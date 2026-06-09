# Benchmarking Takeaways — Reflective SQL Debugging Agent

## Setup

100 Spider tasks (34 easy / 33 medium / 33 hard), max 5 reflection rounds, concurrency=2. Swept two variables: prompt verbosity (full vs compact) and engine configuration (prefix caching × chunked prefill, 4 presets). All runs on Llama-3.1-8B-Instruct via vLLM on Cloud TPU v5litepod-4.

---

## 1. Overall Agent Performance

The agent solved **33–40% of tasks** depending on condition. Success breaks down sharply by tier:

| Tier | Success rate (all configs) |
|---|---|
| Easy | **52.9% — invariant across every engine config** |
| Medium | 27–52% (most sensitive to condition) |
| Hard | 15–24% (universally difficult) |

The easy tier is fully saturated by the model's capability — engine configuration moves it zero. The hard tier is a ceiling effect in the other direction: multi-join schema reasoning and query plan interpretation exceed what Llama-3.1-8B can reliably do in 5 rounds regardless of infrastructure.

---

## 2. Reflection Budget Is the Dominant Application Lever

| Budget | Typical success rate |
|---|---|
| 1 round | **0%** — no task solvable in a single shot |
| 2 rounds | ~1% |
| 3 rounds | 17–22% |
| 4 rounds | 25–34% |
| 5 rounds | 33–40% |

**Zero tasks were solved in one round.** Every success required at least two, and most required three or more. This directly validates the reflection pattern: the agent genuinely needs iterative feedback to correct its mistakes. The curve has not saturated — extending to 10 rounds would likely continue recovering tasks, particularly in the medium tier.

## 3. Prefill Cost Grows with Reflection Depth (Partially Confirmed)

The proposal predicted prompt length would reach 3–5× round 1 by round 5. The measured growth is real but more modest:

| Round | Mean prompt tokens | Mean per-round latency (prefix_on) | Mean per-round latency (prefix_off) |
|---|---|---|---|
| 1 | 547 | 0.475s | 0.483s |
| 2 | 650 | 0.460s | 0.498s |
| 3 | 829 | 0.797s | 0.835s |
| 4 | 1030 | 0.743s | 0.780s |
| 5 | 1242 | 0.849s | 0.954s |

Prompt length grows **2.3× by round 5** (547 → 1242 tokens), not 3–5× as predicted. Spider tasks have relatively compact schemas and error messages, keeping growth below the upper-end estimate. Per-round latency tracks the prompt growth — round 5 takes ~1.8× longer than round 1 with prefix caching on, and ~2.0× longer without. This confirms the proposal's core mechanism: prefill cost grows with reflection depth, and the growth is steeper without prefix caching because each round must reprocess the full growing context from scratch.

The proposal claimed decode time per round should remain roughly constant. We cannot isolate prefill from decode in the per-round latency (our measurements are total round wall time including TTFT + decode + network). However, the latency growth pattern — rising in proportion to prompt length — is consistent with prefill growth driving the increase, since decode output length is approximately constant across rounds (~20–30 completion tokens per round).

---

## 4. Prefix Caching: Hypothesis Confirmed (~48% TTFT Reduction)

From the Prometheus engine metrics:

| Engine config | Mean TTFT | Mean e2e latency |
|---|---|---|
| prefix_on + chunked_on | **26.8 ms** | 546 ms |
| prefix_on + chunked_off | **27.3 ms** | 556 ms |
| prefix_off + chunked_on | **51.7 ms** | 592 ms |
| prefix_off + chunked_off | **51.9 ms** | 595 ms |

Prefix caching cuts TTFT by **~48%** (from ~52ms to ~27ms). The proposal predicted 30–60% — we landed squarely in the middle. The mechanism is exactly as hypothesized: the system prompt, schema definition, and original broken query are identical across all reflection rounds for a given task, so RadixAttention reuses their KV cache. Only the incremental round content (prior attempt + tool result) needs to be prefilled fresh.

The benefit flows through to application-level latency: median task latency is **3.05s with prefix caching vs 3.25–3.38s without** (~0.3s per task, ~30s over the full 100-task benchmark).

At the application level (success rate, tokens, rounds), all pairwise comparisons are **non-significant** — the engine optimization changes how fast each round is served, not whether the model gets the answer right.

---

## 5. Chunked Prefill: No Measurable Effect, Including at the Tail

The proposal specifically predicted chunked prefill would reduce **tail latency** under concurrent load by preventing head-of-line blocking. Measured p50/p95/p99 per engine config:

| Engine config | p50 | p75 | p95 | p99 |
|---|---|---|---|---|
| prefix_on + chunked_on | 3.05s | 3.69s | 5.06s | 6.46s |
| prefix_on + chunked_off | 3.07s | 3.99s | 4.99s | 6.28s |
| prefix_off + chunked_on | 3.25s | 4.22s | 5.14s | 6.36s |
| prefix_off + chunked_off | 3.38s | 4.03s | 5.33s | 6.57s |

Chunked prefill shows no consistent improvement at any percentile — within each prefix group the p95 and p99 differences are <0.2s and inconsistent in direction. The hypothesis is not refuted but untestable at this scale: **concurrency=2 is too low to create head-of-line blocking**. Chunked prefill is designed to interleave decode tokens from queued requests while long prefills are processing. With only 2 concurrent tasks, the queue is almost never deep enough for a long-prefill round-5 request to block a short round-1 request. The effect would likely become visible at concurrency=8–16.

---

## 6. Verbosity: No Significant Effect, Slightly Counterproductive

Full verbosity achieves **39% success vs 35% for compact** on the same engine (prefix_on+chunked_on). The 4pp gap is not statistically significant (Fisher's exact p=0.66), but the direction is consistent across engine configs.

The token numbers are counterintuitive: compact uses **more tokens on average** (median 4302 vs 4210). This is because compact fails more tasks — failed tasks always exhaust all 5 rounds and accumulate the maximum tokens, while full verbosity succeeds sooner and spends fewer tokens per task overall. The additional context in full tool output gives the model marginally more signal to correct its reasoning.

---

## 7. Medium Tier Is Where Conditions Diverge Most

| Config | Medium success |
|---|---|
| prefix_on + chunked_off | **51.5%** |
| prefix_on + chunked_on | 39.4% |
| prefix_off + chunked_on | 36.4% |
| compact (prefix_on + chunked_on) | 33.3% |
| prefix_off + chunked_off | 27.3% |

The variance (27–52%) across 33 medium tasks is the largest of any tier. Medium tasks have longer, more complex schemas and require more multi-round reasoning — exactly the regime where prefix caching saves the most prefill work and where faster token delivery may allow more rounds within a wall-clock budget. n=33 per tier is underpowered to confirm this statistically, but the direction is consistent with prefix caching helping.

---

## 8. Statistical Caveat

All application-level comparisons (success rate, tokens, latency) are non-significant at α=0.05. With n=100 tasks divided into three tiers of ~33 each, detecting a 5pp difference in overall success rate requires roughly n=400 for standard power. The engine-level TTFT numbers (from Prometheus) are direct measurements of infrastructure behavior rather than noisy outcome metrics, so those conclusions are on firmer ground.

---

## Summary Table

| Finding | Direction | Significant? |
|---|---|---|
| Prefix caching → TTFT | −48% (52ms → 27ms) | Yes (direct measurement) |
| Prefix caching → task latency | −0.3s median | Trend (p=0.08) |
| Prefix caching → success rate | +6pp trend | Not sig. (p=0.46) |
| Prompt length growth round 1→5 | +2.3× (547 → 1242 tokens) | Confirmed (below 3–5× prediction) |
| Per-round latency growth round 1→5 | +1.8–2.0× | Confirmed, steeper without prefix cache |
| Chunked prefill → mean TTFT/e2e | ~0 | Not sig. |
| Chunked prefill → tail latency (p95/p99) | ~0 | Not sig. (concurrency too low) |
| Full vs compact verbosity → success | +4pp trend (full better) | Not sig. (p=0.66) |
| Budget 1→5 rounds → success | 0% → 33–40% | Large, monotonic |
| Easy tier → engine sensitivity | Zero effect | Invariant |
| Hard tier → engine sensitivity | Minimal | Near-ceiling failure |
