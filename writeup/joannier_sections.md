# Joannier's sections for the Mini-Project writeup

Paste each section into the corresponding heading in `Mini-Project MasterDoc`. Numbers are pulled from `results/takeaways.md`, `results/engine_analysis/summary.csv`, and `results/engine_analysis/hyp_tests.txt`.

---

## Infrastructure (Joannier)

We serve Llama-3.1-8B-Instruct with vLLM on a Cloud TPU v5litepod-4 (4 chips, 16 GB HBM each, 2x2 topology) in GCP zone us-south1-a. The model is sharded across all four chips with tensor parallelism (TP=4) at bfloat16, with a maximum context length of 8,192 tokens. After weights are loaded, the engine reports 13.5 GiB used of a 58 GiB usable HBM budget, leaving a KV cache of 363,776 tokens — roughly 44 concurrent full-context requests — so KV cache capacity is never a binding constraint at our benchmark's concurrency of 2.

Rather than installing vLLM directly on the TPU VM, we run the official `vllm/vllm-tpu:nightly` Docker image. This decision was forced by experience: vLLM's TPU backend requires torch, torch_xla, libtpu, and JAX at tightly matched versions, and the pip installation path failed in three distinct ways across versions we tried (current vLLM requires a torch newer than torch_xla's TPU wheels support; an older pinned vLLM expected paged-attention XLA ops introduced only in later torch_xla; and the transformers release in between used float8 dtypes absent from the older torch). The container ships a pre-validated combination of all four, reducing the host requirements to Docker and PostgreSQL. We consider this a practical finding in itself: on TPU, the deployment unit for vLLM is the container, not the Python package.

Model choice was similarly constrained by the serving stack. The vllm-tpu image routes models through `tpu_inference`, which has JAX-native implementations for a specific list of architectures (Llama, Qwen, Gemma, DeepSeek, among others). Mistral-7B — one of our proposal's candidates — is not on that list; it falls back to a PyTorch-based path that segfaulted during engine initialization. Qwen2.5-7B failed differently: the loader assumes a multimodal-style config exposing `text_config`, which Qwen2's text-only config lacks. Llama-3.1-8B-Instruct (served from an ungated mirror) uses the well-tested JAX-native `LlamaForCausalLM` path and worked without modification. The attention backend auto-selects FLASH_ATTN, a Pallas flash-attention kernel.

The two engine knobs under study — prefix caching (RadixAttention) and chunked prefill — are passed as startup flags (`--enable-prefix-caching`, `--enable-chunked-prefill` and their negations), giving the four engine presets used in the sweep. Each preset is held fixed for an entire benchmark run; switching presets requires a container restart, which takes about three minutes with weights cached on local disk (weight load ~6 minutes on first boot, then XLA precompilation of the prefill, decode, sampling, and structured-decoding graphs adds ~100 seconds).

The serving endpoint is OpenAI-compatible on port 8000 and is reached from client machines over an SSH tunnel, so no ports are exposed publicly. PostgreSQL runs on the same VM as the SQL execution target, keeping tool-call latency off the measurement path: the agent's SQL executor and the inference engine share a host, so round latency is dominated by inference rather than network hops. Engine telemetry comes from vLLM's Prometheus `/metrics` endpoint, which we snapshot before and after each sweep to derive TTFT and end-to-end latency from histogram sum/count pairs, along with prefix cache hit rate and KV cache utilization.

A single smoke-test request (96 completion tokens) completes in 0.73 s end-to-end with a 0.16 s time-to-first-token, confirming the endpoint is interactive-grade before benchmarking. The VM costs roughly $115/day while running, so the operational discipline was to keep it stopped except during active sweeps; the full benchmark campaign consumed only a few hours of TPU time.

---

## Prefix Caching (Joannier)

*(Results section — pairs with the engine-level table)*

Prefix caching behaved almost exactly as hypothesized. From the Prometheus engine metrics:

| Engine config | Mean TTFT | Mean e2e latency |
|---|---|---|
| prefix_on + chunked_on | **26.8 ms** | 546 ms |
| prefix_on + chunked_off | **27.3 ms** | 556 ms |
| prefix_off + chunked_on | 51.7 ms | 592 ms |
| prefix_off + chunked_off | 51.9 ms | 595 ms |

Enabling prefix caching cuts mean time-to-first-token by ~48%, from ~52 ms to ~27 ms — squarely within the 30–60% reduction our proposal predicted. The mechanism is the one we hypothesized: for a given task, every reflection round shares an identical leading context (system prompt, database schema, and the original broken query), so RadixAttention serves that prefix from cache and only the incremental content of each round — the previous attempt and its tool result — is prefilled fresh.

The effect compounds with reflection depth. Prompt length grows from a mean of 547 tokens in round 1 to 1,242 tokens by round 5 (2.3×, below our 3–5× prediction; Spider schemas and error messages are compact). Per-round latency tracks this growth, but more steeply without caching: round 5 takes 1.8× round-1 latency with prefix caching on (0.475 s → 0.849 s) versus 2.0× with it off (0.483 s → 0.954 s). The cached prefix grows as a fraction of the total prompt precisely when the prompt is longest, which is where the savings matter most.

At the application level the benefit is visible but not statistically significant: median task latency is 3.05 s with prefix caching versus 3.25–3.38 s without (Mann-Whitney p=0.08–0.09), about 0.3 s per task or ~30 s across the 100-task benchmark. Success rate, tokens consumed, and rounds-to-success are all unchanged (all pairwise p > 0.37) — which is the expected result: prefix caching changes how fast each round is served, not what the model generates. The engine-level numbers are direct measurements rather than noisy task outcomes, so we consider the 48% TTFT reduction the firmest quantitative finding of the project.

---

## Chunked Prefill (Joannier)

*(Results section)*

Our proposal predicted chunked prefill would reduce tail latency under concurrent load by preventing long late-round prefills from blocking shorter requests (head-of-line blocking). The data shows no such effect at any percentile:

| Engine config | p50 | p75 | p95 | p99 |
|---|---|---|---|---|
| prefix_on + chunked_on | 3.05 s | 3.69 s | 5.06 s | 6.46 s |
| prefix_on + chunked_off | 3.07 s | 3.99 s | 4.99 s | 6.28 s |
| prefix_off + chunked_on | 3.25 s | 4.22 s | 5.14 s | 6.36 s |
| prefix_off + chunked_off | 3.38 s | 4.03 s | 5.33 s | 6.57 s |

Within each prefix-caching group, the chunked-on and chunked-off distributions differ by less than 0.2 s at p95/p99 and the direction is inconsistent. Mean TTFT is likewise indistinguishable (26.8 vs 27.3 ms with prefix on; 51.7 vs 51.9 ms with prefix off).

We attribute this to experimental scale rather than to the optimization itself: the hypothesis is untestable at concurrency 2, not refuted. Chunked prefill helps by interleaving decode steps of running requests with slices of a long incoming prefill; that requires a queue deep enough for long and short requests to actually contend. With only two concurrent tasks — and per-round prompts that max out around 1,200–2,000 tokens, below or near our 2,048-token `max_num_batched_tokens` budget — a round-5 prefill almost never blocks a round-1 request from the other task. Two conditions would make the predicted effect visible: higher concurrency (8–16 parallel tasks, where queueing is routine) and longer prompts (verbose schemas or accumulated context that exceeds the batched-token budget, forcing multi-chunk prefills). Both are within reach of this setup and are natural follow-on experiments; the v5litepod-4's 44× KV-cache concurrency headroom means the engine itself would not be the bottleneck.

---

## Optimizations (Joannier and Ann)

*(Conceptual analysis of a fully deployed system. I drafted the systems half — Ann should add/edit the application-level items marked below.)*

Our measurements point to four optimization opportunities for a production deployment of a reflective agent, in decreasing order of confidence.

**1. Prefix caching is necessary but incompletely exploited (measured).** The 48% TTFT reduction came from caching the static per-task prefix. In a deployed multi-tenant system the cacheable surface is larger: the system prompt and tool descriptions are shared across *all* tasks, and schema definitions are shared across all tasks targeting the same database. A schema-aware router that assigns tasks for the same database to the same engine replica would raise the cross-request cache hit rate well above what our single-task-stream benchmark achieves. The complementary lever is prompt layout: ordering context from most-shared to least-shared (system prompt → tool spec → schema → task → history) maximizes the reusable prefix, which our agent already does by construction.

**2. Cross-round KV reuse beyond the prefix (conceptual).** Prefix caching only reuses the *unchanged leading* portion of the prompt. In reflection workloads, rounds n and n+1 share nearly the entire round-n prompt — the new content is appended, not interleaved. Engine-side support for append-only conversations (retaining the full per-task KV state between rounds rather than re-matching the radix tree from scratch) would reduce round-n prefill to only the newly appended tokens, turning our observed 1.8× round-5 latency growth into near-constant per-round latency. This is the single largest remaining systems win for the reflection pattern, since per-round prompt growth is intrinsic to it.

**3. Right-size the reflection budget by difficulty tier (measured, application-level — Ann).** Success by round is strongly tier-dependent: easy tasks that succeed do so in early rounds, while hard tasks rarely succeed at any budget (15–24%). A deployed system should not spend a uniform 5-round budget; an early-exit policy (stop when the same error repeats or attempts stop changing materially) and a tier-conditioned budget would cut the largest cost item — failed tasks that burn all 5 rounds and the maximum token count. Our verbosity result reinforces this: compact prompts *increased* median tokens per task (4,302 vs 4,210) because they lowered success and pushed more tasks to budget exhaustion. Token-saving measures that reduce signal are counterproductive; budget-saving measures that detect futility are not.

**4. Concurrency scaling and chunked prefill (conceptual).** At concurrency 2 the engine is heavily underutilized (44× KV-cache headroom) and chunked prefill has nothing to do. A deployment would batch many agent tasks per replica, which raises throughput but re-introduces exactly the head-of-line risk chunked prefill addresses — our null result should not be read as license to disable it at scale. The interesting deployment question is the throughput/latency trade-off curve as concurrency rises toward the KV limit, with chunked prefill expected to flatten the tail; measuring that curve at concurrency 8–32 is the experiment our setup was one knob away from running.

Finally, the capability ceiling matters for where optimization effort goes: no engine configuration moved easy-tier (saturated at 52.9% everywhere) or hard-tier success materially. Infrastructure optimizations in this regime buy latency and cost, not accuracy — closing the hard tier requires a stronger model or richer tools (e.g., loading row data so the agent can verify results), not a faster engine.
