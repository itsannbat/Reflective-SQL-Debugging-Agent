# Joannier's sections for the Mini-Project writeup

Paste each section into the matching heading in `Mini-Project MasterDoc`. Numbers come from `results/takeaways.md`, `results/engine_analysis/summary.csv`, and `results/engine_analysis/hyp_tests.txt`.

---

## Infrastructure (Joannier)

We served Llama-3.1-8B-Instruct with vLLM on a Cloud TPU v5litepod-4 (4 chips, 16 GB HBM each, 2x2 topology) in GCP zone us-south1-a. The model was sharded across all four chips with tensor parallelism (TP=4) at bfloat16, with a max context length of 8,192 tokens. After weights loaded, the engine reported 13.5 GiB used out of a 58 GiB usable HBM. That left a KV cache of ~363,776 tokens, which is about 44 concurrent full-context requests, so KV cache capacity was never a constraint from our benchmark's of 2.

We ran the `vllm/vllm-tpu:nightly` Docker image instead of installing vLLM directly on the TPU VM. We tried the pip route first and it failed three different ways. The current vLLM release required a torch version newer than what torch_xla's TPU wheels support. Pinning an older vLLM didn't work either as it expected paged-attention XLA ops that only exist in later torch_xla versions. And the transformers version in between used float8 dtypes that the older torch doesn't have. The container ships a pre-validated combination of torch, torch_xla, libtpu, and JAX, so the host VM only needed Docker and PostgreSQL. This was a practical finding on its own: on TPU, the deployment unit for vLLM is the container, not the Python package.

Model choice was also constrained by the serving stack in GCP. The vllm-tpu image routes models through `tpu_inference`, which has JAX-native implementations for a fixed list of architectures (Llama, Qwen, Gemma, DeepSeek, and a few others). Mistral-7B, one of our proposal's candidates, is not on that list. It fell back to a PyTorch path that segfaulted during engine initialization. Qwen2.5-7B failed differently: the loader assumes a multimodal-style config with a `text_config` field that Qwen2's text-only config doesn't have. Llama-3.1-8B-Instruct uses the JAX-native `LlamaForCausalLM` path and worked without modification. The attention backend auto-selected flash_attn, a Pallas flash-attention kernel.

The two engine knobs for our study are prefix caching (RadixAttention) and chunked prefill, which we passed as startup flags. Their on/off combinations gave the four engine presets used in the sweep and each preset was held fixed for an entire benchmark. Switching presets required a container restart, which took about three minutes with weights cached on local disk. First boot was slower: weight load took ~6 minutes, and XLA precompilation of the prefill, decode, sampling, and structured-decoding graphs added another ~100 seconds.

We created the endpoint OpenAI-compatible on port 8000, reached from client machines over an SSH tunnel via port forward. PostgreSQL ran on the same VM as the SQL execution target. The agent's SQL executor and the inference engine shared a host, so round latency was dominated by inference rather than network hops. Engine telemetry came from vLLM's Prometheus `/metrics` endpoint, which we snapshotted before and after each sweep to derive TTFT and end-to-end latency from histogram sum/count pairs, plus prefix cache hit rate and KV cache utilization. We ran a single smoke-test request (96 completion tokens) which completed in 0.73 s end to end with a 0.16 s time-to-first-token. This we used to confirmed the endpoint was interactive-grade before benchmarking. The VM cost roughly $115/day while running, so we kept it stopped except during active sweeps. 

---

## Prefix Caching (Joannier)

*(Results section, pairs with the engine-level table)*

Prefix caching behaved almost exactly as hypothesizeds

| Engine config | Mean TTFT | Mean e2e latency |
|---|---|---|
| prefix_on + chunked_on | **26.8 ms** | 546 ms |
| prefix_on + chunked_off | **27.3 ms** | 556 ms |
| prefix_off + chunked_on | 51.7 ms | 592 ms |
| prefix_off + chunked_off | 51.9 ms | 595 ms |

Enabling prefix caching cut mean time-to-first-token by ~48%, from ~52 ms to ~27 ms. Our proposal predicted a 30-60% reduction, and this landed in the middle of that range. For a given task, every reflection round shares an identical leading context: the system prompt, the database schema, and the original broken query. RadixAttention serves that prefix from cache, so only the new content of each round (the previous attempt and its tool result) gets prefilled fresh.

Mean prompt length grew from 547 tokens in round 1 to 1,242 tokens by round 5 (2.3x, below our 3-5x prediction, Spider schemas and error messages are compact). Per-round latency tracked this growth, but more steeply without caching. Round 5 took 1.8x round-1 latency with prefix caching on (0.475 s to 0.849 s) versus 2.0x with it off (0.483 s to 0.954 s). The cached prefix is a larger share of the prompt exactly when the prompt is longest, which is where the savings matter most.

At the application level the benefit showed up but was not statistically significant. Median task latency was 3.05 s with prefix caching versus 3.25-3.38 s without (Mann-Whitney p=0.08-0.09), about 0.3 s per task, or ~30 s across the 100-task benchmark. Success rate, tokens consumed, and rounds-to-success were unchanged (all pairwise p > 0.37). That is the expected result: prefix caching changes how fast each round is served, not what the model generates. The engine-level numbers are direct measurements rather than noisy task outcomes, so we consider the 48% TTFT reduction the firmest quantitative finding of the our experiment.

---

## Chunked Prefill (Joannier)

*(Results section)*

Our proposal predicted chunked prefill would reduce tail latency under concurrent load by preventing long late-round prefills from blocking shorter requests (head-of-line blocking). But we did not obsere this during our experiment:

| Engine config | p50 | p75 | p95 | p99 |
|---|---|---|---|---|
| prefix_on + chunked_on | 3.05 s | 3.69 s | 5.06 s | 6.46 s |
| prefix_on + chunked_off | 3.07 s | 3.99 s | 4.99 s | 6.28 s |
| prefix_off + chunked_on | 3.25 s | 4.22 s | 5.14 s | 6.36 s |
| prefix_off + chunked_off | 3.38 s | 4.03 s | 5.33 s | 6.57 s |

Within each prefix-caching group, the chunked-on and chunked-off distributions differed by less than 0.2 s at p95/p99, and the direction was inconsistent. Mean TTFT was likewise indistinguishable (26.8 vs 27.3 ms with prefix on, 51.7 vs 51.9 ms with prefix off).

We attribute this to experimental scale rather than to the optimization itself. The hypothesis was untestable at concurrency 2, so a good follow up would be test this with higher concurrency. Chunked prefill helps by interleaving decode steps of running requests with slices of a long incoming prefill. That requires a queue deep enough for long and short requests to actually contend. With only two concurrent tasks, and per-round prompts that maxed out around 1,200-2,000 tokens (at or below our 2,048-token `max_num_batched_tokens` budget), a round-5 prefill almost never blocked a round-1 request from the other task. KV cache utilization stayed near zero throughout the runs, which confirms the engine was never under load.

Two conditions would make the predicted effect visible: higher concurrency (8-16 parallel tasks, where queueing is routine) and longer prompts that exceed the batched-token budget and force multi-chunk prefills. Both are within reach of this setup. The v5litepod-4's 44x KV-cache concurrency headroom means the engine itself would not be the bottleneck.

---

## Optimizations (Joannier and Ann)

*(Conceptual analysis of a fully deployed system. I drafted the systems half. Ann should add/edit the application-level items marked below.)*

Our measurements point to four optimization opportunities for a production deployment of a reflective agent, in decreasing order of confidence.

**1. Prefix caching is necessary but incompletely exploited (measured).** The 48% TTFT reduction came from caching the static per-task prefix. In a deployed multi-tenant system the cacheable surface is larger. The system prompt and tool descriptions are shared across all tasks, and schema definitions are shared across all tasks targeting the same database. A schema-aware router that sends tasks for the same database to the same engine replica would raise the cross-request cache hit rate well above what our single-task-stream benchmark achieved. The complementary lever is prompt layout: ordering context from most-shared to least-shared (system prompt, tool spec, schema, task, history) maximizes the reusable prefix. Our agent already did this by construction.

**2. Cross-round KV reuse beyond the prefix (conceptual).** Prefix caching only reuses the unchanged leading portion of the prompt. In reflection workloads, rounds n and n+1 share nearly the entire round-n prompt, because new content is appended rather than interleaved. Engine-side support for append-only conversations (keeping the full per-task KV state alive between rounds instead of re-matching the radix tree from scratch) would reduce round-n prefill to only the newly appended tokens. That would turn the 1.8x round-5 latency growth we measured into near-constant per-round latency. This is the single largest remaining systems win for the reflection pattern, since per-round prompt growth is intrinsic to it.

**3. Right-size the reflection budget by difficulty tier (measured, application-level: Ann).** Success by round was strongly tier-dependent. Easy tasks that succeeded did so in early rounds, while hard tasks rarely succeeded at any budget (15-24%). A deployed system should not spend a uniform 5-round budget. An early-exit policy (stop when the same error repeats or attempts stop changing materially) plus a tier-conditioned budget would cut the largest cost item: failed tasks that burned all 5 rounds and the maximum token count. Our verbosity result reinforces this. Compact prompts increased median tokens per task (4,302 vs 4,210) because they lowered success and pushed more tasks to budget exhaustion. Token-saving measures that reduce signal are counterproductive. Budget-saving measures that detect futility are not.

**4. Concurrency scaling and chunked prefill (conceptual).** At concurrency 2 the engine was heavily underutilized (44x KV-cache headroom) and chunked prefill had nothing to do. A real deployment would batch many agent tasks per replica. That raises throughput but re-introduces exactly the head-of-line risk chunked prefill addresses, so our null result should not be read as license to disable it at scale. The interesting deployment question is the throughput/latency trade-off curve as concurrency rises toward the KV limit, with chunked prefill expected to flatten the tail. Measuring that curve at concurrency 8-32 is the experiment our setup was one knob away from running.

One more point about where we think optimization effort should go. No engine configuration moved easy-tier success (See the 52.9% everywhere) or hard-tier success materially. Infrastructure optimizations in this setup bought latency and cost, not accuracy. Closing the hard tier requires a stronger model or richer tools (for example, loading row data so the agent can verify results), not a faster engine.
