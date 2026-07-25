# TMH Story

Date started: 2026-07-25
Host context: GMKtec EVO-X2, bare-metal Ubuntu 24.04.4, AMD Strix Halo / Radeon 8060S `gfx1151`, ROCm 7.2.4
Primary model: `Qwen/Qwen3-30B-A3B-GPTQ-Int4`
Companion record: `OPTIMIZATIONS.md`

This document is the plain-language history and technical map of TMH in SOCK. It
explains what TMH is, why it exists, how it relates to KV cache and attention,
what we optimized, what the numbers currently say, and what still needs to be
proved.

`OPTIMIZATIONS.md` is the lab notebook. This file is the story of the idea.

## Short Version

TMH means Transformer Memory Hierarchy.

The core idea is simple: the KV cache is the model's working memory during
inference, and standard KV stores that working memory mostly uniformly in raw
GPU memory. TMH makes the KV cache hierarchical. Recent or structurally
important pages stay raw. Older pages can be stored in a more compact warm form.

That creates a tradeoff:

```text
Low-pressure throughput, current robust result:
  TMH vs standard KV: -14.03%

Memory-pressure capacity, current 90% utilization result:
  TMH vs standard KV: +74.03% logical KV residency
```

So TMH is not best described as "faster per token" yet. The current accurate
claim is:

```text
TMH buys about 1.74x more logical KV capacity per GiB under the current
25% hot-budget policy, while still paying a throughput tax in low-pressure
all-raw endpoint serving.
```

That is meaningful because long-context inference often becomes memory-bound
before it becomes compute-bound.

## What KV Cache Is

During generation, every new token attends to previous tokens. The model does
not want to recompute the key and value tensors for all prior tokens on every
step, so it caches them.

Conceptually:

```text
prompt tokens -> compute K/V once -> store in GPU memory
new token     -> read old K/V -> append new K/V
next token    -> read larger K/V -> append again
```

That stored history is the KV cache.

The cache grows with:

```text
context length
x active concurrent requests
x number of layers
x number of KV heads
x head dimension
x bytes per value
```

For short prompts and low concurrency, the KV cache is usually not the main
problem. For long-context chat, coding agents, RAG over large documents, or
multi-tenant serving, KV memory can become the limiting resource.

## Standard KV

Standard vLLM KV is straightforward: every cached page is stored in the raw KV
format expected by the attention kernels.

The upside:

```text
simple layout
fast attention reads
few special cases
excellent low-pressure throughput
```

The downside:

```text
every old token costs as much as a recent token
long contexts consume huge GPU memory
concurrency collapses when resident KV fills the pool
```

Standard KV is the invariant floor for TMH. If there is no memory pressure, TMH
should try to behave as much like standard KV as possible.

## What TMH Changes

TMH changes the physical memory model of KV cache.

Instead of treating all pages equally, TMH assigns page roles:

```text
PINNED_RAW: always-important anchor pages, stored raw
HOT_RAW: recent pages, stored raw
WARM_INT8_INT4: older early-layer warm pages, compressed more aggressively
WARM_INT8_INT8: older late-layer warm pages, compressed less aggressively
```

In the current production-shaped configuration:

```text
--kv-layout tmh
--tmh-hot-budget-pct 25
```

The hot budget controls how much of the request remains raw and recent. The
first page is also anchored raw. The older, non-hot region becomes warm storage.

A useful mental picture:

```text
standard KV:
  [raw][raw][raw][raw][raw][raw][raw][raw]

TMH:
  [pinned raw][warm compressed][warm compressed][warm compressed][hot raw][hot raw]
```

The goal is not to make every attention read cheaper. The goal is to make the
same physical memory hold more logical context.

## Why The Capacity Lift Is Real

The capacity lift comes from the allocator, not from benchmark decoration.

The relevant spec is `TMHFullAttentionSpec`. Standard KV effectively asks:

```text
How many raw KV blocks fit in available memory?
```

TMH asks:

```text
How many logical KV blocks fit if some physical pages are raw and some are warm?
```

The key accounting is represented by:

```text
physical_allocation_bytes(num_logical_blocks)
```

That function budgets:

```text
raw physical pages
+ warm compressed physical pages
+ descriptor metadata
```

Then vLLM's normal startup capacity calculation reports how many logical KV
tokens can fit.

At `--gpu-memory-utilization 0.90`, vLLM reported the same available KV budget
for both layouts:

```text
available KV cache memory: 41.70 GiB
```

Standard KV admitted:

```text
455,424 logical KV tokens
```

TMH admitted:

```text
792,592 logical KV tokens
```

That is a `+74.03%` logical token-capacity lift.

The same ratio reproduced at multiple context lengths:

```text
8K context:  standard 55.59x, TMH 96.75x, +74.04%
16K context: standard 27.80x, TMH 48.38x, +74.03%
32K context: standard 13.90x, TMH 24.19x, +74.03%
```

It also reproduced at a much smaller `6.50 GiB` KV budget:

```text
8K context:  standard 8.66x, TMH 15.05x, +73.79%
16K context: standard 4.33x, TMH 7.53x, +73.90%
```

This tells us the memory model is scaling linearly and predictably. TMH really
is buying about `1.74x` logical KV residency per GiB under the current policy.

## Is TMH A Specific Attention Kernel?

Not exactly. This is important.

TMH is not one magic attention kernel. It is a memory hierarchy plus a dispatch
strategy around attention.

The public facade is:

```text
tmh_physical_attention
```

That attention path decides what to do based on the current page regime and
request shape.

There are three important regimes:

```text
1. all-raw prefill
2. all-raw decode
3. mixed raw/warm attention
```

The fastest path depends on which regime the request is in.

## All-Raw Prefill

A key discovery was that low-pressure TMH requests may be logically all raw.
When that happens, running the full mixed TMH attention path is wasteful.

Early TMH did this:

```text
all pages are raw
but still run through mixed raw/warm TMH machinery
```

That paid for:

```text
page-role checks
dequantization branches
packed int4 handling
warm scale plumbing
extra metadata handoff
```

even when no compressed page could be touched.

The all-raw native prefill cutover changed the execution rule:

```text
all-raw TMH prefill -> native vLLM chunked prefill path
mixed raw/warm      -> TMH mixed executor
```

This was one of the big moves from the old large negative gap toward parity in
the early optimization series.

## All-Raw Decode

Decode is different from prefill because each step usually has `max_query_len ==
1`. For all-raw decode, TMH can often use standard-style paged attention, but it
needs a safe block-table handoff.

The current production-shaped gate for ROCm custom decode is:

```python
use_rocm_custom_decode = (
    max_query_len == 1
    and num_seqs >= 2
    and max_seq_len >= 640
)
```

Meaning:

```text
only during decode
only when at least two sequences are active
only once context is long enough to benefit
```

If the gate is true, all-raw TMH decode uses the ROCm custom paged decode path.
If false, it uses the Triton fallback.

This is not a cosmetic setting. It changes the attention backend for the hot
all-raw decode path.

## Mixed Raw/Warm Attention

When memory pressure is high enough, some pages become warm and compressed. At
that point native standard attention cannot simply read every page as raw KV.

The mixed TMH attention path has to understand:

```text
which pages are raw
which pages are warm
how warm K/V is quantized
how to dequantize or interpret warm pages
how to combine warm history with hot raw recent context
```

That path is necessarily more complex than standard KV. It is where the memory
hierarchy becomes real at inference time.

The long-term objective is not to force every request through the mixed path. It
is to use the mixed path only when the memory savings are worth it, and keep the
all-raw path as close to standard KV as possible.

## Block Tables And Why They Mattered

Paged attention uses block tables to map logical sequence pages to physical KV
slots.

TMH originally had an internal table that used `-1` to make missing descriptors
visible. Standard vLLM/native paged attention expects unused block-table entries
to be padded with `0`.

That mismatch matters:

```text
TMH internal diagnostic sentinel: -1
native vLLM padding convention:   0
```

Letting native attention see `-1` is unsafe.

The safe native decode handoff added a sanitized native table. Then the
maintained native block table optimization went further: instead of rebuilding
and clamping a native view during attention, TMH now maintains a native-shaped
block table when descriptors are applied.

Conceptually:

```text
TMH internal request table:
  keeps -1 sentinels for diagnostics and mixed kernels

TMH native block table:
  keeps standard 0-padded rows for native attention handoff
```

This removed hot-path scratch work and made native decode handoff safer.

## ROCm Decode Partition Optimization

One of the strongest earlier throughput wins came from the ROCm custom paged
attention decode partition size.

The useful constant became:

```text
_PARTITION_SIZE_ROCM = 512
```

The tested path:

```text
256: previous baseline
512: best stable production compromise
768: safe but slightly worse on the key long-context c4 probe
1024: unsafe, caused HIP illegal memory access
```

The `512` partition reduced partition/reduction overhead for the Strix Halo
long-ish concurrent decode shapes in the endpoint suite.

At that point in the optimization timeline, the result looked excellent:

```text
same-day standard KV geomean:     37.8412
TMH partition-512 geomean:        37.9962
Delta vs same-day standard:       +0.41%
```

But that number should not be treated as the current global truth. Later robust
reruns with a larger sample and restored runtime state showed the all-raw TMH
path was still negative in the current endpoint contract.

## Important Falsifications

A lot of TMH progress came from failed ideas. The failures were useful because
they narrowed the real problem.

### Partition 1024

Hypothesis:

```text
larger ROCm decode partition might reduce reductions further
```

Result:

```text
HIP illegal memory access
```

Conclusion:

```text
1024 is outside the safe geometry for this path on gfx1151
```

### Partition 768

Hypothesis:

```text
middle ground between 512 and 1024 might be better
```

Result:

```text
safe but worse than 512 on the key long-context c4 probe
```

Conclusion:

```text
512 remains the best stable global constant so far
```

### C1 Custom Decode

Hypothesis:

```text
ROCm custom decode might now help single-stream decode too
```

Result:

```text
c1-only probe looked promising
full suite regressed badly
```

Conclusion:

```text
keep num_seqs >= 2 in the decode gate
```

### Native Cache Update

Hypothesis:

```text
TMH all-raw cache writes might be able to call native reshape_and_cache_flash
```

Result:

```text
corrupted adjacent TMH physical storage and broke tests
```

Conclusion:

```text
TMH raw views are compatible with native attention reads, but not with that
native write-kernel ABI
```

### Prefix Hot-Raw Sharing

Hypothesis:

```text
canonical shared hot raw prefix pages might recover standard KV's prefix
sharing advantage
```

Result:

```text
long-context c4 got worse
```

Conclusion:

```text
request-local hot raw overlays are currently part of the performance envelope
```

### Larger max-num-batched-tokens

Hypothesis:

```text
2048 batched tokens might let TMH use its memory headroom better
```

Result:

```text
narrow c4 probe improved, full suite regressed
```

Conclusion:

```text
1024 remains the production-shaped scheduler setting until policy becomes
shape-aware
```

## Current Honest Scoreboards

TMH needs two scoreboards.

### Low-Pressure Throughput Scoreboard

This asks:

```text
When memory is not the bottleneck, is TMH as fast as standard KV?
```

Current robust answer:

```text
standard KV geomean completion tok/s: 36.5928
TMH geomean completion tok/s:         31.4599
Delta:                                -14.03%
```

That is not where we want to be. It means the all-raw TMH path still has
meaningful overhead in the current robust endpoint suite.

### Runnable Upstream vLLM Scoreboard

This asks:

```text
How does SOCK standard KV compare to runnable upstream vanilla vLLM?
```

Current answer:

```text
SOCK standard-KV geomean:      36.5928
runnable upstream vLLM:        38.0710
Delta:                         -3.88%
```

This says the broader SOCK stack is close to runnable upstream, but not ahead in
the robust all-raw standard-KV endpoint contract.

### Memory-Pressure Capacity Scoreboard

This asks:

```text
With the same physical KV memory budget, how much logical context can the
engine admit?
```

Current 90% utilization answer:

```text
available KV cache memory: 41.70 GiB
standard logical KV tokens: 455,424
TMH logical KV tokens:      792,592
Delta:                      +74.03%
```

This is the strongest current TMH result.

## What This Means For Real Inference

TMH matters when the user experience is limited by resident KV memory.

Examples:

```text
long-context chat:
  more active 32K sessions fit on one GPU before queueing or eviction

agentic coding:
  more repo context, tool history, traces, and intermediate reasoning can stay
  resident without forcing a context reset

RAG over large documents:
  larger retrieved context sets can remain available instead of being truncated
  or aggressively chunked

multi-tenant serving:
  the same GPU can host more active long-context customers before hitting the KV
  wall

bursty traffic:
  standard KV may look faster at low occupancy, then queue hard when KV fills;
  TMH should degrade later because it has more residency headroom

large context product tiers:
  TMH can support larger or more simultaneous 32K/64K/128K-style sessions on the
  same hardware
```

The product framing should be precise:

```text
TMH is not yet faster per request in low-pressure serving.
TMH is more context and concurrency per GPU under KV memory pressure.
```

## Why The Throughput Tax Still Exists

The throughput tax likely comes from a few overlapping sources:

```text
all-raw TMH metadata still sitting near the hot path
request-row to sequence-row translation
cache write path differences
backend dispatch and block-table handoff
mixed-path generality leaking into raw-path execution
scheduler policy that is not yet shape-aware
```

The first-principles target is:

```text
When the active request set is all raw, TMH should be observationally identical
to standard KV at the attention and scheduler boundary.
```

Only when warm pages are reachable should TMH pay mixed-path costs.

## Code Map

Important code areas:

```text
vllm/config/cache.py
  exposes kv_layout, tmh_kv_policy, tmh_hot_budget_pct

vllm/model_executor/layers/attention/attention.py
  creates TMHFullAttentionSpec when kv_layout=tmh physical mode is active

vllm/v1/kv_cache_interface.py
  defines TMHFullAttentionSpec and physical_allocation_bytes

vllm/v1/core/kv_cache_utils.py
  converts available KV memory into admitted logical blocks/tokens

vllm/v1/core/tmh_policy.py
  assigns pinned/hot/warm roles and emits physical page descriptors

vllm/v1/tmh_physical.py
  owns physical TMH cache structures

vllm/v1/attention/ops/tmh_triton_ops.py
  contains tmh_physical_attention, native handoff helpers, and shape gates

vllm/v1/attention/ops/chunked_prefill_paged_decode.py
  contains the ROCm paged-decode partition behavior that was tuned to 512
```

## Timeline

### 1. Bare-metal ROCm foundation

The GMK machine was moved to bare-metal Ubuntu with ROCm 7.2.4. PyTorch was
built from source for `gfx1151`, and vLLM ROCm extensions were rebuilt against
that runtime. This made endpoint benchmarking stable enough to trust.

### 2. Pressure-adaptive raw placement

TMH stopped compressing when there was no memory-pressure reason to compress.
This made low-pressure TMH requests stay raw.

### 3. All-raw native prefill

All-raw prefill stopped paying the mixed TMH executor tax and moved to native
vLLM chunked prefill.

### 4. Safe native decode handoff

TMH added a safe block-table boundary so native attention would see standard
`0` padding rather than TMH's internal `-1` sentinel.

### 5. Shape-adaptive decode

ROCm custom paged decode became gated by decode shape:

```text
max_query_len == 1
num_seqs >= 2
max_seq_len >= 640
```

### 6. Maintained native block table

TMH stopped rebuilding/clamping native block tables during attention calls and
started maintaining a native-shaped table as descriptor metadata changed.

### 7. ROCm partition 512

The ROCm custom decode partition was tuned to `512`, which improved concurrent
decode in the early production-shaped suite.

### 8. Robust rebaseline

A larger 10-run, 2-warmup benchmark corrected the story: current TMH is still
`-14.03%` versus standard KV in low-pressure endpoint throughput.

### 9. Capacity frontier

The memory-pressure benchmark revealed the real positive TMH claim: about
`+74%` logical KV capacity at both small and large KV budgets.

## Next Proofs

The next benchmark should not be another low-pressure throughput suite. It
should be live saturation around the memory frontier.

For example:

```text
max_model_len: 32768
gpu_memory_utilization: 0.90
standard frontier: about c14
TMH frontier: about c24
```

Measure:

```text
successful requests
error rate
queueing behavior
p50 latency
p90 latency
completed tokens per second
completed tokens per GiB
quality of degradation past the standard KV frontier
```

This will answer the key product question:

```text
How much of TMH's +74% allocator headroom becomes a better inference experience
when the server is actually saturated?
```

## Current Thesis

TMH is a real memory-capacity mechanism with an unfinished execution-efficiency
story.

The allocator/capacity result is strong and reproducible:

```text
about 1.74x logical KV residency per GiB
```

The low-pressure throughput result is still negative:

```text
-14.03% versus standard KV in the current robust endpoint suite
```

The path forward is therefore not to claim TMH is generically faster. The path
is to make the all-raw path indistinguishable from standard KV, keep improving
shape-adaptive decode, and prove that the capacity headroom turns into lower
queueing, fewer failures, and more long-context users per GPU under real memory
pressure.

