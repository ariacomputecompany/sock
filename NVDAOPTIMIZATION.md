# NVIDIA / Blackwell Optimization Notes for TMH

Date started: 2026-07-25
Companion documents: `TMHSTORY.md`, `OPTIMIZATIONS.md`
Current implementation host: GMKtec EVO-X2, AMD Strix Halo / Radeon 8060S `gfx1151`, ROCm 7.2.4
Primary model used for current evidence: `Qwen/Qwen3-30B-A3B-GPTQ-Int4`

This document records what should and should not be assumed when moving TMH from
the current Strix Halo / ROCm implementation toward NVIDIA, especially Hopper and
Blackwell-class systems.

The short answer:

```text
TMH as a memory hierarchy should transfer.
TMH as a high-performance serving implementation must be retuned per backend.
```

## Current Ground Truth

The current TMH evidence has two separate scoreboards.

Low-pressure endpoint throughput is still negative in the current robust run:

```text
standard KV geomean completion tok/s: 36.5928
TMH geomean completion tok/s:         31.4599
Delta:                                -14.03%
```

Memory-pressure capacity is strongly positive and reproduced at two KV budgets:

```text
6.50 GiB KV budget:
  standard: 70,944 logical KV tokens
  TMH:      123,312 logical KV tokens
  lift:     about +73.8%

41.70 GiB KV budget:
  standard: 455,424 logical KV tokens
  TMH:      792,592 logical KV tokens
  lift:     about +74.0%
```

The right claim is therefore not that TMH is universally faster. The right claim
is that TMH is a capacity-first KV memory hierarchy that currently buys about
`1.74x` logical KV residency per GiB under the `25%` hot-budget policy.

## What Is Portable

The following TMH concepts are architecture-independent and should transfer to
NVIDIA with little conceptual change.

### KV Hierarchy

TMH's core abstraction is independent of ROCm:

```text
PINNED_RAW pages: anchor pages kept raw
HOT_RAW pages: recent pages kept raw
WARM_INT8_INT4 pages: older early-layer warm pages
WARM_INT8_INT8 pages: older late-layer warm pages
```

The principle is general:

```text
recent and structurally important KV should stay raw
older KV can trade precision/layout simplicity for residency
```

This is about the shape of transformer inference, not the shape of AMD hardware.

### Capacity Accounting

`TMHFullAttentionSpec` and its `physical_allocation_bytes(num_logical_blocks)`
model should transfer conceptually.

Standard KV asks:

```text
How many raw blocks fit in available memory?
```

TMH asks:

```text
How many logical blocks fit if the physical pool contains raw and warm storage?
```

That accounting depends on:

```text
block size
number of layers
number of KV heads
head dimension
raw dtype size
warm K/V representation
hot-budget percentage
small descriptor metadata
```

It does not inherently depend on ROCm. NVIDIA should show the same kind of
logical-capacity lift if the same storage policy and compression formats are
implemented.

### Hot-Budget Policy

The current policy uses:

```text
--tmh-hot-budget-pct 25
```

That means roughly: keep the anchor/recent pages raw and store older pages more
compactly.

The policy itself should be portable. The optimal percentage may differ by
hardware, model, workload, and target latency. NVIDIA needs its own hot-budget
ablation, but the knob remains meaningful.

### All-Raw Fast-Path Principle

This principle should transfer directly:

```text
If the active request set is all raw, TMH should behave like standard KV at the
attention boundary.
```

The low-pressure path should not pay mixed raw/warm overhead. This is true on
AMD, NVIDIA, Blackwell, and any future backend.

### Native Block-Table Boundary

The maintained native block-table idea should transfer:

```text
TMH internal table can keep diagnostic sentinels.
Native attention sees a backend-compatible block table.
```

On ROCm the important mismatch was:

```text
TMH internal sentinel: -1
native vLLM padding:   0
```

On NVIDIA, the exact backend contract may differ, but the boundary remains
important. Native attention kernels should receive metadata in the format they
already expect.

### Two-Scoreboard Evaluation

The evaluation method should transfer:

```text
Scoreboard 1: low-pressure throughput vs standard KV
Scoreboard 2: memory-pressure capacity and live saturation behavior
```

Any NVIDIA paper result should preserve this distinction. A low-pressure tok/s
loss and a memory-pressure capacity win are not contradictory; they are the
tradeoff TMH is designed to expose.

## What Is ROCm / Strix Halo Specific

The current implementation contains several optimizations that should not be
assumed to transfer as-is.

### ROCm Decode Partition Size

The current ROCm custom paged decode path found a useful partition size:

```text
_PARTITION_SIZE_ROCM = 512
```

Tested values:

```text
256: earlier baseline
512: best stable production compromise on gfx1151
768: safe but worse on the key long-context c4 probe
1024: unsafe, produced HIP illegal memory access
```

This is explicitly hardware/backend specific. It reflects ROCm paged-attention
geometry, Strix Halo behavior, and the current vLLM/SOCK kernel path. It should
not be copied blindly to CUDA or Blackwell.

### ROCm Custom Decode Gate

The current gate is shaped around ROCm behavior:

```python
use_rocm_custom_decode = (
    max_query_len == 1
    and num_seqs >= 2
    and max_seq_len >= 640
)
```

This was measured on `gfx1151`. NVIDIA may want a different gate, or no such
gate if FlashAttention/FlashInfer paths dominate.

### ROCm Kernel Stability Boundaries

Several failures were specific to ROCm or this Strix Halo stack:

```text
ROCm paged-attention illegal memory accesses
1024 decode partition instability
ROCm MoE WNA16 / routing crash surfaces
WSL friction before the bare-metal Ubuntu rebuild
custom PyTorch wheel requirements for gfx1151
```

These should be treated as implementation history, not universal TMH behavior.

### AITER / Triton / ROCm Backend Selection

The current path used Triton attention heavily because ROCm paged attention had
stability issues on this host. NVIDIA has a stronger CUDA attention ecosystem,
so backend choice should be re-opened from scratch.

Do not assume the ROCm conclusion transfers:

```text
ROCm: Triton was often the stable production-shaped fallback.
NVIDIA: FlashAttention, FlashInfer, CUDA graph paths, or TensorRT-LLM-style
kernels may be better baselines.
```

## What NVIDIA / Blackwell Needs

A proper NVIDIA port should be treated as a backend optimization project, not a
simple recompile.

### 1. Establish Standard KV Baselines

Before TMH claims anything on NVIDIA, collect strong standard-KV baselines:

```text
Hopper and/or Blackwell GPU
same model family if possible
standard vLLM
SOCK standard KV
FlashAttention / FlashInfer backend variants
compiled vs eager where appropriate
context windows: 8K, 16K, 32K, model max
concurrency sweeps near capacity frontier
```

The standard-KV baseline must be strong. A weak baseline makes TMH look better
for the wrong reason.

### 2. Port Capacity Accounting First

The first NVIDIA milestone should not be throughput. It should be startup
capacity parity with the ROCm allocator story:

```text
same available KV memory
standard logical KV tokens
TMH logical KV tokens
expected lift: around the same storage-model ratio, currently ~1.7x
```

If the capacity lift does not appear, the port is wrong at the allocator/spec
level before kernel work matters.

### 3. Preserve All-Raw Native Fast Paths

On NVIDIA, all-raw TMH should route into the best native CUDA attention path
available.

Candidates to evaluate:

```text
FlashAttention
FlashInfer paged attention
vLLM CUDA native paged attention
CUDA graph captured decode paths
TensorRT-LLM-style attention paths where available
```

The goal is:

```text
all-raw TMH ~= standard KV
```

The all-raw path should be treated as a correctness and performance invariant.

### 4. Implement Or Adapt Mixed Raw/Warm Attention

The mixed path is the true TMH-specific work. NVIDIA needs efficient handling of:

```text
raw page reads
warm int8/int4 page reads
warm scale handling
dequantization or fused dequant-attention
page-role metadata
prefix-cached page behavior
```

The key question is whether warm-page handling should be:

```text
separate dequant + native attention
fused warm-page attention kernel
FlashInfer-style custom paged backend
Triton kernel specialized for NVIDIA
a CUDA/CUTLASS-style custom kernel
```

This is where Blackwell-specific optimization likely matters most.

### 5. Tune Backend Shape Gates

ROCm's gate should be replaced by NVIDIA measurements.

NVIDIA should sweep:

```text
max_query_len: prefill vs decode
num_seqs: 1, 2, 4, 8, 16, 24+
max_seq_len: 512, 2K, 8K, 16K, 32K, model max
backend: FlashAttention, FlashInfer, Triton, native CUDA
CUDA graph mode: none, piecewise, full if applicable
```

The output should be a NVIDIA-specific dispatch table or policy, not a copy of
`max_seq_len >= 640` from ROCm.

### 6. Revisit Cache Write Kernels

On ROCm, direct use of native `reshape_and_cache_flash` against TMH raw views
failed because read-layout compatibility did not imply write-kernel ABI
compatibility.

NVIDIA needs to retest this boundary. It may be possible that CUDA native cache
write paths are easier to integrate, or it may require a TMH-owned translated
write kernel.

Do not assume either result.

### 7. Test CUDA Graph Compatibility

Blackwell performance may depend heavily on graph capture and stable metadata
addresses.

TMH metadata must be audited for:

```text
stable tensor shapes
stable block-table addresses
no host syncs in graph-captured regions
no dynamic allocations in hot decode
no per-step Python-side metadata rebuilds
```

The maintained native block-table optimization was a step in this direction, but
NVIDIA graph capture should be measured directly.

## Blackwell-Specific Expectations

Blackwell should not change the TMH memory thesis, but it may change the best
execution strategy.

Likely Blackwell advantages:

```text
stronger attention kernels
better FP8 / low-precision ecosystem
better graph-captured decode potential
higher memory bandwidth
more mature CUDA tooling
FlashAttention/FlashInfer availability
```

Likely Blackwell work items:

```text
fused warm-page dequant + attention
backend-specific tile sizes
graph-safe metadata path
FlashInfer/TensorRT-LLM integration
hot-budget retuning for larger memory and faster kernels
mixed-path correctness and quality validation
```

Possible outcome:

```text
TMH capacity lift transfers immediately.
TMH throughput tax may shrink on Blackwell if all-raw native handoff and mixed
warm-page kernels are better than the current ROCm path.
```

But this is a hypothesis, not proven yet.

## Paper Positioning

The paper should separate universal contribution from backend-specific
engineering.

Universal contribution:

```text
Transformer Memory Hierarchy as a KV-cache residency mechanism
page-role policy: pinned, hot, warm
logical KV capacity per GiB as a first-class serving metric
all-raw fast-path invariant
capacity-adjusted serving evaluation
```

Backend-specific engineering:

```text
ROCm gfx1151 implementation
ROCm custom decode partition 512
ROCm/Triton backend gates
Strix Halo build/runtime fixes
```

NVIDIA future-work / extension section:

```text
Port TMH allocator to CUDA backend.
Route all-raw TMH through the strongest CUDA native attention path.
Design or integrate fused warm-page attention for Blackwell.
Retune dispatch policy using Blackwell shape sweeps.
Evaluate live saturation at model-max context.
```

Avoid claiming:

```text
The ROCm partition-512 result is generally optimal.
The current gate is architecture-independent.
TMH is faster than standard KV in all settings.
Blackwell will automatically improve TMH without tuning.
```

Prefer claiming:

```text
The TMH memory hierarchy is architecture-general.
The current ROCm implementation proves the allocator/capacity mechanism.
High-performance realization requires backend-specific attention and cache-write
optimization.
```

## NVIDIA Benchmark Plan

A minimal NVIDIA validation plan should have four phases.

### Phase 1: Capacity Frontier

Run startup capacity probes:

```text
standard KV vs TMH
same available KV budget
contexts: 8K, 16K, 32K, model max
utilization: 0.35, 0.70, 0.90
metrics: logical KV tokens, max concurrency, available KV GiB
```

Expected if allocator transfer is correct:

```text
TMH token capacity lift near the storage-model ratio, currently about 1.7x
```

### Phase 2: Low-Pressure Throughput

Run the robust endpoint suite under low pressure:

```text
small/medium/long cases
concurrency 1, 2, 4
10 measured runs, 2 warmup
standard KV vs TMH
```

Goal:

```text
measure all-raw overhead on NVIDIA
```

### Phase 3: Live Saturation

Run near the memory frontier:

```text
model max context
standard at/above its admitted concurrency frontier
TMH at/above standard frontier and near its own frontier
metrics: success rate, queueing, p50/p90 latency, tokens/sec/GiB
```

Goal:

```text
show whether capacity headroom becomes a better user experience
```

### Phase 4: Backend Sweep

Sweep attention/cache backends:

```text
FlashAttention
FlashInfer
Triton
native CUDA paged attention
CUDA graph modes
custom warm-page kernels if present
```

Goal:

```text
find NVIDIA-specific dispatch policy
```

## Engineering Checklist For Porting

Before claiming NVIDIA support:

```text
1. TMH allocator creates correct logical capacity.
2. Standard KV baseline is strong and reproducible.
3. All-raw TMH path routes to native CUDA attention safely.
4. Native block-table handoff matches CUDA backend expectations.
5. Cache write path is verified, not assumed from read compatibility.
6. Mixed raw/warm path passes correctness tests.
7. Warm compressed pages preserve acceptable model quality.
8. Backend gates are measured on NVIDIA, not copied from ROCm.
9. CUDA graph behavior is explicitly tested.
10. Live saturation proves queueing/tail-latency benefit.
```

## Current Thesis

TMH is not Strix Halo specific as an idea. It is a general KV memory hierarchy.

The current SOCK implementation, however, contains ROCm/Strix-specific execution
choices that should be treated as one backend realization:

```text
ROCm custom decode partition 512
ROCm/Triton stability choices
Strix Halo-specific build/runtime fixes
shape gates measured on gfx1151
```

A NVIDIA or Blackwell implementation should reuse the memory hierarchy,
capacity accounting, page-role policy, and all-raw fast-path invariant, but it
should retune attention, cache-write, graph-capture, and mixed warm-page kernels
for CUDA.

The expected research story is therefore:

```text
TMH is architecture-general at the memory-system level.
TMH is architecture-specific at the high-performance kernel/backend level.
```

