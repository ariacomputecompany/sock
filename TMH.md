# TMH Status And Optimization Handoff

This document is the current single-purpose record for TMH: what is implemented,
what is proven, what is still wrong, and what optimization work has already been
tried. It intentionally avoids broader sock benchmarking detail except where it
directly explains TMH behavior.

## Current Production State

TMH is implemented as a first-class KV layout inside the vendored vLLM runtime.
It is selected through sock/vLLM cache configuration with `--kv-layout tmh`, which
derives the internal `tmh_kv_policy=physical` path. This is not a placeholder
and not an accounting-only shim.

The production-shaped TMH path currently includes:

- Canonical cache configuration for `kv_layout="tmh"` and physical TMH policy.
- Scheduler/runtime propagation of TMH policy and hot-page budget.
- Physical TMH cache allocation with separate raw pinned/hot pages and warm
  compressed pages.
- Per-request physical page descriptors for scheduler-side role, storage kind,
  quantization mode, prefix-cache awareness, and logical-to-physical slot
  assignment. Runtime kernels consume only the physical slot table; page role is
  derived deterministically from sequence geometry and the hot-page budget.
- Physical cache materialization and reclamation through the real vLLM worker
  path. Raw TMH pages are exposed as a zero-copy ROCm-native KV view so eligible
  raw-only batches can use the standard ROCm paged-attention backend.
- TMH cache update kernels that write raw pages and warm compressed pages.
- TMH attention kernels that read raw, warm int8/int4, and warm int8/int8 pages.
- Prefix-cache-aware descriptor handling.
- Startup warmup for the physical TMH kernels, eliminating request-time TMH JIT
  compilation in the measured 30B runs.
- CUDA physical TMH functional validation through the FlashInfer backend.
- ROCm physical TMH functional validation through the ROCm attention backend.

The important safety point: the committed runtime on `main` is sane. The
validated sock runtime works, the physical TMH path starts and serves, and the
large uncommitted optimization experiments from the latest investigation were
removed from the working tree instead of being left as pseudo-production code.
At the time this document was written, the local tree was reset to clean before
adding this file.

## Core Thesis

TMH is designed to reduce inference-time KV memory pressure by storing different
regions of the KV cache at different fidelities:

- The pinned anchor page remains raw.
- Recent hot pages remain raw.
- Older warm pages are compressed.
- Early layers can use int8 K plus int4 V.
- Late layers use int8 K plus int8 V.

The memory-pressure thesis has held up in accounting and functional tests. The
remaining issue is not whether TMH can represent the intended layout. It can. The
issue is whether the physical TMH attention kernel can execute that layout fast
enough to beat or match standard paged KV on real hardware.

## Current Performance Problem

The unresolved production blocker is the GMK/AMD Qwen3-30B physical TMH
throughput delta.

The best committed GMK/AMD 30B physical TMH result is still materially behind
standard KV:

| Runtime | Suite wall s | Ready s | Geomean completion tok/s | Delta vs standard | TMH JIT warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard KV baseline | 765.69 | 73 | 36.70 | baseline | n/a |
| Physical TMH, first physical kernel | 1075.15 | 65 | 26.44 | -27.96% | 0 |
| Physical TMH, optimized kernel | 945.04 | 75 | 29.76 | -18.92% | 0 |
| Physical TMH, page-descriptor kernel | 935.04 | 76 | 29.98 | -18.33% | 0 |
| Physical TMH, scoped warmup rerun | 993.69 | 58 | 28.49 | -18.08% vs same-day standard | 0 |

The headline bad number remains:

`Physical TMH scoped-warmup rerun geomean delta vs same-day standard: -18.08%`

That is too large to accept. It means the physical TMH layout is functional, but
the physical attention path is not yet performance-competitive on this AMD 30B
profile.

Benchmark profile for this number:

- Model: `Qwen/Qwen3-30B-A3B-GPTQ-Int4`
- Hardware: GMK AMD Strix Halo / Radeon 8060S through WSL ROCm
- Endpoint: OpenAI-compatible `/v1/completions`
- Serve path: sock CLI into vendored vLLM
- `max_model_len=2048`
- `max_num_seqs=4`
- `max_num_batched_tokens=1024`
- `gpu_memory_utilization=0.35`
- `enforce_eager=true`
- Suite: 6 prompt classes, concurrency 1/2/4, 1 warmup batch, 2 measured batches
- TMH: `--kv-layout tmh --tmh-hot-budget-pct 25`

The raw benchmark artifacts are:

- `benchmarks/2026-07-19-gmk-qwen3-30b-physical-tmh/`
- `benchmarks/2026-07-19-gmk-qwen3-30b-physical-tmh-kernel-opt/`
- `benchmarks/2026-07-19-gmk-qwen3-30b-physical-tmh-page-desc-opt/`
- `benchmarks/2026-07-23-gmk-qwen3-30b-tmh-native-rerun/`

## Current Optimization Pass

The latest source pass targets self-inflicted overhead identified by comparing
TMH against the ROCm/vLLM fast path and AMD's current ROCm guidance. The
official ROCm tuning guidance says MHA workloads should use the optimized ROCm
attention backend where possible and notes that backend-specific KV layout can
materially affect decode throughput. TMH therefore should not replace the
backend-native attention path unless compressed pages are actually present.

Implemented changes:

- Raw TMH pages now have a zero-copy `[2, raw_pages, block, heads, head]`
  ROCm-native KV view alongside the existing raw key/value views.
- ROCm attention now routes TMH batches through standard
  `chunked_prefill_paged_decode` when the batch has only pinned/hot raw pages.
- The TMH Triton cache-update and attention kernels no longer load per-page role
  descriptors from global memory. They derive raw/warm role from `seq_len`,
  `block_size`, and `tmh_hot_budget_pct`, matching the scheduler policy.
- Dead GPU request descriptor tables for block id, role, and storage kind were
  removed. The kernels retain only the live physical slot table.
- Slot allocation no longer zero-fills whole raw/warm pages before use; live
  tokens are overwritten by the cache writer and attention masks exclude
  unwritten page tail values.

Verification for this pass:

- `./vllm/.venv/bin/python -m pytest -q vllm/tests/v1/core/test_tmh_physical.py vllm/tests/v1/core/test_tmh_triton_ops.py`
- Result: `9 passed`

Benchmark status: this pass has now been endpoint-benchmarked against the
GMK/AMD Qwen3-30B suite. The scoped warmup fixed startup reliability: TMH reached
`/health` in 58s and the direct warmup no longer wedges in synthetic MoE decode.
The throughput problem did not move enough: same-day standard geomean was
`34.78` completion tok/s, physical TMH geomean was `28.49` completion tok/s, for
`-18.08%` geomean completion throughput and `+22.13%` geomean wall-clock
latency. This confirms startup coupling was a real production bug, but it was
not the root cause of the physical attention throughput gap.

## What Has Worked

### Physical Bring-Up

Physical TMH is live end-to-end. It can start, allocate physical KV storage,
materialize descriptors, warm kernels, serve completions, and release physical
slots across request lifetimes.

This matters because the implementation is not merely reporting theoretical
memory reductions. It is wired into the real inference runtime.

### Scheduler Accounting Fix

The CUDA accounting regression on RTX 4090 was fixed. Before that fix, TMH
accounting work was placed on the scheduler hot path and caused a false
throughput regression even without the physical layout being active.

After the accounting fix:

| Host | Standard geomean tok/s | TMH geomean tok/s | Geomean delta |
| --- | ---: | ---: | ---: |
| RTX 4090 after fix | 107.48 | 107.48 | +0.00% |
| GB10 after fix | 28.24 | 28.63 | +1.37% |

This proved that sock/TMH accounting can be production-safe when it is kept out
of the scheduler hot path.

### First AMD Physical Kernel Optimization

The first physical TMH kernel was very slow: `-27.96%` geomean vs standard.

The optimized physical kernel improved geomean throughput by `+12.56%` versus
the first physical kernel and reduced suite wall clock from `1075.15s` to
`945.04s`.

This pass included:

- Splitting all-raw, all-warm, and mixed tile handling.
- Aligning TMH attention tiles with the 16-token physical page.
- Reusing the GPU request-row map across layers instead of rebuilding it per
  attention call.

This was real progress, but it only reduced the gap to `-18.92%`.

### Page-Descriptor Optimization

The page-descriptor pass used the invariant that each 16-token tile maps to one
physical TMH page. The kernel loads role/slot metadata once per page-aligned tile
instead of classifying every token lane.

This helped, but only a little:

- `+0.73%` geomean over the previous optimized physical kernel.
- Suite wall clock improved from `945.04s` to `935.04s`.
- Final delta remained `-18.33%` versus standard.

This ruled in descriptor overhead as a factor, but ruled it out as the whole
problem.

### CUDA Tile-Shape Tuning

On RTX 4090 Qwen3-8B, CUDA tile-shape tuning produced a small net win:

- `+0.85%` mean completion tok/s versus the prior physical TMH CUDA slice.
- Kept in production.

This was useful but not transformational.

### Segmented Decode, Gated

Segmented decode was implemented and tested. It increased sampled GPU
utilization in the diagnostic slice, but regressed the 1k-context endpoint path:

- `-4.53%` at the 1k-context RTX 4090 slice.

The correct production decision was to keep segmented decode implemented but
gate it to longer contexts where segment/reduce overhead has a chance to
amortize:

`max_seq_len >= 1025`

## What Did Not Work

### Packed-V Split Accumulator

The packed-V split accumulator was correct but slower overall. It was reverted.

Production decision: do not keep it.

### Hot Recent Floor Diagnostic

Hypothesis: the `-18.33%` AMD delta might be caused by compressing warm pages too
early. With `tmh_hot_budget_pct=25`, short and medium prompts can enter the warm
compressed path before memory pressure justifies the extra decode overhead.

Diagnostic attempted:

- Add `tmh_hot_min_pages=64`.
- Keep the first 64 trailing non-anchor pages raw.
- Propagate the knob through cache config, CLI, physical pool planning, runtime
  policy, and tests.
- Run the same GMK/AMD 30B endpoint suite against the existing standard
  baseline.

Result:

| Metric | Value |
| --- | ---: |
| Previous page-descriptor TMH delta vs standard | -18.33% |
| Hot-floor TMH geomean completion tok/s | 30.33 |
| Hot-floor TMH delta vs standard | -17.37% |
| Recouped delta | +0.96 percentage points |

This did not solve the problem. It showed that early warm-page compression is
not the dominant cause of the regression.

Production decision: do not merge this as a performance fix. It may still be a
reasonable future policy knob, but it is not the fix for the `-18%` gap.

### Native Raw Layout Diagnostic

Hypothesis: the physical TMH raw pages are stored in a layout that forces the
TMH kernel to do extra addressing work. If raw pages are stored in the same
native paged KV layout used by ROCm standard paged attention, raw-page reads may
get faster and enable later fast paths.

Diagnostic attempted:

- Change physical TMH raw key layout to native vLLM paged format:
  `[num_blocks, num_kv_heads, head_size // x, block_size, x]`.
- Change physical TMH raw value layout to native vLLM paged format:
  `[num_blocks, num_kv_heads, head_size_v, block_size]`.
- Keep a fallback for tiny test shapes where `head_size` is smaller than the
  native key packing factor.
- Update TMH cache update and TMH attention raw reads for the native layout.
- Run focused ROCm/GPU tests.

Result:

- Focused tests passed: `15 passed`.
- Server started cleanly after removing the unsafe native bypass.
- A focused `tiny_fact_64` diagnostic still showed roughly the same throughput:
  `25.14 / 28.87 / 40.92 tok/s` at concurrency 1/2/4.

This did not materially improve the throughput gap by itself.

Production decision: native raw layout alone is not enough. It may still be a
useful prerequisite for a later backend-native fast path, but it was not merged
as a standalone optimization in this pause.

### Backend-Native ROCm Attention Bypass

Hypothesis: if active TMH pages are all raw, TMH should bypass the custom TMH
Triton attention kernel and call the existing ROCm standard paged-attention
implementation with a remapped physical-slot block table.

This is architecturally attractive because it avoids paying TMH role/dequant
overhead when the active window is entirely raw.

Diagnostic attempted:

- Add raw-page native layout.
- Add an all-raw fast path from `tmh_backend_paged_attention` into
  `chunked_prefill_paged_decode`.
- Pass ROCm key/value tensors and scale metadata through the TMH backend.
- Restrict the fast path to raw windows.
- Then restrict it further to decode-only after warmup stalled.

Result:

- The first version stalled during physical warmup.
- The decode-only version also stalled during physical warmup.
- The bypass was removed rather than hidden behind a flag.

Production decision: do not keep this bypass. It needs a separate investigation
before it can be considered correct. The likely problem is metadata and/or cache
update interaction during physical warmup, not the high-level idea itself.

## Current Best Diagnosis

The `-18.33%` AMD gap is not primarily caused by:

- CLI wiring.
- sock runtime bring-up.
- Kernel JIT during measured inference.
- Scheduler accounting.
- Descriptor lookup alone.
- Early warm-page compression alone.
- Raw storage layout alone.

The best current diagnosis is:

The physical TMH attention kernel itself is still too expensive relative to
standard paged attention. Even when active pages are mostly raw, the TMH kernel
still carries role/slot indirection, branch structure, and warm-page support in
the hot loop. The standard ROCm path is highly specialized for regular paged KV,
while TMH is currently a more general layout-aware kernel.

The next optimization should therefore focus on the physical attention hot loop,
not policy knobs.

## Next Optimization Direction

The next pass should be a production-safe raw-window specialization of the TMH
Triton kernel, not a direct backend-native bypass.

The safe shape is:

- Keep TMH descriptors.
- Keep TMH physical allocation.
- Keep standard TMH warm compressed path.
- Add a separate raw-only TMH attention kernel or a compile-time-specialized
  path where warm-page branch/dequant code is not present.
- Only dispatch to that raw-only path when the active page window is provably
  pinned/hot raw.
- Do not call ROCm native paged attention until the metadata and warmup contract
  is understood independently.

The raw-only specialization should remove from the hot loop:

- Warm role checks.
- Warm K/V scale loads.
- int8/int4 dequant branches.
- packed int4 value unpacking.
- mixed tile logic.

The first validation target should be `tiny_fact_64`, because it is short,
stable, and exposes raw-window overhead quickly.

A useful acceptance ladder:

1. Focused GPU tests pass.
2. Server starts and physical warmup completes.
3. `tiny_fact_64` c1/c2/c4 improves materially versus page-descriptor TMH.
4. Full GMK/AMD 30B six-case suite recovers a meaningful portion of `-18.33%`.
5. Only then update `BENCH.md` and commit.

## What To Avoid Next

Avoid these paths unless there is new evidence:

- Do not merge policy-only fixes as the answer to the throughput gap.
- Do not hide the backend-native bypass behind a flag and call it production.
- Do not broaden segmented decode to short contexts.
- Do not reintroduce the packed-V split accumulator.
- Do not treat memory-pressure wins as sufficient if endpoint throughput still
  regresses by double digits.
- Do not leave experimental local patches dirty while documenting production
  status.

## Current Answer To The Sanity Question

Yes: the committed core runtime is sane and correct.

More precisely:

- sock itself is working well.
- TMH physical runtime is real and functionally wired.
- The accounting and scheduler path are production-safe after the earlier fix.
- The currently committed physical TMH kernel starts, warms, serves, and avoids
  request-time TMH JIT in the measured AMD 30B suite.
- The latest unsafe optimization experiments were not kept in production code.

What is not yet production-good is TMH physical throughput on AMD 30B. The
remaining work is performance engineering in the physical attention kernel.

