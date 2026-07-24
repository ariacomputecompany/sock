# TMH Status And Optimization Handoff

This document is the current single-purpose record for TMH: what is implemented,
what is proven, what is still wrong, and what optimization work has already been
tried. It intentionally avoids broader sock benchmarking detail except where it
directly explains TMH behavior.

## Current GMK Checkpoint - 2026-07-24

The latest production-shaped GMK/Ubuntu result supersedes the older `-16.35%`
and `-7.22%` resume points below:

| Runtime | Geomean completion tok/s | Delta vs same-day Ubuntu standard |
| --- | ---: | ---: |
| Standard KV, Ubuntu full suite | 37.8412 | baseline |
| TMH all-raw native prefill + Triton decode | 35.1083 | -7.22% |
| TMH all-raw native prefill + always ROCm custom decode | 35.1797 | -7.03% |
| TMH all-raw native prefill + gated ROCm custom decode | 35.4940 | -6.20% |

The current committed production thesis is now shape-adaptive all-raw execution:
all-raw prefill uses the native vLLM path; all-raw decode normally uses generic
Triton paged decode; long-context concurrent decode uses ROCm custom paged decode
after sanitizing the native block-table handoff. Mixed raw/warm requests still
use the TMH mixed attention kernel.

The main remaining hot cell is still long-context concurrency:

```text
long_context_summary_256 c4:
standard:       70.1954 completion tok/s
TMH Triton:     40.8045
TMH gated:      50.5300
current gap:   about -28.0%
```

Detailed optimization history and benchmark commands are in `OPTIMIZATIONS.md`.

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
  path. Raw TMH pages retain a zero-copy native-shaped KV view internally.
  All-raw prefill now uses the native vLLM path; all-raw decode uses a
  shape-adaptive native handoff with a sanitized standard-style block table.
  Mixed raw/warm batches remain on the TMH-owned backend path.
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
| Physical TMH, adaptive raw placement | 976.15 | 58 | 29.10 | -16.35% vs same-day standard | 0 |

The headline bad number is improved but not fixed:

`Physical TMH adaptive-raw geomean delta vs same-day standard: -16.35%`

That is too large to accept. It means the physical TMH layout is functional, but
the physical attention path is not yet performance-competitive on this AMD 30B
profile. The latest adaptive raw-placement pass recouped only `+2.12%` over the
old physical TMH run and reduced geomean wall overhead from `+22.13%` to
`+19.55%`.

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

- Raw TMH pages have a zero-copy `[2, raw_pages, block, heads, head]`
  native-shaped KV view alongside the existing raw key/value views.
- Physical TMH now uses pressure-adaptive placement: when the configured raw
  pool can hold every active request at the current page count, scheduler warm
  descriptors are promoted to real raw slots instead of storing those pages in
  compressed warm storage.
- The TMH Triton cache-update and attention kernels use the same raw-only
  threshold, so low-pressure requests write/read all pages as raw and only enter
  compressed warm storage when the raw pool would be oversubscribed.
- Dead GPU request descriptor tables for block id, role, and storage kind remain
  removed. The kernels retain only the live physical slot table.
- The attempted ROCm-native paged-attention handoff for all-raw TMH batches was
  removed from production. Endpoint smoke tests with both upper-bound and exact
  active sequence lengths wedged the engine after `_tmh_unified_attention_kernel`
  JIT. Keeping that handoff would be unsafe.

Verification for this pass:

- `./vllm/.venv/bin/python -m pytest -q vllm/tests/v1/core/test_tmh_physical.py vllm/tests/v1/core/test_tmh_triton_ops.py`
- Result: `9 passed`

Benchmark status: this pass has now been endpoint-benchmarked against the
GMK/AMD Qwen3-30B suite. The scoped warmup fixed startup reliability, but not
throughput. The adaptive raw-placement pass produced `29.10` geomean completion
tok/s versus the same-day standard `34.78`, for `-16.35%` geomean completion
throughput and `+19.55%` geomean wall-clock latency. The prior physical TMH run
was `28.49` geomean completion tok/s, so adaptive placement helped by `+2.12%`
versus old TMH. This is real but not close to enough.

Raw artifacts:

- Standard and prior physical TMH: `benchmarks/2026-07-23-gmk-qwen3-30b-tmh-native-rerun/`
- Adaptive raw-placement TMH: `benchmarks/2026-07-23-gmk-qwen3-30b-tmh-adaptive-custom/`

Conclusion from this pass: the `-18%` gap is dominated by the custom TMH
attention kernel remaining on the hot path, not by early warm-page compression
alone. The next viable optimization is a TMH-owned raw fast attention kernel or
a properly integrated backend contract that can reuse standard ROCm paged
attention without substituting an invalid block table.

## Raw Fast-Path Cutover

The physical TMH attention entrypoint is now split by execution regime instead
of routing every request through one mixed-layout kernel. The production facade
is `tmh_physical_attention`; it dispatches all-raw batches to dedicated raw-only
Triton kernels and keeps the mixed raw/warm kernel only for batches that can
actually touch compressed pages.

Implemented in this pass:

- `_tmh_raw_reshape_and_cache_kernel` writes all-raw batches without warm scale,
  quantization, or packed-value code.
- `_tmh_raw_attention_kernel` reads raw physical slots only; warm branches,
  dequantization, packed int4 unpacking, and page-role checks are not present in
  that kernel.
- `_tmh_mixed_attention_kernel` remains the compressed-page executor instead of
  the universal hot path.
- Physical TMH warmup now covers first-page raw prompt lengths plus the existing
  multi-page shape for both single-request and max-request launches.

Verification so far:

- Focused TMH tests: `10 passed`.
- Endpoint start/smoke on GMK Qwen3-30B TMH profile completed without wedging.
- Focused `tiny_fact_64` one-run smoke after warmup ladder:
  - c1: `27.59` completion tok/s, `2.32s` wall.
  - c2: `30.35` completion tok/s, `4.22s` wall.
  - c4: `47.14` completion tok/s, `5.43s` wall.
- A second smoke on the same live server reached c4 `49.67` completion tok/s.

Caveat: the first endpoint request still emitted one `_tmh_raw_attention_kernel`
JIT warning, even after direct warmup covered the relevant token counts. No
additional raw-kernel JIT appeared on the second smoke. This points at a remaining
real-model tensor-layout specialization gap in direct warmup, not repeated raw
executor compilation. Do not claim full production victory or full-suite delta
recovery until the six-case benchmark is rerun and the first-request raw JIT is
understood or accepted as a startup tradeoff.

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

## 2026-07-24 Ubuntu/GFX1151 Native All-Raw Update

The post-Ubuntu ROCm stack changed the denominator and clarified the real safe
fast path.

Runtime status:

- Bare-metal Ubuntu + ROCm 7.2.4 + source-built torch 2.11.0+gfx1151 is the
  active GMK stack.
- Torch had to be rebuilt with Gloo enabled for vLLM engine startup.
- vLLM native ROCm extensions were rebuilt against that torch wheel.
- Focused TMH tests pass: `13 passed`.

Execution-method update:

- All-raw TMH prefill now receives live K/V through the ROCm attention backend
  boundary and uses vLLM's native context prefill path.
- All-raw TMH decode uses the generic Triton paged decode fallback.
- The old TMH custom raw attention kernel is no longer used for all-raw endpoint
  batches.
- ROCm custom C++ paged decode must not be used for TMH physical raw cache on
  this stack: serialized endpoint debugging traced the illegal-address failure
  to `paged_attention_custom_launcher_navi`.
- Mixed raw+warm batches still use the TMH mixed kernel, because native raw-only
  attention cannot represent compressed warm pages.

Current full-suite numbers:

- Standard KV, current Ubuntu full suite: `37.8412` geomean completion tok/s.
- TMH native-allraw/Triton-decode full suite: `35.1083` geomean completion
  tok/s.
- Current apples-to-apples gap: `-7.22%`.
- Versus earlier same-day standard baseline `34.78`: `+0.94%`.
- Versus previous raw-fastpath TMH `29.7749`: `+17.91%`.

Per-case cliff:

- `long_context_summary_256` remains the dominant blocker:
  - c1: `20.76` TMH vs `26.85` standard (`-22.7%`)
  - c2: `28.00` TMH vs `36.67` standard (`-23.6%`)
  - c4: `40.80` TMH vs `70.20` standard (`-41.9%`)

Artifacts:

- `benchmarks/2026-07-24-gmk-qwen3-30b-native-ubuntu-tmh-gloo-smoke/tmh-suite-native-allraw-triton-decode.json`
- `benchmarks/2026-07-24-gmk-qwen3-30b-native-ubuntu-standard-full/standard-suite.json`

Next performance thesis:

The gap is no longer a general all-raw short-context tax. Short and medium
cases are competitive or positive in several cells. The main remaining problem
is long-context summary under concurrency, where the generic Triton decode
fallback and/or mixed warm-page path cannot match standard KV's specialized
ROCm paged attention. The next moonshot should focus on a TMH-safe long-context
decode specialization rather than more policy tuning.
