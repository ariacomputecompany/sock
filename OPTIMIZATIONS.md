# TMH Optimization Record

Date: 2026-07-24
Host: GMKtec EVO-X2 bare-metal Ubuntu 24.04.4
GPU target: AMD Strix Halo / Radeon 8060S, `gfx1151`
Runtime: ROCm 7.2.4, HIP 7.2.53211, source-built PyTorch `2.11.0+gfx1151`, sock vLLM `0.0.0+sock.rocm724`
Model: `Qwen/Qwen3-30B-A3B-GPTQ-Int4`

This document records what changed to move physical TMH from a large negative
throughput gap toward parity on the GMK 30B endpoint benchmark. It is intended
to be enough context for a future continuation without relying on chat memory.

## Headline Result

The current best production-shaped TMH result is the maintained native block
table path on top of shape-adaptive all-raw decode:

```text
Same-day Ubuntu standard KV:         37.8412 geomean completion tok/s
TMH all-raw native prefill, Triton decode: 35.1083 geomean completion tok/s
TMH all-raw native prefill, always ROCm custom decode: 35.1797
TMH all-raw native prefill, gated ROCm custom decode:  35.4940
TMH gated decode, maintained native block table:        35.8999

Current gap vs same-day standard: -5.13%
```

The long-context concurrency-4 hot cell improved materially:

```text
long_context_summary_256 c4
standard KV:             70.1954 completion tok/s
TMH Triton decode:       40.8045 completion tok/s
TMH always custom decode:46.6415 completion tok/s
TMH gated custom decode: 50.5300 completion tok/s
TMH maintained native table: 50.5007 completion tok/s
```

The overall path from the original physical TMH result now looks like this:

```text
Old physical TMH:            28.49, -18.08% vs then-current standard
Pressure-adaptive TMH:       29.10, -16.35%
Raw fast path pre-Ubuntu:    29.7749, -14.41% vs 34.78
Ubuntu standard KV:          37.8412
All-raw native prefill TMH:  35.1083, -7.22%
Always custom decode TMH:    35.1797, -7.03%
Gated custom decode TMH:     35.4940, -6.20%
Maintained native table TMH: 35.8999, -5.13%
```

Relative to the adaptive `29.10` checkpoint, the current `35.8999` result is
`+23.37%`. Relative to the old physical `28.49` result, it is `+26.01%`.

## First-Principles Finding

The physical TMH layout was not the main remaining tax once pressure-adaptive
raw placement was in. The bottleneck was execution shape:

- Low-pressure requests were logically all raw, but still paid for the mixed
  raw/warm TMH attention kernel.
- The mixed kernel carries page-role checks, dequantization branches, packed
  int4 handling, and warm scale plumbing even when no compressed pages can be
  touched.
- Standard vLLM wins by specializing the hot path to the actual shape.
- TMH needed the same architectural property: a raw-only execution path for
  all-raw requests, and the mixed executor only when warm compressed pages are
  actually reachable.

The key abstraction is therefore not "make the TMH kernel faster everywhere."
It is "select the right attention backend for the current TMH page regime and
decode shape."

## Production Changes

### Bare-Metal ROCm Runtime

Windows/WSL was removed from the GMK, and the host was rebuilt as bare-metal
Ubuntu. That removed WSL friction and made ROCm validation deterministic enough
to continue endpoint benchmarking.

Runtime work completed:

- Installed ROCm 7.2.4 under `/opt/rocm-7.2.4`.
- Verified `rocminfo` and `rocm-smi` see the `gfx1151` target.
- Built PyTorch from source for `gfx1151` because public ROCm wheels were not
  reliable on Strix Halo.
- Rebuilt the PyTorch wheel with Gloo enabled so vLLM distributed startup works.
- Rebuilt vLLM ROCm extensions against the source-built torch wheel.
- Centralized runtime environment in `~/.gmk_env`.

This produced a stable same-day standard KV benchmark of `37.8412` geomean
completion tok/s.

### Pressure-Adaptive Raw Placement

The first meaningful production optimization was to stop compressing pages when
there was no pressure reason to do so.

Implemented behavior:

- TMH requests stay fully raw while the active request set fits in the raw page
  pool.
- Warm descriptors are promoted to raw physical slots under low pressure.
- The scheduler/runtime/kernel agree on the same all-raw threshold.
- Compression is used only when raw storage would be oversubscribed.

This improved old physical TMH from `28.49` to `29.10` geomean completion tok/s,
but it still left a `-16.35%` gap because all-raw requests were still executing
through the TMH mixed attention path.

### All-Raw Native Prefill Cutover

The next cutover removed the custom TMH raw attention kernel from the all-raw
prefill hot path.

Implemented behavior:

- The public facade remains `tmh_physical_attention`.
- All-raw TMH batches are detected before dispatch.
- All-raw prefill now hands the raw TMH cache to vLLM's native chunked prefill
  path.
- Mixed raw/warm batches still route to TMH's mixed executor.
- The raw cache exposes a native-shaped KV view compatible with standard vLLM
  expectations.
- A persistent identity scale tensor is kept on the TMH physical cache so the
  native attention call does not allocate scale tensors per request.

This is the large step that moved TMH from the `-14%` to `-7%` region:

```text
Ubuntu standard KV:              37.8412
TMH native all-raw prefill path: 35.1083
Gap:                             -7.22%
```

### Safe Native Decode Block Table Handoff

Earlier unsafe native ROCm handoff attempts wedged the engine. The production
version makes the native block-table boundary explicit and sanitizes TMH's
internal sentinel values before handing the table to vLLM/native paged attention.

Implemented behavior:

- `_tmh_native_decode_block_table` copies the active TMH request-row block table
  into a native per-sequence workspace.
- If a scheduler sequence-to-request-row map is present, rows are gathered with
  `torch.index_select`.
- The native view is clamped with `block_table.clamp_min_(0)`.

Reason:

- Standard vLLM block tables pad unused cells with block `0`.
- TMH request rows use `-1` internally to make missing descriptors visible.
- ROCm custom paged attention should never observe the TMH-only `-1` sentinel.

Focused synthetic checks showed the ROCm custom decode helper can run against
the TMH raw cache with the sanitized block table, and endpoint smokes no longer
wedged the engine.

### Shape-Adaptive Decode Backend

Always enabling ROCm custom paged decode was stable, but not globally best. It
helped the long-context concurrent decode cells and regressed several short or
single-stream cells. The production cutover is therefore shape-adaptive.

Current gate in `vllm/v1/attention/ops/tmh_triton_ops.py`:

```python
use_rocm_custom_decode = (
    max_query_len == 1
    and num_seqs >= 2
    and max_seq_len >= 640
)
```

If the gate is true, all-raw TMH decode uses ROCm custom paged decode through
`chunked_prefill_paged_decode`. If false, it forces the generic Triton paged
decode fallback.

Why this gate exists:

- `max_query_len == 1` identifies decode, not prefill.
- `num_seqs >= 2` avoids the single-stream cases where custom decode was not
  consistently better.
- `max_seq_len >= 640` targets the long-context regime where decode backend
  specialization pays off on this host.

This is a production policy, not a no-op. It changes the attention backend used
by all-raw TMH decode on long-context concurrent shapes, while preserving the
Triton fallback for shapes where it is faster.

### Maintained Native Block Table

The next successful pass removed more scaffolding from the all-raw native
attention call. The previous native handoff rebuilt a standard-style block table
inside `_tmh_native_decode_block_table` for every attention invocation:

- If the sequence-to-request-row map was absent, it copied
  `request_slot_by_row_page` into a native workspace and clamped `-1` padding to
  `0`.
- In practice `tmh_physical_attention` always created a `seq_rows` tensor before
  the native branch, so the helper usually took the gather path anyway.
- The gather path copied the request-row map, indexed the request slot table,
  and clamped the result.

That was architecturally backward. The standard vLLM block table is not
attention scratch; it is cache metadata. TMH now maintains a standard-style
native block table at descriptor-application time:

- `TMHPhysicalKVCache.native_block_table_by_seq` is zero-initialized.
- `_clear_request_rows` clears the TMH internal request row to `-1` and clears
  the native row to `0`.
- `_apply_descriptor` writes the physical slot into both
  `request_slot_by_row_page` and `native_block_table_by_seq`.
- `_tmh_native_decode_block_table` returns the maintained table directly when
  the sequence order is identity.
- Non-identity request-row maps gather from the maintained native table into a
  separate `native_block_table_gather` workspace, with no clamp.
- `tmh_physical_attention` no longer creates `torch.arange` for the native
  branch when no sequence-to-request-row map was provided.

This keeps TMH's internal `-1` sentinel for mixed kernels and diagnostics while
presenting native attention with a standard vLLM block table whose unused cells
are already `0`.

Result:

```text
Previous gated decode geomean: 35.4940
Maintained native table geomean: 35.8999
Delta vs previous gated: +1.14%
Gap vs same-day standard: -5.13%
```

The targeted long-context slice also improved versus the previous gated slice:

```text
long_context_summary_256 targeted slice
c1: 22.0323 -> 23.2895
c2: 33.5267 -> 35.7113
c4: 45.7216 -> 49.5003
```

The full-suite `long_context_summary_256 c4` result stayed essentially flat
against the prior gated full run (`50.5300 -> 50.5007`), but the geomean moved
because the same metadata-lifetime fix improved several other long/medium
concurrent shapes:

```text
long_cosmology_512 c2:      34.3576 -> 38.8443
long_cosmology_512 c4:      45.9218 -> 49.6267
medium_architecture_256 c1: 26.6897 -> 27.4398
medium_architecture_256 c4: 51.9031 -> 54.2022
```

## Benchmark Artifacts

Standard baseline:

```text
benchmarks/2026-07-24-gmk-qwen3-30b-native-ubuntu-standard-full/standard-suite.json
geomean: 37.8412
```

All-raw native prefill plus Triton decode:

```text
benchmarks/2026-07-24-gmk-qwen3-30b-native-ubuntu-tmh-gloo-smoke/tmh-suite-native-allraw-triton-decode.json
geomean: 35.1083
gap: -7.22%
```

All-raw native prefill plus always ROCm custom decode:

```text
benchmarks/2026-07-24-gmk-qwen3-30b-native-ubuntu-tmh-custom-decode/tmh-suite-custom-decode.json
geomean: 35.1797
gap: -7.03%
```

All-raw native prefill plus shape-adaptive decode:

```text
benchmarks/2026-07-24-gmk-qwen3-30b-native-ubuntu-tmh-gated-decode/long-context-summary.json
benchmarks/2026-07-24-gmk-qwen3-30b-native-ubuntu-tmh-gated-decode/tmh-suite-gated-decode.json
geomean: 35.4940
gap: -6.20%
```

Shape-adaptive decode plus maintained native block table:

```text
benchmarks/2026-07-24-gmk-qwen3-30b-native-ubuntu-tmh-native-table/long-context-summary.json
benchmarks/2026-07-24-gmk-qwen3-30b-native-ubuntu-tmh-native-table/tmh-suite-native-table.json
geomean: 35.8999
gap: -5.13%
```

Benchmark command shape:

```bash
cd ~/work/sock
source ~/.gmk_env
vllm/.venv/bin/python scripts/sock_endpoint_bench_suite.py \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-30B-A3B-GPTQ-Int4 \
  --runs 2 \
  --warmup-runs 1 \
  --concurrency-levels 1,2,4 \
  --timeout-s 1200
```

Serve command shape:

```bash
./target/debug/sock serve Qwen/Qwen3-30B-A3B-GPTQ-Int4 \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name Qwen/Qwen3-30B-A3B-GPTQ-Int4 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.35 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 4 \
  --enforce-eager \
  --kv-layout tmh \
  --tmh-hot-budget-pct 25
```

## Verification

Focused tests:

```text
cd ~/work/sock/vllm
source ~/.gmk_env
./.venv/bin/python -m pytest -q \
  tests/v1/core/test_tmh_physical.py \
  tests/v1/core/test_tmh_triton_ops.py

Result: 13 passed
```

Endpoint validation:

- Standard KV full benchmark passed.
- TMH all-raw native prefill plus Triton decode full benchmark passed.
- TMH always-custom decode tiny smoke passed.
- TMH always-custom decode long-context slice passed.
- TMH always-custom decode full benchmark passed.
- TMH gated decode long-context slice passed.
- TMH gated decode full benchmark passed.
- TMH maintained-native-table long-context slice passed.
- TMH maintained-native-table full benchmark passed.

The previous unsafe native handoff failure mode was an engine wedge during
endpoint smoke. The sanitized/gated native decode path survived health checks,
smoke, long-context slice, and the full endpoint suite.

## Remaining Work

The current `-5.13%` is a large improvement, but still not parity. The next
largest target remains long-context concurrency:

```text
long_context_summary_256 c4:
standard: 70.1954
maintained-table TMH: 50.5007
remaining gap: about -28.0%
```

The likely next moonshot is deeper decode specialization:

- Tune or replace the `max_seq_len >= 640` gate with measured shape buckets.
- Investigate why standard c4 long-context reaches `70.1954` while TMH now
  reaches about `50.5`, even though TMH is all-raw in this regime.
- Profile the remaining native handoff overhead under c4 now that block-table
  copy/clamp is off the identity path.
- Compare generic Triton decode, ROCm custom decode, and any AITER decode
  variants on the same raw TMH cache.
- Consider a TMH-owned all-raw decode kernel only if native handoff overhead or
  layout impedance remains the limiter.

The implementation rule for the next pass should remain the same: do not tune
policy around a slow executor. Preserve the all-raw native fast path and move
only the shapes with evidence onto a faster decode backend.

## 2026-07-24 Partition-512 ROCm Decode Breakthrough

After the maintained native block table result, several prefix-cache lifetime
experiments were falsified:

- `tmh-prefix-retain`: long-context c4 `43.3248`, worse than maintained table.
- `tmh-prefix-retain-hash-only`: long-context c4 `41.7972`, also worse.
- `tmh-no-prefix`: long-context c4 `35.6198`, proving TMH still benefits from prefix caching.
- `tmh-identity-rows`: full sweep was not run; warmed c4 repeated at `50.2733`, essentially neutral.
- `tmh-rocm-workspace-reuse`: c4 `45.4609`, worse than maintained table.

The successful moonshot was changing the ROCm custom paged-attention decode
partition from `256` to `512` in
`vllm/v1/attention/ops/chunked_prefill_paged_decode.py`. This reduces the
number of partitions/reductions for the Strix Halo long-ish decode shapes used
by the endpoint suite.

Production-shaped full-suite result:

```text
Same-day standard KV full geomean:       37.8412
TMH maintained native table geomean:     35.8999
TMH ROCm decode partition 512 geomean:   37.9962

Delta vs maintained TMH: +5.84%
Delta vs same-day standard: +0.41%
```

The win is concurrency-heavy rather than universal. Single-stream cells regress,
but concurrency-2 and concurrency-4 cells improve enough to move the full suite
positive:

```text
tiny_fact_64 c4:           55.5541 -> 82.5883
short_codegen_128 c4:      53.7511 -> 74.7029
medium_architecture_256 c4:54.2022 -> 66.5721
long_cosmology_512 c4:     49.6267 -> 62.0212
long_context_summary_256 c4:50.5007 -> 51.6829
extended_generation_768 c4:58.0344 -> 58.4819
```

This changes the current thesis: after the native-table work, the remaining
negative gap was not primarily prefix-cache lifetime. The lamp was the
partition/reduction geometry inside ROCm paged decode. On this `gfx1151` host,
`512` is a better production compromise for the benchmark mix than the previous
`256`.

## 2026-07-24 Partition-1024 Falsification

A direct follow-up tested `_PARTITION_SIZE_ROCM = 1024` after the `512` full-suite
win. The hypothesis was that, for the endpoint suite sub-2048 token decode
regime, larger partitions might reduce the reduction count further and move the
concurrency-heavy cells closer to the +20% target.

Result: the c4 long-context probe failed with HTTP 500 after EngineCore hit a
HIP illegal memory access during the first request. The reported stack surfaced
later in `_tmh_raw_reshape_and_cache_kernel`, which is consistent with an
asynchronous device fault from the preceding ROCm paged-attention geometry.

Conclusion: `1024` is outside the safe/supported geometry for this ROCm custom
paged-attention path on `gfx1151`, at least under the current TMH all-raw native
handoff. Keep `512` as the current production constant and continue searching
below or around it, not above it.

## 2026-07-24 C1 Custom Decode Gate Falsification

After `512` won as the ROCm decode partition, a follow-up tested relaxing the
TMH native decode gate from `num_seqs >= 2` to `num_seqs >= 1`. The hypothesis
was that the new partition geometry might make ROCm custom paged decode useful
for single-stream decode too.

A concurrency-1-only suite looked promising: c1 geomean improved from `22.8834`
to `25.1543` versus the partition-512 full-suite c1 cells. But the production
full suite falsified the change:

```text
TMH partition-512 full geomean: 37.9962
TMH c1-custom full geomean:     32.7044
Delta vs partition-512:        -13.93%
Delta vs standard KV:          -13.57%
```

Conclusion: the old `num_seqs >= 2` gate should stay. The c1-only result was a
run-shape artifact, while the full suite showed broad c2/c4 regressions. The
next search should preserve the proven concurrency gate and look for a more
selective shape policy or a deeper kernel-level improvement.


## 2026-07-24 Partition-768 Falsification

After `1024` proved unsafe and `512` proved production-positive, a follow-up
probe tested `_PARTITION_SIZE_ROCM = 768`. The hypothesis was that a middle
partition could keep the safe geometry of `512` while reducing partition/reduce
overhead for long-context concurrency-4 decode.

Result: the c4 long-context probe was stable but did not beat `512`:

```text
TMH partition-512 long_context_summary_256 c4 median: 54.8502 completion tok/s
TMH partition-768 long_context_summary_256 c4 median: 54.2934 completion tok/s
Delta vs partition-512: -1.02%
```

Conclusion: `768` is safe but worse than `512` on the most relevant early-gate
shape. The useful abstraction is no longer "larger partition is better"; it is
"Strix Halo needs a shape-aware ROCm decode policy, and `512` is the current
best stable global constant." Keep production code at `512` and spend the next
passes on selective gating or deeper kernel launch/cache behavior rather than
pushing the partition upward.

## 2026-07-24 Translated Native Cache-Update Falsification

A deeper pass attacked the long-context prefill tax instead of another decode
partition. The hypothesis was that all-raw TMH cache updates still pay a custom
per-token/per-head Triton reshape kernel, while standard KV uses vLLM native
`reshape_and_cache_flash`. The experiment translated TMH request/page metadata
into a physical slot mapping and attempted to call native `reshape_and_cache_flash`
against the TMH raw key/value views for all-raw prefill only.

Result: focused TMH GPU tests failed before endpoint benchmarking. The native
cache update corrupted adjacent TMH physical storage and broke decode correctness:

```text
tests/v1/core/test_tmh_triton_ops.py::test_tmh_triton_attention_uses_raw_fast_path_for_all_raw_batches FAILED
tests/v1/core/test_tmh_triton_ops.py::test_tmh_native_raw_decode_reads_physical_raw_pages FAILED
```

Conclusion: the raw TMH views are compatible with native attention reads, but
not with this native cache-write op boundary. The useful distinction is
read-layout compatibility versus write-kernel ABI compatibility. Reverted the
experiment and kept the production all-raw reshape kernel in place. A future
version could still win here, but it needs a TMH-owned translated write kernel
or a verified native write ABI, not a direct `reshape_and_cache_flash` call.

## 2026-07-24 Single-Sequence Raw Reshape Falsification

A narrow kernel specialization removed the binary-search sequence lookup from
`_tmh_raw_reshape_and_cache_kernel` when `num_seqs == 1`. The hypothesis was
that the remaining single-stream long-context gap was partly a repeated per-head
metadata lookup in the all-raw cache-write path.

The first c1 long-context probe looked promising:

```text
TMH partition-512 long_context_summary_256 c1 median: 18.3756 completion tok/s
TMH single-seq reshape probe c1 median:              21.7416 completion tok/s
```

But the production-shaped full suite falsified the change:

```text
TMH partition-512 full geomean:       37.9962
TMH single-seq reshape full geomean:  33.5783
Delta vs partition-512:             -11.63%
```

The regression was broadest in the short concurrent cells that previously made
partition-512 positive:

```text
tiny_fact_64 c4:      82.5883 -> 51.7367
short_codegen_128 c4: 74.7029 -> 51.8042
short_codegen_128 c2: 43.2201 -> 31.1047
```

Conclusion: this was another run-shape artifact. The c1-only probe improved in
isolation, but the full suite showed the specialization disturbed the broader
kernel/autotune/runtime balance enough to erase the main concurrency win. Revert
and keep the proven raw reshape kernel unchanged. Future single-stream work
should be validated with a full suite immediately after a narrow probe.

## 2026-07-24 Prefix Hot-Raw Sharing Falsification On Partition-512

After the partition-512 decode win, prefix sharing was retested from first
principles. Standard KV still dominates `long_context_summary_256 c4`, and TMH
stores prefix-cached `HOT_RAW` pages as request overlays rather than canonical
shared raw pages. The hypothesis was that canonical/shared hot raw prefix pages
would recover some of standard KV long-context prefix-sharing advantage.

The experiment changed `_storage_kind_for_role` in both the scheduler policy and
runtime descriptor application so `PINNED_RAW` and `HOT_RAW` both used canonical
storage even when `prefix_cached=True`.

Focused tests passed, but the c4 long-context benchmark falsified the change:

```text
TMH partition-512 focused long_context_summary_256 c4 median: 54.8502 completion tok/s
TMH partition-512 full-suite long_context_summary_256 c4:      51.6829 completion tok/s
TMH prefix hot-raw sharing c4 median:                         47.5727 completion tok/s
```

Conclusion: prefix-cached hot raw request overlays are not accidental overhead;
on this workload they are part of the working performance envelope. Forcing
canonical hot-raw sharing likely increases contention/lifetime coupling or
reduces useful request-local placement. Revert and keep the overlay policy. The
long-context standard gap remains real, but this is not the route to closing it.

## 2026-07-24 Max Batched Tokens 2048 Falsification

A runtime-shape pass tested `--max-num-batched-tokens 2048` with the proven
partition-512 code. The hypothesis was that TMH might convert its memory-layout
headroom into better long-context batching, especially for the c4 prefill/decode
mix.

The narrow gate looked positive:

```text
TMH partition-512 focused long_context_summary_256 c4 median: 54.8502 completion tok/s
TMH max-num-batched-tokens=2048 c4 median:           55.9429 completion tok/s
```

But the full suite falsified the serve-config change:

```text
Standard KV full geomean:             37.8412
TMH partition-512 full geomean:        37.9962
TMH batched-2048 full geomean:         33.6334
Delta vs partition-512:              -11.48%
```

The larger scheduler window again damaged the short and medium concurrent cells
that carry the partition-512 win:

```text
tiny_fact_64 c4:           82.5883 -> 53.2313
short_codegen_128 c4:      74.7029 -> 52.1535
medium_architecture_256 c4:66.5721 -> 50.3301
```

Conclusion: the endpoint mix prefers the original `--max-num-batched-tokens 1024`.
The c4 long-context gate alone is insufficient because scheduler batching is a
global latency/throughput policy. Keep `1024` in the production-shaped command
until a shape-aware scheduler policy exists.

## 2026-07-24 Partition-512 Refresh Rebaseline Warning

After several falsified moonshots, the current production-shaped partition-512
code was rerun without code changes to check whether the original short c4 wins
were reproducible. The code still had `_PARTITION_SIZE_ROCM = 512` and the
shape-adaptive decode gate intact.

The refresh did not reproduce the original partition-512 full-suite result:

```text
Original same-day standard KV full geomean:       37.8412
Original TMH partition-512 full geomean:          37.9962
Refreshed TMH partition-512 full geomean:         33.4570
Older TMH maintained native table full geomean:   35.8999
```

The largest difference was the short concurrent cells that had carried the
original partition-512 win:

```text
tiny_fact_64 c4:      82.5883 original -> 51.4558 refreshed
short_codegen_128 c4: 74.7029 original -> 51.9847 refreshed
medium_architecture c4:66.5721 original -> 54.5527 refreshed
```

Conclusion: the partition-512 code path remains the checked-in production
candidate, but the old `+0.41%` headline should be treated as unstable until it
is paired with a same-period standard full refresh. The next pass should first
establish paired standard/TMH baselines under identical thermal/runtime state,
then optimize against that live baseline. Do not chase +20 from the stale
partition-512 artifact alone.

## 2026-07-24 Paired Standard Baseline Crash

After the partition-512 refresh failed to reproduce the original short-c4 spike,
the next first-principles pass attempted a paired full baseline: fresh standard
KV full suite followed immediately by fresh TMH full suite under the same host
state.

The standard full suite did not complete. It failed with HTTP 500 after EngineCore
hit a ROCm illegal memory access in the Qwen3-MoE router/gate path, not in TMH
attention:

```text
standard-paired-refresh-full-suite: FAILED
EngineCore stack: qwen3_moe.py -> fused_moe runner -> gate(hidden_states)
ROCm op: torch.ops._rocm_C.wvSplitK via rocm_unquantized_gemm
Error: hip/CUDA illegal memory access
```

Conclusion: the live benchmark state is currently unstable below the TMH layer.
This invalidates immediate standard-vs-TMH claims from the paired pass. Before
any +20 optimization claim, the next step is to prove baseline health again:
run a narrow standard smoke, then either a serialized `AMD_SERIALIZE_KERNEL=3`
repro for `wvSplitK` or a clean reboot if the GPU remains poisoned. Treat all
post-crash throughput comparisons as suspect until the standard endpoint can
complete the suite again.

## 2026-07-24 Standard Health Reprobe Narrows Crash Surface

After the paired standard full-suite crash, the standard endpoint was restarted
and reprobed case-by-case.

Result:

```text
standard tiny_fact_64 c4:      passed, median 53.5054 completion tok/s
standard short_codegen_128 c4: passed, median 55.7930 completion tok/s
standard medium/long c4 slice: failed with HTTP 500
```

The repeated failure is below TMH and below attention. The second crash surfaced
in Qwen3-MoE WNA16 fused experts:

```text
qwen3_moe.py -> fused_moe runner -> moe_wna16.py -> fused_experts
Failure site surfaced at fused_moe.py cache allocation after fused expert work
Error: hip/CUDA illegal memory access
```

Conclusion: the live baseline blocker is now narrower than the full endpoint
suite. Standard KV can serve short concurrent cells, but medium/long concurrent
MoE shapes can poison EngineCore. The next useful pass is not a TMH attention
optimization; it is a core runtime stability/backend pass around ROCm MoE WNA16
or the shared/gate GEMM path. +20 is unreachable as a credible claim until this
standard-path crash is either eliminated or isolated behind a safer backend.

## 2026-07-24 Triton-Unfused MoE Backend Falsification

The first core-runtime escape hatch was to route standard KV away from the
default fused MoE expert backend with `--moe-backend triton_unfused`. The
hypothesis was that the medium/long c4 crash was inside the fused expert kernel,
so a simpler Triton expert path might stabilize the baseline even if it was
slower.

The endpoint came up cleanly, but the exact narrowed failure slice still died:

```text
standard --moe-backend triton_unfused health: passed
medium_architecture_256 / long_cosmology_512 / long_context_summary_256 c4: HTTP 500
```

The stack still points through the stable MoE custom extension's top-k path:

```text
_moe_C_stable_libtorch.abi3.so -> topk_softmax
Error: hip/CUDA illegal memory access
```

The log also showed inference-time Triton JIT for `_fwd_kernel`,
`fused_moe_kernel_gptq_awq`, and `_gemm_kernel`, so this flag does not remove
all of the relevant MoE custom/routing surface for Qwen3 GPTQ WNA16 on gfx1151.

Conclusion: backend-level expert selection is not enough. The crash survives
when the visible backend is `triton_unfused`, which makes the routing/top-k or
shared WNA16 MoE path the more likely fault line. The next pass should disable
fused grouped top-k directly with `VLLM_USE_FUSED_MOE_GROUPED_TOPK=0`; if that
still fails, use `AMD_SERIALIZE_KERNEL=3` to force a more truthful crash site.

## 2026-07-24 Fused Grouped Top-K Disable Falsification

The next MoE-router pass disabled the fused grouped top-k path directly with
`VLLM_USE_FUSED_MOE_GROUPED_TOPK=0` while keeping the standard KV serve shape.
The hypothesis was that the grouped router/top-k custom path was corrupting GPU
state for medium/long c4 requests.

The endpoint loaded and passed health, but the same narrowed failure slice still
failed:

```text
standard VLLM_USE_FUSED_MOE_GROUPED_TOPK=0 health: passed
medium_architecture_256 / long_cosmology_512 / long_context_summary_256 c4: HTTP 500
```

The visible stack moved to the next attention output allocation:

```text
qwen3_moe.py -> self_attn -> attention.py torch.empty(output_shape, ...)
Error: hip/CUDA illegal memory access
```

Because ROCm reports illegal memory accesses asynchronously, this stack is
probably not the root kernel. It does prove that disabling fused grouped top-k is
not sufficient to stabilize the standard medium/long c4 path.

Conclusion: the fault is broader than the grouped top-k switch. It may still be
an earlier MoE/router/WNA16 kernel, but the current non-serialized run only shows
where the poisoned stream was observed. The next pass should rerun a minimal
reproducer with `AMD_SERIALIZE_KERNEL=3` so the crash is attributed closer to the
launch that causes it.

