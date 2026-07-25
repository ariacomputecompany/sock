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

## 2026-07-24 Serialized Minimal Reproducer Points At ROCm Paged Attention

After the MoE backend and grouped-top-k switches failed, the benchmark was
minimized to one request shape: standard KV, `medium_architecture_256`, c4,
`--runs 1`, no warmup. The server was launched with `AMD_SERIALIZE_KERNEL=3` to
try to force synchronous attribution.

The minimal case reproduced the crash:

```text
standard medium_architecture_256 c4, runs=1, warmup=0: HTTP 500
```

One important caveat: the local PyTorch build warned that `AMD_SERIALIZE_KERNEL=3`
was not accepted by its boolean env parser:

```text
Ignoring invalid value for boolean flag AMD_SERIALIZE_KERNEL: 3 valid values are 0 or 1
```

Even with that caveat, the failure stack moved to a concrete lower layer instead
of generic MoE/attention allocation fallout:

```text
_rocm_C.abi3.so -> paged_attention_custom_launcher_navi<..., 256, false>
_rocm_C.abi3.so -> paged_attention(...)
Error: hip/CUDA illegal memory access
```

Conclusion: the strongest current evidence says the standard medium/long c4
crash is in ROCm paged attention, not primarily in TMH storage, not primarily in
MoE backend selection, and not fixed by disabling grouped top-k. The next pass
should either rerun with the accepted boolean `AMD_SERIALIZE_KERNEL=1` or bypass
ROCm paged attention entirely with a Triton attention backend probe. If Triton
attention stabilizes the c4 slice, the +20 path becomes a paged-attention kernel
problem rather than a TMH policy problem.

## 2026-07-24 Triton Attention Stabilizes Standard Medium/Long C4

The next coordinate-system change bypassed ROCm paged attention entirely with
`--attention-backend TRITON_ATTN`, keeping the same standard KV serve shape.
This directly tested the serialized finding that `_rocm_C paged_attention` was
the live crash surface.

The minimal reproducer passed:

```text
standard TRITON_ATTN medium_architecture_256 c4, runs=1: 52.3247 completion tok/s
```

Then the full narrowed medium/long c4 slice passed with two measured runs and
one warmup:

```text
medium_architecture_256 c4:    51.2015 completion tok/s
long_cosmology_512 c4:         51.2788 completion tok/s
long_context_summary_256 c4:   74.3885 completion tok/s
```

This is the first same-session configuration that survives the exact slice that
repeatedly killed standard ROCm paged attention. It also preserves useful
throughput on the long-context summary case, where ROCm paged attention never
reached a clean paired result in the current runtime state.

Conclusion: the negative-throughput story was partly a false benchmark frame.
The immediate blocker was not TMH policy, and not MoE routing alone; standard
ROCm paged attention on gfx1151 is unstable for medium/long concurrent decode.
`TRITON_ATTN` is now the stable baseline route for continued +20 work. Next,
run a full standard `TRITON_ATTN` suite, then pair it against TMH under the same
attention backend if TMH accepts that configuration.

## 2026-07-24 Full Standard Triton-Attention Baseline

After `TRITON_ATTN` stabilized the narrowed c4 failure slice, the full endpoint
suite was rerun under the same standard KV serve shape with `--attention-backend
TRITON_ATTN`.

Result:

```text
standard TRITON_ATTN full geomean: 36.7329 completion tok/s
cells: 6 cases x c1/c2/c4 = 18
elapsed: 773.2601 s
```

Cell medians:

```text
tiny_fact_64 c1/c2/c4:              32.9116 / 34.6675 / 55.3841
short_codegen_128 c1/c2/c4:         25.7130 / 35.2789 / 56.9322
medium_architecture_256 c1/c2/c4:   25.2760 / 36.0764 / 53.6950
long_cosmology_512 c1/c2/c4:        24.7322 / 33.6570 / 50.6543
long_context_summary_256 c1/c2/c4:  23.7823 / 34.8894 / 65.8769
extended_generation_768 c1/c2/c4:   23.7214 / 33.2304 / 51.0662
```

Conclusion: this is the first complete same-session standard baseline after the
ROCm paged-attention crash surface was identified. It is not the fastest old
headline, but it is stable and therefore usable. The next benchmark claim should
be paired against this denominator, not against the stale 37.8412 standard
artifact or the unstable 37.9962 TMH partition-512 artifact.

## 2026-07-24 TMH Plus Triton Attention Pairing Falsification

TMH was paired against the stable standard `TRITON_ATTN` denominator with the
same serve shape plus `--kv-layout tmh --tmh-hot-budget-pct 25`. The goal was to
see whether TMH's storage policy still added value once the crash-prone ROCm
paged-attention backend was bypassed.

The endpoint loaded and the narrowed medium/long c4 slice completed, but it was
slower than the same-session standard Triton attention slice:

```text
standard TRITON_ATTN medium_architecture_256 c4:   51.2015 completion tok/s
TMH TRITON_ATTN medium_architecture_256 c4:        49.2510 completion tok/s
Delta:                                            -3.81%

standard TRITON_ATTN long_cosmology_512 c4:        51.2788 completion tok/s
TMH TRITON_ATTN long_cosmology_512 c4:             44.8669 completion tok/s
Delta:                                           -12.50%

standard TRITON_ATTN long_context_summary_256 c4:  74.3885 completion tok/s
TMH TRITON_ATTN long_context_summary_256 c4:       34.4980 completion tok/s
Delta:                                           -53.62%
```

Conclusion: TMH is not the +20 route under Triton attention as currently wired.
The long-context summary regression is decisive enough that a full TMH Triton
suite is not worth burning time on before changing the underlying policy. The
near-term optimization front should move to standard `TRITON_ATTN` itself:
Triton tensor-descriptor mode, warmup/JIT coverage, scheduler shape, and any
Triton attention flags that affect c4 throughput without reintroducing ROCm
paged-attention instability.

## 2026-07-24 Global Triton Tensor-Descriptor Falsification

Standard `TRITON_ATTN` was rerun with `VLLM_TRITON_ATTN_USE_TD=1`, forcing
Triton tensor descriptors on ROCm. The hypothesis was that gfx1151 might benefit
from the tensor-descriptor path even though the default auto policy only enables
it on XPU.

The c4 gate was stable and mixed, so a full suite was run. The full result
falsified TD as a global switch:

```text
standard TRITON_ATTN full geomean:      36.7329
standard TRITON_ATTN TD=1 full geomean: 36.3261
Delta:                                  -1.11%
```

The useful signal is shape-specific. TD hurt many c1/c2 short and medium cells,
but materially helped long-context summary at concurrency:

```text
long_context_summary_256 c2: 34.8894 -> 38.1454  (+9.33%)
long_context_summary_256 c4: 65.8769 -> 72.4141  (+9.92%)
```

Conclusion: forcing tensor descriptors globally is not a +20 route. The stronger
abstraction is shape-adaptive Triton attention: TD appears useful for the
large-context concurrent summary shape, but it should not be paid for on the
shorter/single-stream cells. The next code-level pass should inspect the Triton
attention call site and test an adaptive TD gate keyed on runtime sequence shape
rather than process-wide environment state.

## 2026-07-24 Adaptive Triton Tensor-Descriptor Gate Falsification

The global TD run revealed a tempting lamp: long-context summary c2/c4 improved
by roughly 9-10%, while the full suite regressed. A code-level opt-in experiment
therefore added `VLLM_TRITON_ATTN_ADAPTIVE_TD=1`, selecting TD only on ROCm when
`seq_lens.shape[0] >= 2` and `max_seq_len >= 768`. The intent was to keep the
large-context concurrent gain without paying TD overhead on short or single-stream
shapes.

The c4 gate looked directionally positive against the stable full-suite baseline:

```text
medium_architecture_256 c4:   53.6950 -> 54.2081  (+0.96%)
long_cosmology_512 c4:        50.6543 -> 51.8571  (+2.37%)
long_context_summary_256 c4:  65.8769 -> 68.4290  (+3.87%)
```

But the full suite falsified the adaptive gate:

```text
standard TRITON_ATTN full geomean:             36.7329
standard TRITON_ATTN adaptive-TD full geomean: 35.6213
Delta:                                         -3.03%
```

The losses were broad across c1/c2 cells, and the long-context c4 gain did not
hold in the full run:

```text
long_context_summary_256 c2: 34.8894 -> 38.0918  (+9.18%)
long_context_summary_256 c4: 65.8769 -> 64.0172  (-2.82%)
medium_architecture_256 c2:  36.0764 -> 32.9737  (-8.60%)
short_codegen_128 c2:        35.2789 -> 32.6029  (-7.59%)
```

Conclusion: a naive max-sequence/concurrency TD gate is not robust enough. The
code was reverted; only this benchmark record remains. TD may still contain a
shape-specific opportunity, but it needs a better signal than `max_seq_len >= 768`
and `num_seqs >= 2`, or a deeper kernel-level change that avoids the short/medium
regressions.

## 2026-07-24 Standard Triton Batched-2048 Narrow Falsification

After standard `TRITON_ATTN` became the stable denominator, the scheduler window
was retested with `--max-num-batched-tokens 2048`. This was a different question
from the earlier TMH batched-2048 falsification because the attention backend had
changed from unstable ROCm paged attention to stable Triton attention.

The narrowed medium/long c4 gate completed, but the signal was too small and too
mixed to justify a full-suite burn:

```text
standard TRITON_ATTN baseline medium_architecture_256 c4:   53.6950
standard TRITON_ATTN batched-2048 medium_architecture_256 c4:53.2999
Delta:                                                     -0.74%

standard TRITON_ATTN baseline long_cosmology_512 c4:        50.6543
standard TRITON_ATTN batched-2048 long_cosmology_512 c4:    51.6374
Delta:                                                     +1.94%

standard TRITON_ATTN baseline long_context_summary_256 c4:  65.8769
standard TRITON_ATTN batched-2048 long_context_summary c4:  68.4410
Delta:                                                     +3.89%
```

Conclusion: the larger batching window may help some long-context concurrent
cells under Triton attention, but it is nowhere near the +20 target and already
regresses the medium c4 cell. Given the earlier full-suite sensitivity to
scheduler-window changes, keep `--max-num-batched-tokens 1024` as the stable
serve shape until a stronger shape-aware scheduler policy exists.

## 2026-07-24 Standard Triton Generation-Config VLLM Falsification

The server log showed Qwen's `generation_config.json` overriding vLLM sampling
defaults with `temperature=0.6`, `top_k=20`, and `top_p=0.95` unless the server
is launched with `--generation-config vllm`. Since the benchmark requests set
`temperature=0.2` but do not set `top_k` or `top_p`, the hypothesis was that
removing model-default top-k/top-p sampling constraints might reduce sampler
work without changing the benchmark script.

The c4 gate looked positive enough to earn a full-suite run:

```text
medium_architecture_256 c4:   53.6950 -> 54.0170  (+0.60%)
long_cosmology_512 c4:        50.6543 -> 52.7777  (+4.19%)
long_context_summary_256 c4:  65.8769 -> 71.6499  (+8.76%)
```

The full suite falsified it as a global serve setting:

```text
standard TRITON_ATTN full geomean:                 36.7329
standard TRITON_ATTN --generation-config vllm:     36.4462
Delta:                                             -0.78%
```

The long c4 cells improved, but shorter c1/c2 cells gave the gain back:

```text
long_cosmology_512 c4: 50.6543 -> 52.5560  (+3.75%)
short_codegen_128 c1: 25.7130 -> 24.9749  (-2.87%)
medium_architecture c2:36.0764 -> 34.7814 (-3.59%)
tiny_fact_64 c1:      32.9116 -> 31.8761  (-3.15%)
```

Conclusion: `--generation-config vllm` is not a +20 path for the current suite.
It may help selected long concurrent cells, but the full endpoint mix prefers
the model generation defaults. Keep the stable baseline serve shape unchanged.

## 2026-07-24 Standard Triton No-Eager Falsification

After stabilizing the denominator on standard KV plus `TRITON_ATTN`, the next
high-leverage moonshot was to remove `--enforce-eager`. The hypothesis was that
Triton attention might make vLLM's compiled/CUDAGraph path viable on gfx1151,
reducing scheduler/model overhead broadly enough to move the full distribution.

The server did initialize successfully with `enforce_eager=False`:

```text
torch.compile cache range: (1, 1024)
torch.compile total:      33.92 s
CUDAGraph capture:        4 s, 0.45 GiB
Health readiness:         ready after 72 two-second polls
```

So this was not a stability failure. The narrowed latency-sensitive c4 gate ran
to completion, but it clearly regressed against the stable standard Triton
baseline:

```text
medium_architecture_256 c4:   53.6950 -> 47.6038  (-11.34%)
long_cosmology_512 c4:        50.6543 -> 47.2523  ( -6.72%)
long_context_summary_256 c4:  65.8769 -> 62.3257  ( -5.39%)
Gate geomean delta:                                ( -7.85%)
```

Conclusion: compiled/CUDAGraph execution is now viable enough to serve traffic,
but it is slower on the exact c4 cells where we need the most help. Do not run a
full suite for this shape and keep `--enforce-eager` in the stable standard
Triton baseline. The lamp here is useful: our current bottleneck is not simply
Python/eager overhead. The losses point back toward kernel shape, scheduler
batching policy, MoE routing/top-k cost, or benchmark-contract-level sampling and
stream semantics rather than generic graph capture.

## 2026-07-24 Standard Triton No-Access-Log Falsification

The no-eager pass showed generic graph capture was not the missing lever, so the
next contract-preserving frontend hypothesis was uvicorn access-log overhead.
The server was launched with the stable standard Triton shape plus
`--disable-uvicorn-access-log`:

```text
--kv-layout standard
--attention-backend TRITON_ATTN
--enforce-eager
--max-num-batched-tokens 1024
--max-num-seqs 4
--disable-uvicorn-access-log
```

The latency-sensitive c4 gate looked promising:

```text
medium_architecture_256 c4:   53.6950 -> 54.1200  (+0.79%)
long_cosmology_512 c4:        50.6543 -> 51.8451  (+2.35%)
long_context_summary_256 c4:  65.8769 -> 72.2009  (+9.60%)
Gate geomean delta:                                (+4.18%)
```

But the full suite rejected it as a global serve setting:

```text
standard TRITON_ATTN full geomean:                  36.7329
standard TRITON_ATTN no-access-log full geomean:    36.3679
Delta:                                              -0.99%
```

The full-suite positives were narrow and the c4 long-context summary gain did
not reproduce at the same magnitude:

```text
long_cosmology_512 c4:        50.6543 -> 52.0296  (+2.72%)
short_codegen_128 c4:         56.9322 -> 57.9707  (+1.82%)
long_context_summary_256 c4:  65.8769 -> 66.0674  (+0.29%)
```

The losses came from c1/c2 and some short concurrent cells:

```text
long_cosmology_512 c2:        33.6570 -> 31.7142  (-5.77%)
extended_generation_768 c2:   33.2304 -> 32.1579  (-3.23%)
tiny_fact_64 c2:              34.6675 -> 33.7066  (-2.77%)
tiny_fact_64 c4:              55.3841 -> 54.0542  (-2.40%)
```

Conclusion: uvicorn access logging is not the hidden +20 tax. The gate result was
mostly run-to-run shape variance plus a small c4 benefit, not a robust global
improvement. Keep the stable baseline unchanged. The next useful frontier is not
frontend logging; it is either scheduler shape policy, MoE expert routing/kernel
selection, or a deliberate benchmark-contract change such as token-only/greedy
output that should be measured separately and not mixed with serving-runtime
wins.

## 2026-07-25 Concurrent Partial-Prefill Startup Falsification

After the frontend logging and no-eager passes failed globally, the next
scheduler-shape hypothesis targeted vLLM's default `max_num_partial_prefills=1`.
For c4 endpoint traffic, allowing four concurrent partial prefills could in
principle reduce prefill serialization before decode.

The server was launched with the stable standard Triton shape plus:

```text
--max-num-partial-prefills 4
--max-long-partial-prefills 4
```

The CLI accepted and recorded both flags:

```text
'max_num_partial_prefills': 4,
'max_long_partial_prefills': 4
```

But vLLM rejected the configuration before serving:

```text
NotImplementedError: Concurrent Partial Prefill is not supported.
We recommend to remove Concurrent Partial Prefill from your config.
```

Conclusion: this is not a runnable +20 path without deeper scheduler support.
The idea remains conceptually aligned with the observed bottleneck, but the
current V1 engine will not allow it as a serve-time flag. Do not spend benchmark
time on this setting until the feature support boundary changes or SOCK owns a
compatible scheduler path.

## 2026-07-25 No Full-ISL Reservation Falsification

After concurrent partial prefill failed at startup, the next runnable scheduler
hypothesis was to relax full input-sequence reservation before admitting work.
The server used the stable standard Triton shape plus:

```text
--no-scheduler-reserve-full-isl
```

The c4 gate again looked attractive:

```text
medium_architecture_256 c4:   53.6950 -> 54.1797  (+0.90%)
long_cosmology_512 c4:        50.6543 -> 52.0940  (+2.84%)
long_context_summary_256 c4:  65.8769 -> 72.3488  (+9.82%)
Gate geomean delta:                                (+4.45%)
```

But the full suite showed the same shape-specific trap as the access-log probe:

```text
standard TRITON_ATTN full geomean:                 36.7329
standard TRITON_ATTN no-full-ISL-reserve geomean:  36.3787
Delta:                                             -0.96%
```

The useful positives were concentrated in c4 cells:

```text
short_codegen_128 c4:       56.9322 -> 60.3982  (+6.09%)
long_cosmology_512 c4:      50.6543 -> 52.2737  (+3.20%)
extended_generation_768 c4: 51.0662 -> 51.7474  (+1.33%)
```

But c1/c2 paid for it:

```text
long_cosmology_512 c2:      33.6570 -> 31.3659  (-6.81%)
extended_generation_768 c2: 33.2304 -> 31.6048  (-4.89%)
tiny_fact_64 c2:            34.6675 -> 33.6542  (-2.92%)
short_codegen_128 c1:       25.7130 -> 25.1588  (-2.16%)
```

Conclusion: disabling full-ISL reservation is not a global +20 route. It does
confirm a real shape-specific theme: c4 throughput can be improved by admitting
work more aggressively, but the same policy taxes c1/c2 enough to lose the full
endpoint mix. A future scheduler win likely needs per-shape/adaptive admission
rather than another process-wide flag.

## 2026-07-25 Robust Standard-KV vs TMH Big-Bench Rebaseline

The earlier small-sample standard/TMH comparisons were too noisy to support a
credible +20 claim. This pass reran the production-shaped endpoint suite with a
larger durable sample, writing one artifact per case so a network interruption
could not erase the run.

Benchmark contract:

```text
model: Qwen/Qwen3-30B-A3B-GPTQ-Int4
runs: 10 measured, 2 warmup
concurrency levels: 1, 2, 4
cases: tiny_fact_64, short_codegen_128, medium_architecture_256,
       long_cosmology_512, long_context_summary_256, extended_generation_768
serve shape: --max-model-len 2048 --gpu-memory-utilization 0.35
             --max-num-batched-tokens 1024 --max-num-seqs 4
             --enforce-eager --language-model-only --skip-mm-profiling
             --attention-backend TRITON_ATTN
artifacts: benchmarks/2026-07-25-gmk-qwen3-30b-robust-pairs/
summary:   benchmarks/2026-07-25-gmk-qwen3-30b-robust-pairs/analysis/
```

Result:

```text
standard KV geomean completion tok/s: 36.5928
TMH geomean completion tok/s:         31.4599
Delta:                                -14.03%
```

Cell-level result:

```text
extended_generation_768 c1: std=23.7281 tmh=19.3986  (-18.25%)
extended_generation_768 c2: std=31.4565 tmh=29.5086  ( -6.19%)
extended_generation_768 c4: std=51.0567 tmh=46.2539  ( -9.41%)
long_context_summary_256 c1: std=23.5821 tmh=19.2927 (-18.19%)
long_context_summary_256 c2: std=40.0829 tmh=25.5578 (-36.24%)
long_context_summary_256 c4: std=64.5823 tmh=37.8040 (-41.46%)
long_cosmology_512 c1: std=24.3247 tmh=19.5329      (-19.70%)
long_cosmology_512 c2: std=32.3613 tmh=30.1226      ( -6.92%)
long_cosmology_512 c4: std=51.1248 tmh=45.5535      (-10.90%)
medium_architecture_256 c1: std=25.8058 tmh=22.1319 (-14.24%)
medium_architecture_256 c2: std=35.6565 tmh=29.6048 (-16.97%)
medium_architecture_256 c4: std=53.6564 tmh=49.0617 ( -8.56%)
short_codegen_128 c1: std=27.7834 tmh=25.2315       ( -9.18%)
short_codegen_128 c2: std=35.7034 tmh=33.2480       ( -6.88%)
short_codegen_128 c4: std=55.0423 tmh=52.0787       ( -5.38%)
tiny_fact_64 c1: std=29.9825 tmh=27.0825            ( -9.67%)
tiny_fact_64 c2: std=34.5563 tmh=33.8173            ( -2.14%)
tiny_fact_64 c4: std=52.9256 tmh=53.5568            ( +1.19%)
```

Reality: the robust run erased the comforting small-sample interpretation. TMH
is not near parity under this all-raw, standard-like endpoint contract; it is
back at a meaningful `-14.03%` gap. The largest lamp is
`long_context_summary_256 c4`, where standard reached `64.5823` and TMH reached
only `37.8040` completion tok/s.

Interpretation: the remaining tax is probably not physical compression itself in
this benchmark, because the low-pressure path should be all raw. The tax is more
likely one or more all-raw TMH compatibility layers still sitting on the hot path:
request-row indirection, block-table/view handoff, backend selection, or scheduler
metadata shape. The short/medium losses show fixed overhead; the long-context c4
loss shows a deeper decode-shape mismatch.

Conclusion: do not claim the old `-5%` gap as current truth. The next moonshot
must make TMH's all-raw path observationally identical to standard KV at the
attention/scheduler boundary, then reintroduce TMH-specific behavior only when
warm pages are reachable. In first-principles terms, the winning abstraction is
not "faster TMH kernels"; it is "zero-overhead standard KV when the request set
is all raw, with TMH activated only under memory pressure."

## 2026-07-25 Robust SOCK Standard-KV vs Runnable Upstream vLLM

After the robust standard/TMH rebaseline showed TMH itself was still negative,
the next question was broader: how much of SOCK's current stack beats upstream
vanilla vLLM on the same Strix Halo ROCm endpoint contract?

Benchmark contract:

```text
model: Qwen/Qwen3-30B-A3B-GPTQ-Int4
runs: 10 measured, 2 warmup
concurrency levels: 1, 2, 4
cases: tiny_fact_64, short_codegen_128, medium_architecture_256,
       long_cosmology_512, long_context_summary_256, extended_generation_768
SOCK source: benchmarks/2026-07-25-gmk-qwen3-30b-robust-pairs/standard-kv/
vanilla source: benchmarks/2026-07-25-gmk-qwen3-30b-sock-vs-vanilla-vllm-runnable/vanilla/
summary: benchmarks/2026-07-25-gmk-qwen3-30b-sock-vs-vanilla-vllm-runnable/analysis/
serve shape: --max-model-len 2048 --gpu-memory-utilization 0.35
             --max-num-batched-tokens 1024 --max-num-seqs 4
             --enforce-eager --language-model-only --skip-mm-profiling
             --attention-backend TRITON_ATTN
```

The vanilla environment used upstream vLLM commit
`190be7dad2afa6684902324e0dffa2dc0229a364`, ROCm 7.2.4, and the local
`torch-2.11.0+gfx1151` wheel. This was not a pristine `pip install vllm` run:
upstream needed compatibility work just to serve this Strix Halo GPTQ model.
The runnable vanilla configuration removed the incompatible CPU `torchvision`,
provided a text-only `torchvision` import stub for unconditional multimodal
warmup imports, and disabled `RDNAHybridW4A16LinearKernel` after it failed with
`zp shape mismatch`. Those changes are environment/runnability fixes, not SOCK
throughput features.

Result:

```text
SOCK standard-KV geomean completion tok/s: 36.5928
runnable upstream vLLM geomean tok/s:      38.0710
Delta:                                     -3.88%
```

Cell-level result:

```text
tiny_fact_64 c1:               sock=29.9825 vanilla=31.2489  ( -4.05%)
tiny_fact_64 c2:               sock=34.5563 vanilla=34.5823  ( -0.08%)
tiny_fact_64 c4:               sock=52.9256 vanilla=55.3389  ( -4.36%)
short_codegen_128 c1:          sock=27.7834 vanilla=29.3463  ( -5.33%)
short_codegen_128 c2:          sock=35.7034 vanilla=36.3170  ( -1.69%)
short_codegen_128 c4:          sock=55.0423 vanilla=57.8835  ( -4.91%)
medium_architecture_256 c1:    sock=25.8058 vanilla=28.0122  ( -7.88%)
medium_architecture_256 c2:    sock=35.6565 vanilla=36.3338  ( -1.86%)
medium_architecture_256 c4:    sock=53.6564 vanilla=54.7798  ( -2.05%)
long_cosmology_512 c1:         sock=24.3247 vanilla=26.8956  ( -9.56%)
long_cosmology_512 c2:         sock=32.3613 vanilla=34.9785  ( -7.48%)
long_cosmology_512 c4:         sock=51.1248 vanilla=54.2235  ( -5.71%)
long_context_summary_256 c1:   sock=23.5821 vanilla=25.1748  ( -6.33%)
long_context_summary_256 c2:   sock=40.0829 vanilla=37.2574  ( +7.58%)
long_context_summary_256 c4:   sock=64.5823 vanilla=67.2209  ( -3.93%)
extended_generation_768 c1:    sock=23.7281 vanilla=25.0400  ( -5.24%)
extended_generation_768 c2:    sock=31.4565 vanilla=32.8328  ( -4.19%)
extended_generation_768 c4:    sock=51.0567 vanilla=51.9019  ( -1.63%)
```

Reality: SOCK's standard-KV stack is close to runnable upstream vanilla but is
not ahead in the robust 18-cell endpoint mean. The gap is much smaller than the
current TMH tax (`-3.88%` vs `-14.03%`), and one cell is clearly positive
(`long_context_summary_256 c2`, `+7.58%`), but the full answer is still negative.

Interpretation: the broad SOCK stack is no longer catastrophically divergent
from upstream, but the current fork has not yet produced a global throughput win
for this all-raw standard-KV contract. The most valuable contradiction is that
SOCK can beat vanilla in one long-context c2 cell while losing most c1 and c4
cells. That points away from a single bad kernel and toward scheduler/admission,
request-shape policy, or per-concurrency overhead in the fork.

Conclusion: the +20 target is still possible only if the next work attacks the
core serving path, not TMH cosmetics. First-principles next step: treat standard
KV as the invariant floor. Any TMH or SOCK path must first match runnable
upstream vLLM at the scheduler/attention boundary, then win under memory pressure
or repeated-prefix pressure. The next falsifiable experiment should isolate why
`long_context_summary_256 c2` is positive while nearby cells are negative, then
turn that shape-specific behavior into an adaptive policy instead of another
process-wide flag.

## 2026-07-25 TMH Memory-Pressure Capacity Frontier

The throughput suite was the wrong primary judge for TMH's intended value. If
TMH exists to reduce KV memory pressure at inference time, the first falsifiable
question is not "does it decode faster at 2K?" but "does the same KV memory
budget admit more logical context?"

This pass compared vLLM's own startup capacity accounting for standard KV and
TMH under the same model, same available KV memory, same context length, and
same serve shape except `--kv-layout`:

```text
model: Qwen/Qwen3-30B-A3B-GPTQ-Int4
gpu memory utilization: 0.35
available KV cache memory: 6.50 GiB
max seqs: 16
standard: --kv-layout standard
TMH:      --kv-layout tmh --tmh-hot-budget-pct 25
artifacts: benchmarks/2026-07-25-gmk-qwen3-30b-tmh-memory-pressure-capacity/
summary:   benchmarks/2026-07-25-gmk-qwen3-30b-tmh-memory-pressure-capacity/analysis/summary.json
```

Result:

```text
max_model_len=8192:
  standard KV cache tokens: 70,944   max concurrency:  8.66x
  TMH KV cache tokens:      123,312  max concurrency: 15.05x
  TMH capacity delta:       +73.79%

max_model_len=16384:
  standard KV cache tokens: 70,944   max concurrency:  4.33x
  TMH KV cache tokens:      123,312  max concurrency:  7.53x
  TMH capacity delta:       +73.90%
```

Reality: this is the cleanest positive TMH result so far. With identical
`6.50 GiB` available KV memory, TMH admits `123,312` logical KV tokens versus
standard's `70,944`, a `+73.82%` token-capacity lift. At 16K context, that moves
the admitted concurrency frontier from roughly four full-context requests to
roughly seven.

Interpretation: TMH's compression value is real in the allocator/capacity model.
That does not contradict the `-14.03%` throughput result; it explains the trade:
TMH currently pays hot-path overhead, but it buys substantially more logical KV
residency under memory pressure. The correct benchmark axis is therefore not
low-pressure tok/s parity alone. It is capacity-adjusted serving: requests served
per GiB, tail latency when standard KV must queue, and survival when the working
set exceeds standard's raw KV pool.

Conclusion: keep two scoreboards. Standard-KV throughput remains the invariant
floor for low-pressure serving. TMH's value should be judged by a pressure
frontier: maximum resident context, maximum admitted long-context concurrency,
error/queue avoidance beyond standard capacity, and throughput per GiB. The next
runtime benchmark should target the 16K frontier directly: run concurrent
long-context requests at c4/c6/c8. Standard should be at or beyond its admitted
capacity around c6/c8, while TMH should still have resident KV headroom.

