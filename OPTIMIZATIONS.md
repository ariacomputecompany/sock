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

The current best production-shaped TMH result is the shape-adaptive native
all-raw path:

```text
Same-day Ubuntu standard KV:         37.8412 geomean completion tok/s
TMH all-raw native prefill, Triton decode: 35.1083 geomean completion tok/s
TMH all-raw native prefill, always ROCm custom decode: 35.1797
TMH all-raw native prefill, gated ROCm custom decode:  35.4940

Current gap vs same-day standard: -6.20%
```

The long-context concurrency-4 hot cell improved materially:

```text
long_context_summary_256 c4
standard KV:             70.1954 completion tok/s
TMH Triton decode:       40.8045 completion tok/s
TMH always custom decode:46.6415 completion tok/s
TMH gated custom decode: 50.5300 completion tok/s
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
```

Relative to the adaptive `29.10` checkpoint, the current `35.4940` result is
`+21.97%`. Relative to the old physical `28.49` result, it is `+24.58%`.

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

The previous unsafe native handoff failure mode was an engine wedge during
endpoint smoke. The sanitized/gated native decode path survived health checks,
smoke, long-context slice, and the full endpoint suite.

## Remaining Work

The current `-6.20%` is a large improvement, but still not parity. The next
largest target remains long-context concurrency:

```text
long_context_summary_256 c4:
standard: 70.1954
gated TMH: 50.5300
remaining gap: about -28.0%
```

The likely next moonshot is deeper decode specialization:

- Tune or replace the `max_seq_len >= 640` gate with measured shape buckets.
- Investigate why standard c4 long-context reaches `70.1954` while TMH gated
  reaches `50.5300`, even though TMH is all-raw in this regime.
- Profile native decode block-table copy/gather overhead under c4.
- Compare generic Triton decode, ROCm custom decode, and any AITER decode
  variants on the same raw TMH cache.
- Consider a TMH-owned all-raw decode kernel only if native handoff overhead or
  layout impedance remains the limiter.

The implementation rule for the next pass should remain the same: do not tune
policy around a slow executor. Preserve the all-raw native fast path and move
only the shapes with evidence onto a faster decode backend.
