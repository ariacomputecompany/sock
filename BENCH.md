# Benchmarks

This file is the durable benchmark ledger for live sock runs. Raw bulky logs and
full endpoint responses can live under `tmp/` during development; durable
summaries should be promoted into `benchmarks/<run-id>/summary.json`.

## 2026-07-18: GMK EVO-X2, Qwen3-4B, ROCm WSL

### Status

| Claim | Current evidence | Status |
| --- | --- | --- |
| Cleaner DX to runnable inference | `sock serve` reaches a live OpenAI-compatible endpoint on the AMD WSL/ROCm machine with the repo-local runtime defaults. PyPI vanilla installs a CUDA stack and fails. Official upstream ROCm wheel needs explicit ROCm/WSL env plus three local import/detection patches before it serves. | Proven on this machine |
| Shorter time to runnable inference | Fresh `sock serve` restart reached `/health` in 48 seconds. Patched official upstream ROCm wheel reached `/health` in 52 seconds after manual install/env/patching. PyPI vanilla never reached `/health`. | Proven for runnable endpoint on this workload |
| Higher throughput than vanilla | The expanded 6-case suite over concurrency 1, 2, and 4 shows sock and patched official upstream ROCm at statistical parity for this Qwen3-4B eager endpoint workload. | Not proven here; measured parity |

### Hardware And Runtime

| Field | Value |
| --- | --- |
| Machine | GMK EVO-X2 / AMD Ryzen AI Max+ 395 with Radeon 8060S |
| OS | Linux WSL2, glibc 2.39 |
| Accelerator vendor | AMD |
| GPU arch | `gfx1151` |
| ROCm/driver reported by sock doctor | `7.14.0~pre3-29052710811` |
| Python ABI | `cp312` |
| sock runtime | vendored runtime, `vllm 0.25.1`, `torch 2.11.0+gitd0c8b1f`, HIP `7.2.53211`, `device_count=1` |
| vanilla runtime attempted | PyPI `vllm 0.25.1`, `torch 2.11.0`, CUDA `13.0`, HIP `null`, `device_count=0` |
| upstream ROCm runtime attempted | Official ROCm wheel `vllm 0.25.1+rocm723`, `torch 2.11.0+gitd0c8b1f`, HIP `7.2.53211`, `device_count=1` |

### Model And Server Settings

| Field | Value |
| --- | --- |
| Model | `Qwen/Qwen3-4B` |
| Endpoint | OpenAI-compatible `/v1/completions` |
| Prompt | `Explain how the universe came into being...` long-form cosmology prompt |
| `max_model_len` | `1024` |
| `gpu_memory_utilization` | `0.8` |
| `enforce_eager` | `true` |
| `max_tokens` | `512` |
| `temperature` | `0.2` |
| Initial benchmark shape | 1 warmup request + 5 measured requests |
| Expanded suite shape | 6 prompt classes, 1 warmup batch per case/concurrency, 2 measured batches per case/concurrency, concurrency 1/2/4 |

### sock Results

Fresh restart command:

```bash
SOCK_RUNTIME_PROFILE=rocm target/debug/sock serve Qwen/Qwen3-4B \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.8 \
  --enforce-eager \
  --disable-log-stats
```

Startup result:

| Metric | Value |
| --- | --- |
| Health status | healthy |
| Time to `/health` | 48 seconds |
| Checkpoint size | 7.49 GiB |
| Model memory | 7.56 GiB |
| Available KV cache memory | 68.21 GiB |
| GPU KV cache size | 496,672 tokens |
| Max concurrency at 1024 tokens | 485.03x |
| Attention backend selected | `TRITON_ATTN` after `ROCM_ATTN` and `TURBOQUANT` were rejected |

This initial baseline exposed a production backend-selection bug: ROCm custom
attention rejected `block_size=None` during backend probing even though the
runtime block size is materialized later. The fix allows unmaterialized
`block_size` during eligibility checks while still rejecting concrete non-16
block sizes on gfx1x. After the fix, `sock serve` naturally selects
`ROCM_ATTN`.

Endpoint throughput after fresh restart:

| Metric | Min | Mean | Median | Max | P90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Completion tok/s | 25.0569 | 25.2355 | 25.0992 | 25.7161 | 25.7161 |
| Total tok/s | 27.3570 | 27.5520 | 27.4033 | 28.0767 | 28.0767 |
| Elapsed seconds | 19.9097 | 20.2908 | 20.3990 | 20.4335 | 20.4335 |
| Completion tokens | 512 | 512 | 512 | 512 | 512 |
| Total tokens | 559 | 559 | 559 | 559 | 559 |

Earlier warm-server run, same endpoint and settings:

| Metric | Min | Mean | Median | Max | P90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Completion tok/s | 24.8395 | 25.1259 | 25.0648 | 25.6536 | 25.6536 |
| Total tok/s | 27.1197 | 27.4324 | 27.3657 | 28.0085 | 28.0085 |
| Elapsed seconds | 19.9582 | 20.3798 | 20.4271 | 20.6123 | 20.6123 |

### Vanilla vLLM Results

Vanilla setup attempted:

```bash
python3 -m venv /home/deepsaint/work/bench-vanilla-vllm/.venv
source /home/deepsaint/work/bench-vanilla-vllm/.venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install vllm
vllm serve Qwen/Qwen3-4B \
  --host 127.0.0.1 \
  --port 8001 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.8 \
  --enforce-eager \
  --disable-log-stats
```

Result:

| Attempt | Status | Time | Notes |
| --- | --- | ---: | --- |
| Naive upstream PyPI install | exited before serving | 2 seconds | Import-time circular import in upstream vLLM before endpoint startup |
| Same install with ROCm WSL env hints | exited before serving | 2 seconds | Same import-time failure |

The isolated vanilla environment also installed CUDA PyTorch and reported
`torch.version.cuda="13.0"`, `torch.version.hip=null`,
`torch.cuda.is_available()=false`, and `device_count=0`. That means the
straight upstream PyPI path is not a runnable AMD/ROCm baseline on this machine.

Failure excerpt:

```text
ImportError: cannot import name 'direct_register_custom_op' from partially initialized module 'vllm.utils.torch_utils'
```

### Upstream ROCm Wheel Results

Official upstream ROCm setup attempted:

```bash
python3 -m venv /home/deepsaint/work/bench-upstream-vllm-rocm-wheel/.venv
source /home/deepsaint/work/bench-upstream-vllm-rocm-wheel/.venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install 'vllm==0.25.1+rocm723' \
  --extra-index-url https://wheels.vllm.ai/rocm/752a3a504485790a2e8491cacbb35c137339ad34/rocm723
```

The official ROCm wheel installs the correct GPU stack and reports
`torch.version.hip="7.2.53211"`, `torch.cuda.is_available()=true`, and
`device_count=1`. It still did not serve out of the box on WSL because amdsmi
fails with `AMDSMI_STATUS_DRIVER_NOT_LOADED`.

Local patches required before upstream would serve:

| File | Patch |
| --- | --- |
| `vllm/platforms/interface.py` | Scope the WSL pin-memory `warning_once` to `process` so it does not import distributed rank state during `torch_utils` initialization. |
| `vllm/platforms/__init__.py` | When `VLLM_TARGET_DEVICE=rocm` and amdsmi fails, activate ROCm if PyTorch HIP is present and sees a device. |
| `vllm/platforms/rocm.py` | Use plain `logger.warning` for the amdsmi GCN arch fallback to avoid another import-time distributed-state cycle. |

Patched upstream ROCm command:

```bash
VLLM_TARGET_DEVICE=rocm \
VLLM_USE_V2_MODEL_RUNNER=0 \
VLLM_WSL2_ENABLE_PIN_MEMORY=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
PYTHONNOUSERSITE=1 \
PYTHONHASHSEED=0 \
TOKENIZERS_PARALLELISM=false \
vllm serve Qwen/Qwen3-4B \
  --host 127.0.0.1 \
  --port 8001 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.8 \
  --enforce-eager \
  --disable-log-stats
```

Patched upstream startup result:

| Metric | Value |
| --- | --- |
| Health status | healthy |
| Time to `/health` | 52 seconds |
| Checkpoint size | 7.49 GiB |
| Model memory | 7.56 GiB |
| Available KV cache memory | 68.21 GiB |
| GPU KV cache size | 496,672 tokens |
| Max concurrency at 1024 tokens | 485.03x |
| Attention backend selected | `ROCM_ATTN` after `TURBOQUANT` was rejected |

Patched upstream endpoint throughput:

| Metric | Min | Mean | Median | Max | P90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Completion tok/s | 24.9711 | 25.2333 | 25.2652 | 25.5303 | 25.5303 |
| Total tok/s | 27.2634 | 27.5497 | 27.5845 | 27.8739 | 27.8739 |
| Elapsed seconds | 20.0546 | 20.2917 | 20.2650 | 20.5037 | 20.5037 |
| Completion tokens | 512 | 512 | 512 | 512 | 512 |
| Total tokens | 559 | 559 | 559 | 559 | 559 |

Comparison:

| Runtime | Time to health | Mean completion tok/s | Mean total tok/s | Notes |
| --- | ---: | ---: | ---: | --- |
| sock vendored runtime | 48s | 25.2355 | 27.5520 | Runs via `sock serve` with repo-local runtime defaults |
| patched upstream ROCm wheel | 52s | 25.2333 | 27.5497 | Needs manual ROCm wheel install, explicit env, and three WSL/amdsmi patches |
| PyPI vanilla | n/a | n/a | n/a | Installs CUDA stack and fails before serving |

### Expanded Suite Results

The expanded suite uses `scripts/sock_endpoint_bench_suite.py` against both
OpenAI-compatible endpoints. It runs six prompt classes from tiny factual output
through long-form generation and long-context summarization, with concurrency
levels 1, 2, and 4. Each case/concurrency pair uses one warmup batch and two
measured batches.

Post-fix startup comparison:

| Runtime | Time to health | Attention backend | Notes |
| --- | ---: | --- | --- |
| sock vendored runtime | 56s | `ROCM_ATTN` | Default backend selection after the `block_size=None` fix |
| patched upstream ROCm wheel | 52s | `ROCM_ATTN` | Manual ROCm wheel/env/patch path |

Mean completion tok/s by case and concurrency:

| Case | Concurrency | sock | patched upstream | sock delta |
| --- | ---: | ---: | ---: | ---: |
| `tiny_fact_64` | 1 | 25.2527 | 25.2995 | -0.185% |
| `tiny_fact_64` | 2 | 46.8422 | 44.2165 | +5.938% |
| `tiny_fact_64` | 4 | 93.7006 | 93.9555 | -0.271% |
| `short_codegen_128` | 1 | 25.2840 | 25.0988 | +0.738% |
| `short_codegen_128` | 2 | 49.5227 | 49.4282 | +0.191% |
| `short_codegen_128` | 4 | 95.1425 | 95.0075 | +0.142% |
| `medium_architecture_256` | 1 | 25.1850 | 25.5831 | -1.556% |
| `medium_architecture_256` | 2 | 50.5021 | 49.8364 | +1.336% |
| `medium_architecture_256` | 4 | 94.7088 | 94.2985 | +0.435% |
| `long_cosmology_512` | 1 | 25.1013 | 25.0967 | +0.018% |
| `long_cosmology_512` | 2 | 49.7113 | 49.6695 | +0.084% |
| `long_cosmology_512` | 4 | 92.6525 | 93.3443 | -0.741% |
| `long_context_summary_256` | 1 | 25.1928 | 24.5416 | +2.653% |
| `long_context_summary_256` | 2 | 48.0964 | 48.0434 | +0.110% |
| `long_context_summary_256` | 4 | 90.5848 | 91.6758 | -1.190% |
| `extended_generation_768` | 1 | 24.9289 | 25.1307 | -0.803% |
| `extended_generation_768` | 2 | 49.1037 | 48.8041 | +0.614% |
| `extended_generation_768` | 4 | 91.7100 | 91.9298 | -0.239% |

Expanded-suite conclusion: throughput is effectively parity for this specific
eager Qwen3-4B shape. sock wins the product/DX thesis here because it owns the
least-dependency ROCm path, the backend-selection policy, and the production fix
that made the default ROCm path choose `ROCM_ATTN` cleanly.

### Raw Artifacts

| Artifact | Purpose |
| --- | --- |
| `benchmarks/2026-07-18-gmk-qwen3-4b/summary.json` | Tracked compact summary |
| `tmp/bench-sock-restart-ready.json` | Fresh sock startup readiness result |
| `tmp/bench-sock-restart-serve.log` | Fresh sock serve log |
| `tmp/bench-sock-qwen3-4b-restart.json` | Full fresh sock endpoint benchmark responses |
| `tmp/bench-sock-qwen3-4b.json` | Full earlier warm-server sock endpoint benchmark responses |
| `tmp/bench-vanilla-naive-serve.log` | Naive vanilla failure log |
| `tmp/bench-vanilla-rocm-env-serve.log` | Vanilla with ROCm env hints failure log |
| `tmp/bench-upstream-rocm-wheel-patched-env2-serve.log` | Patched official upstream ROCm wheel serve log |
| `tmp/bench-upstream-rocm-wheel-qwen3-4b.json` | Full patched upstream ROCm endpoint benchmark responses |
| `tmp/bench-suite-upstream-rocm-wheel-qwen3-4b.json` | Full expanded-suite upstream ROCm responses |
| `tmp/bench-suite-sock-fixed-qwen3-4b.json` | Full expanded-suite fixed sock responses |
| `benchmarks/2026-07-18-gmk-qwen3-4b/suite-summary.json` | Tracked compact expanded-suite comparison |
| `benchmarks/2026-07-18-gmk-qwen3-4b/upstream-rocm-wsl.patch` | Tracked upstream patch needed to make official ROCm wheel serve under WSL |

### Sitrep

sock has a real production endpoint baseline on the GMK AMD machine:
`Qwen/Qwen3-4B` serves reliably at `max_model_len=1024` and produces about
25 completion tok/s for a 512-token long-form prompt.

The DX claim is strong: the sock path reaches live inference, while the straight
vanilla PyPI path downloads a CUDA-oriented stack, sees no GPU, and fails before
serving. The official upstream ROCm wheel can be made runnable on this WSL AMD
machine, but only after manual wheel-index selection, explicit ROCm/WSL runtime
environment, and three local patches that sock already carries.

The throughput claim is not proven on this Qwen3-4B eager endpoint workload.
Once upstream is repaired enough to run, it reaches statistical parity with sock
across the expanded prompt/concurrency suite. The engineering win from this pass
is stronger than a narrow tok/s headline: the robust benchmark found a real
default ROCm backend-selection bug, sock now selects `ROCM_ATTN` cleanly, and the
upstream comparison requires manual dependency/index/env/patch work that sock is
designed to erase.

The next performance proof should target modes where sock intentionally differs
from upstream, such as broader context, compilation/cache warmup behavior,
non-eager paths, larger batch curves, and backend-selection policy under mixed
model shapes.

## Qwen3-32B 2-bit GPTQ Large-Model Suite

This run validates the new SOC AutoGPTQ 2-bit ROCm path against a real
large dense checkpoint: `kaitchup/Qwen3-32B-autoround-2bit-gptq`. The
suite uses the same six prompt classes as the Qwen3-4B expanded benchmark,
with one warmup batch and two measured batches at concurrency levels 1, 2,
and 4. The full run completed successfully in 4113.4 seconds.

Startup and capacity:

| Metric | Value |
| --- | ---: |
| Checkpoint size | 12.22 GiB |
| Weight load time | 9.96 s |
| Model load time | 11.55 s |
| Model memory | 12.3 GiB |
| Engine warmup | 25.34 s |
| Available KV cache memory | 63.42 GiB |
| GPU KV cache size | 259,776 tokens |
| Max concurrency at 1024 tokens | 253.69x |
| Attention backend selected | `ROCM_ATTN` |

Mean completion tok/s by case and concurrency:

| Case | Concurrency | Mean completion tok/s | Mean total tok/s | Mean wall s |
| --- | ---: | ---: | ---: | ---: |
| `tiny_fact_64` | 1 | 3.3239 | 3.8952 | 19.2543 |
| `tiny_fact_64` | 2 | 6.6570 | 7.8012 | 19.2279 |
| `tiny_fact_64` | 4 | 13.3305 | 15.6216 | 19.2043 |
| `short_codegen_128` | 1 | 3.4319 | 3.9949 | 37.2976 |
| `short_codegen_128` | 2 | 6.7790 | 7.8911 | 37.7643 |
| `short_codegen_128` | 4 | 13.3482 | 15.5381 | 38.3577 |
| `medium_architecture_256` | 1 | 3.4256 | 3.7869 | 74.7314 |
| `medium_architecture_256` | 2 | 6.7505 | 7.4625 | 75.8479 |
| `medium_architecture_256` | 4 | 13.4219 | 14.8376 | 76.2929 |
| `long_cosmology_512` | 1 | 3.4124 | 3.7257 | 150.0390 |
| `long_cosmology_512` | 2 | 6.7657 | 7.3867 | 151.3519 |
| `long_cosmology_512` | 4 | 13.3469 | 14.5721 | 153.4467 |
| `long_context_summary_256` | 1 | 3.3880 | 12.5726 | 75.5612 |
| `long_context_summary_256` | 2 | 6.7188 | 24.9332 | 76.2039 |
| `long_context_summary_256` | 4 | 13.2919 | 49.3252 | 77.0398 |
| `extended_generation_768` | 1 | 3.4023 | 3.6016 | 225.7308 |
| `extended_generation_768` | 2 | 6.7544 | 7.1502 | 227.4075 |
| `extended_generation_768` | 4 | 13.4041 | 14.1895 | 229.1840 |

Quality and correctness notes:

- The first live chat request after the zero-point fix produced coherent Big Bang reasoning: 384 completion tokens in 111.63 s, or 3.44 completion tok/s. The pre-fix output was token soup, so this is a real numerical-path correctness fix, not just a startup fix.
- The full benchmark recorded 84 measured responses with zero low-ASCII/token-soup suspects. Completions-endpoint samples often continue instruction text, which is expected for raw completions prompts and should not be confused with the earlier corrupted generation.
- Throughput scales nearly linearly with request concurrency on this workload: roughly 3.3-3.4 completion tok/s at concurrency 1, 6.6-6.8 at concurrency 2, and 13.3-13.4 at concurrency 4.
- Patched upstream vLLM cannot run the same 2-bit benchmark: it rejects `bits=2, sym=True` with `ValueError: Unsupported quantization config: bits=2, sym=True` before serving. There is no honest vanilla throughput number for this model without carrying SOC's new 2-bit implementation.

Artifacts:

| Artifact | Purpose |
| --- | --- |
| `benchmarks/2026-07-18-gmk-qwen3-32b-2bit-gptq/suite-summary.json` | Tracked compact full-suite summary |
| `tmp/bench-suite-sock-qwen3-32b-2bit-gptq-full.json` | Full raw SOC endpoint responses and per-batch stats |
| `tmp/bench-large-qwen3-32b-2bit-fixed-serve.log` | SOC serve log for startup, backend, JIT, and request status |
