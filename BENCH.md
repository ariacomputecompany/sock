# Benchmarks

This file is the durable benchmark ledger for live sock runs. Raw bulky logs and
full endpoint responses can live under `tmp/` during development; durable
summaries should be promoted into `benchmarks/<run-id>/summary.json`.

## 2026-07-18: GMK EVO-X2, Qwen3-4B, ROCm WSL

### Status

| Claim | Current evidence | Status |
| --- | --- | --- |
| Cleaner DX to runnable inference | `sock serve` reaches a live OpenAI-compatible endpoint on the AMD WSL/ROCm machine with the repo-local runtime defaults. Upstream `pip install vllm && vllm serve ...` exits before serving. | Proven on this machine |
| Shorter time to runnable inference | Fresh `sock serve` restart reached `/health` in 48 seconds for Qwen3-4B at `max_model_len=1024`. Vanilla PyPI vLLM never reached `/health`; it exited in about 2 seconds at import time. | Proven for runnable endpoint; clean install/build timing still needs a timed cold setup run |
| Higher throughput than vanilla | sock produced stable endpoint throughput around 25.2 completion tok/s. Vanilla PyPI vLLM did not run on the ROCm target, so throughput comparison is blocked until we have a true ROCm-capable upstream baseline. | Sock measured; vanilla blocked |

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
| Benchmark shape | 1 warmup request + 5 measured requests |

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

### Sitrep

sock has a real production endpoint baseline on the GMK AMD machine:
`Qwen/Qwen3-4B` serves reliably at `max_model_len=1024` and produces about
25 completion tok/s for a 512-token long-form prompt.

The DX claim is already strong: the sock path reaches live inference, while the
straight vanilla PyPI path downloads a CUDA-oriented stack, sees no GPU, and
fails before serving. The throughput-over-vanilla claim is not yet measurable
against PyPI vanilla because vanilla is not runnable here. To prove that claim
cleanly, the next benchmark target should be an upstream-source ROCm build with
the same ROCm PyTorch class as sock, then run the same `scripts/sock_endpoint_bench.py`
matrix against both endpoints.
