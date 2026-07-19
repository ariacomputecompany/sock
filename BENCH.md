# Benchmarks

This is the durable performance ledger for live sock runs on the GMK Strix Halo
AMD/ROCm machine. Raw endpoint responses and bulky serve logs may live in
`tmp/`; compact summaries live under `benchmarks/<run-id>/`.

## Measurement Notes

| Metric | Current coverage |
| --- | --- |
| Completion throughput | Captured as completion tokens per second from OpenAI-compatible endpoint responses. |
| Total throughput | Captured as total tokens per second where endpoint usage reports prompt + completion tokens. |
| Wall clock latency | Captured as request elapsed seconds and suite elapsed seconds. |
| Startup latency | Captured as time to `/health` or server-ready bind where available. |
| Time to first token | Captured in streaming endpoint probes where the benchmark report includes `ttft_s`; older non-streaming runs omit it. |

## Testbed

| Field | Value |
| --- | --- |
| Machine | GMK EVO-X2 / AMD Ryzen AI Max+ 395 with Radeon 8060S |
| OS | Linux WSL2, glibc 2.39 |
| GPU arch | `gfx1151` |
| ROCm/driver reported by `sock doctor` | `7.14.0~pre3-29052710811` |
| Python ABI | `cp312` |
| sock runtime | vendored vLLM `0.25.1`, torch `2.11.0+gitd0c8b1f`, HIP `7.2.53211` |
| Upstream vLLM ROCm baseline | official ROCm wheel `vllm 0.25.1+rocm723`, torch `2.11.0+gitd0c8b1f`, HIP `7.2.53211` |

## Supported sock vs Upstream vLLM Comparison: Qwen3-4B

This is the current apples-to-apples comparison where both sock and an upstream
vLLM ROCm baseline served the same model on the same hardware.

| Field | Value |
| --- | --- |
| Model | `Qwen/Qwen3-4B` |
| Endpoint | `/v1/completions` |
| `max_model_len` | `1024` |
| `gpu_memory_utilization` | `0.8` |
| `enforce_eager` | `true` |
| Suite shape | 6 prompt classes, concurrency 1/2/4, 1 warmup batch, 2 measured batches |
| sock suite wall clock | 571.84 s |
| Upstream suite wall clock | 571.04 s |

### Startup

| Runtime | Ready after | Attention backend | Notes |
| --- | ---: | --- | --- |
| sock vendored runtime | 56 s | `ROCM_ATTN` | `sock serve` path |
| Upstream vLLM ROCm baseline | 52 s | `ROCM_ATTN` | upstream ROCm wheel baseline |

### Single Long-Form 512-Token Prompt

| Runtime | Mean completion tok/s | Mean total tok/s | Mean wall clock/request | Completion tokens |
| --- | ---: | ---: | ---: | ---: |
| sock vendored runtime | 25.24 | 27.55 | 20.29 s | 512 |
| Upstream vLLM ROCm baseline | 25.23 | 27.55 | 20.29 s | 512 |

### Multi-Case Endpoint Suite

| Case | Concurrency | sock completion tok/s | sock wall s | Upstream completion tok/s | Upstream wall s | sock delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `tiny_fact_64` | 1 | 25.25 | 2.55 | 25.30 | 2.54 | -0.18% |
| `tiny_fact_64` | 2 | 46.84 | 2.77 | 44.22 | 3.00 | +5.94% |
| `tiny_fact_64` | 4 | 93.70 | 2.75 | 93.96 | 2.74 | -0.27% |
| `short_codegen_128` | 1 | 25.28 | 5.06 | 25.10 | 5.10 | +0.74% |
| `short_codegen_128` | 2 | 49.52 | 5.17 | 49.43 | 5.18 | +0.19% |
| `short_codegen_128` | 4 | 95.14 | 5.38 | 95.01 | 5.39 | +0.14% |
| `medium_architecture_256` | 1 | 25.18 | 10.16 | 25.58 | 10.01 | -1.56% |
| `medium_architecture_256` | 2 | 50.50 | 10.14 | 49.84 | 10.27 | +1.34% |
| `medium_architecture_256` | 4 | 94.71 | 10.81 | 94.30 | 10.86 | +0.43% |
| `long_cosmology_512` | 1 | 25.10 | 20.40 | 25.10 | 20.40 | +0.02% |
| `long_cosmology_512` | 2 | 49.71 | 20.60 | 49.67 | 20.62 | +0.08% |
| `long_cosmology_512` | 4 | 92.65 | 22.11 | 93.34 | 21.94 | -0.74% |
| `long_context_summary_256` | 1 | 25.19 | 10.17 | 24.54 | 10.43 | +2.65% |
| `long_context_summary_256` | 2 | 48.10 | 10.65 | 48.04 | 10.66 | +0.11% |
| `long_context_summary_256` | 4 | 90.58 | 11.31 | 91.68 | 11.17 | -1.19% |
| `extended_generation_768` | 1 | 24.93 | 30.81 | 25.13 | 30.56 | -0.80% |
| `extended_generation_768` | 2 | 49.10 | 31.28 | 48.80 | 31.47 | +0.61% |
| `extended_generation_768` | 4 | 91.71 | 33.50 | 91.93 | 33.42 | -0.24% |

Result: Qwen3-4B throughput is effectively parity on this eager ROCm endpoint
shape. sock's win for this comparison is the shorter, cleaner path to a runnable
ROCm endpoint, not a meaningful tok/s advantage on this small model.

## sock Large-Model Runs

These runs validate the sock runtime paths that matter for the larger AMD box:
AutoGPTQ 2-bit, AutoGPTQ 4-bit, and MoE WNA16. Comparable upstream vLLM numbers
are not recorded here unless the upstream runtime supports the same model and
quantization path end-to-end.

### Qwen3-32B AutoGPTQ 2-Bit

| Field | Value |
| --- | ---: |
| Model | `kaitchup/Qwen3-32B-autoround-2bit-gptq` |
| Suite wall clock | 4113.42 s |
| Runs per case/concurrency | 2 |
| Warmups per case/concurrency | 1 |
| Checkpoint size | 12.22 GiB |
| Weight load | 9.96 s |
| Model load | 11.55 s |
| Model memory | 12.30 GiB |
| Engine warmup | 25.34 s |
| KV cache memory | 63.42 GiB |
| KV cache tokens | 259,776 |
| Max concurrency at 1024 tokens | 253.69x |

| Case | Concurrency | Completion tok/s | Total tok/s | Wall s |
| --- | ---: | ---: | ---: | ---: |
| `tiny_fact_64` | 1 | 3.32 | 3.90 | 19.25 |
| `tiny_fact_64` | 2 | 6.66 | 7.80 | 19.23 |
| `tiny_fact_64` | 4 | 13.33 | 15.62 | 19.20 |
| `short_codegen_128` | 1 | 3.43 | 3.99 | 37.30 |
| `short_codegen_128` | 2 | 6.78 | 7.89 | 37.76 |
| `short_codegen_128` | 4 | 13.35 | 15.54 | 38.36 |
| `medium_architecture_256` | 1 | 3.43 | 3.79 | 74.73 |
| `medium_architecture_256` | 2 | 6.75 | 7.46 | 75.85 |
| `medium_architecture_256` | 4 | 13.42 | 14.84 | 76.29 |
| `long_cosmology_512` | 1 | 3.41 | 3.73 | 150.04 |
| `long_cosmology_512` | 2 | 6.77 | 7.39 | 151.35 |
| `long_cosmology_512` | 4 | 13.35 | 14.57 | 153.45 |
| `long_context_summary_256` | 1 | 3.39 | 12.57 | 75.56 |
| `long_context_summary_256` | 2 | 6.72 | 24.93 | 76.20 |
| `long_context_summary_256` | 4 | 13.29 | 49.33 | 77.04 |
| `extended_generation_768` | 1 | 3.40 | 3.60 | 225.73 |
| `extended_generation_768` | 2 | 6.75 | 7.15 | 227.41 |
| `extended_generation_768` | 4 | 13.40 | 14.19 | 229.18 |

Direct chat quality check: 384 completion tokens in 111.63 s (3.44 completion tok/s), coherent output.

### Qwen3-32B AutoGPTQ 4-Bit

| Field | Value |
| --- | ---: |
| Model | `kaitchup/Qwen3-32B-autoround-4bit-gptq` |
| Suite wall clock | 2725.41 s |
| Runs per case/concurrency | 2 |
| Warmups per case/concurrency | 1 |
| Checkpoint size | 18.01 GiB |
| Weight load | 80.11 s |
| Model load | 82.25 s |
| Model memory | 18.15 GiB |
| Engine warmup | 28.60 s |
| KV cache memory | 57.56 GiB |
| KV cache tokens | 235,744 |
| Max concurrency at 1024 tokens | 230.22x |

| Case | Concurrency | Completion tok/s | Total tok/s | Wall s |
| --- | ---: | ---: | ---: | ---: |
| `tiny_fact_64` | 1 | 5.23 | 6.13 | 12.23 |
| `tiny_fact_64` | 2 | 10.10 | 11.83 | 12.68 |
| `tiny_fact_64` | 4 | 20.03 | 23.48 | 12.78 |
| `short_codegen_128` | 1 | 5.20 | 6.05 | 24.64 |
| `short_codegen_128` | 2 | 10.27 | 11.95 | 24.93 |
| `short_codegen_128` | 4 | 20.11 | 23.41 | 25.46 |
| `medium_architecture_256` | 1 | 5.20 | 5.74 | 49.27 |
| `medium_architecture_256` | 2 | 10.24 | 11.31 | 50.02 |
| `medium_architecture_256` | 4 | 20.15 | 22.27 | 50.82 |
| `long_cosmology_512` | 1 | 5.16 | 5.63 | 99.23 |
| `long_cosmology_512` | 2 | 10.16 | 11.09 | 100.79 |
| `long_cosmology_512` | 4 | 20.06 | 21.90 | 102.09 |
| `long_context_summary_256` | 1 | 5.12 | 19.01 | 49.97 |
| `long_context_summary_256` | 2 | 10.06 | 37.32 | 50.91 |
| `long_context_summary_256` | 4 | 19.84 | 73.64 | 51.60 |
| `extended_generation_768` | 1 | 5.11 | 5.41 | 150.29 |
| `extended_generation_768` | 2 | 10.25 | 10.85 | 149.87 |
| `extended_generation_768` | 4 | 20.38 | 21.57 | 150.74 |

Direct chat quality check: 384 completion tokens in 75.15 s (5.11 completion tok/s), coherent output.

### Qwen3-30B-A3B MoE GPTQ Int4

| Field | Value |
| --- | ---: |
| Model | `Qwen/Qwen3-30B-A3B-GPTQ-Int4` |
| Serve profile | `max_model_len=512`, `max_num_seqs=8`, `max_num_batched_tokens=512`, `gpu_memory_utilization=0.35` |
| Suite wall clock | 238.26 s |
| Measured requests | 63 |
| Checkpoint size | 15.77 GiB |
| Weight load | 17.38 s |
| Model load | 19.81 s |
| Model memory | 15.56 GiB |
| Engine init/warmup | 7.02 s |
| KV cache memory | 17.62 GiB |
| KV cache tokens | 192,480 |
| Max concurrency at 512 tokens | 375.94x |

| Case | Concurrency | Completion tok/s | Total tok/s | Wall s |
| --- | ---: | ---: | ---: | ---: |
| `tiny_fact_64` | 1 | 29.71 | 34.82 | 2.16 |
| `tiny_fact_64` | 2 | 32.33 | 37.88 | 3.96 |
| `tiny_fact_64` | 4 | 51.63 | 60.50 | 4.96 |
| `short_codegen_128` | 1 | 29.96 | 34.88 | 4.27 |
| `short_codegen_128` | 2 | 34.45 | 40.10 | 7.43 |
| `short_codegen_128` | 4 | 57.13 | 66.50 | 9.07 |
| `medium_architecture_256` | 1 | 30.16 | 33.34 | 8.49 |
| `medium_architecture_256` | 2 | 33.90 | 37.48 | 15.11 |
| `medium_architecture_256` | 4 | 53.88 | 59.56 | 19.02 |

Direct chat quality check: 220 completion tokens in 8.86 s (24.84 completion tok/s), coherent output.

## TMH Runtime Integration: Qwen3-30B-A3B MoE GPTQ Int4

This run compares regular `sock serve` against the TMH allocator-path runtime
policy on the same endpoint suite. The important boundary is explicit:
`--tmh-kv-policy accounting` is wired into live vendored-vLLM `KVCacheManager`
allocation and records TMH pressure during real traffic, while
`--tmh-kv-policy physical` currently fails closed until mixed-fidelity warm-page
tensors and attention kernels are implemented.

| Field | Value |
| --- | --- |
| Model | `Qwen/Qwen3-30B-A3B-GPTQ-Int4` |
| Serve path | `sock serve` OpenAI-compatible endpoint |
| Serve profile | `max_model_len=2048`, `max_num_seqs=4`, `max_num_batched_tokens=1024`, `gpu_memory_utilization=0.35`, `enforce_eager=true` |
| Suite shape | 10 smoke pressure cases, concurrency 1/2/4, 1 warmup batch, 1 measured batch, 10 streaming TTFT probes |
| TMH policy | `--tmh-kv-policy accounting --tmh-hot-budget-pct 25` |

| Runtime mode | Wall clock | Mean TTFT | Target retention mean | C1 mean tok/s | C2 mean tok/s | C4 mean tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Regular KV | 349.93 s | 0.1179 s | 81.667% | 28.5550 | 35.4938 | 64.9079 |
| TMH accounting, allocation logging on | 359.56 s | 0.1202 s | 80.833% | 27.9153 | 33.7384 | 61.6312 |
| TMH accounting, allocation logging off | 354.52 s | 0.1224 s | 80.833% | 28.0377 | 34.7627 | 63.8627 |

| TMH mode | Wall delta vs regular | C1 tok/s delta | C2 tok/s delta | C4 tok/s delta |
| --- | ---: | ---: | ---: | ---: |
| Accounting + allocation logging | +2.751% | -2.240% | -4.946% | -5.048% |
| Accounting without allocation logging | +1.312% | -1.812% | -2.060% | -1.610% |

Allocator-path pressure recorded from the live vLLM core:

| Metric | Value |
| --- | ---: |
| TMH allocation pressure log lines | 14,016 |
| Old-KV pressure rows | 14,016 |
| Old/warm reduction floor vs same-hot uniform-int8 old KV | 16.667% |
| Old/warm reduction mean vs same-hot uniform-int8 old KV | 16.667% |
| Total effective reduction floor vs same-hot uniform-int8 total KV | 7.407% |
| Total effective reduction mean vs same-hot uniform-int8 total KV | 9.102% |

Readout: TMH is now present in the live allocator path and survives real
endpoint traffic without server instability. This proves runtime integration for
policy/accounting, not physical mixed-fidelity KV execution. The physical runtime
claim remains deliberately blocked until the warm-page tensor layout and
attention kernels are implemented and benchmarked against this same suite.

## Artifacts

| Artifact | Purpose |
| --- | --- |
| `benchmarks/2026-07-18-gmk-qwen3-4b/summary.json` | Qwen3-4B compact sock/upstream summary |
| `benchmarks/2026-07-18-gmk-qwen3-4b/suite-summary.json` | Qwen3-4B compact suite comparison |
| `benchmarks/2026-07-18-gmk-qwen3-32b-2bit-gptq/suite-summary.json` | Qwen3-32B 2-bit compact suite summary |
| `benchmarks/2026-07-18-gmk-qwen3-32b-4bit-gptq/suite-summary.json` | Qwen3-32B 4-bit compact suite summary |
| `benchmarks/2026-07-18-gmk-qwen3-30b-a3b-gptq-int4/suite-summary.json` | Qwen3-30B-A3B MoE compact suite summary |
| `artifacts/tmh_runtime_integration/REPORT.md` | Matched regular vs TMH allocator-path endpoint comparison |
| `artifacts/tmh_runtime_integration/summary.json` | Machine-readable TMH runtime integration summary |
| `artifacts/tmh_runtime_integration/logs/tmh_accounting_server.log` | TMH allocation-pressure log with 14,016 live allocator records |
| `tmp/bench-suite-sock-fixed-qwen3-4b.json` | Raw sock Qwen3-4B endpoint suite |
| `tmp/bench-suite-upstream-rocm-wheel-qwen3-4b.json` | Raw upstream vLLM ROCm Qwen3-4B endpoint suite |
| `tmp/bench-suite-sock-qwen3-32b-2bit-gptq-full.json` | Raw sock Qwen3-32B 2-bit suite |
| `tmp/bench-suite-sock-qwen3-32b-4bit-gptq-full.json` | Raw sock Qwen3-32B 4-bit suite |
| `tmp/bench-suite-sock-qwen3-30b-a3b-gptq-int4-small.json` | Raw sock Qwen3-30B-A3B MoE suite |
