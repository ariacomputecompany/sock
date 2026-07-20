# Benchmarks

This is the durable performance ledger for live sock runs on the GMK Strix Halo
AMD/ROCm machine and rented NVIDIA/CUDA validation hosts. Raw endpoint responses
and bulky serve logs may live in `tmp/`; compact summaries live under
`benchmarks/<run-id>/`.

## Measurement Notes

| Metric | Current coverage |
| --- | --- |
| Completion throughput | Captured as completion tokens per second from OpenAI-compatible endpoint responses. |
| Total throughput | Captured as total tokens per second where endpoint usage reports prompt + completion tokens. |
| Wall clock latency | Captured as request elapsed seconds and suite elapsed seconds. |
| Startup latency | Captured as time to `/health` or server-ready bind where available. |
| Time to first token | Captured in streaming endpoint probes where the benchmark report includes `ttft_s`; older non-streaming runs omit it. |

## Runtime Bring-Up Contract

Fresh AMD/ROCm and NVIDIA/CUDA machines should use `sock install-runtime` as the
canonical zero-to-runnable path. The command resolves `runtime.buildplan.json`,
creates the vendored `vllm/.venv`, installs the backend-neutral top-level
`requirements.txt`, installs the accelerator-specific vendored vLLM requirement
set, and builds the vendored vLLM editable package with a deterministic runtime
environment.
Host preflight is intentionally narrow: compiler/toolchain, Git, Python headers,
Python venv support, and the vendor accelerator probe. Python build tools such
as CMake and Ninja are installed into `vllm/.venv` from `requirements.txt`.

Use `--dry-run --format json` to record the exact build profile, environment,
requirements, and command steps before applying changes:

```bash
cargo run --bin sock -- install-runtime --profile cuda --build-profile gptq-marlin --dry-run --format json
cargo run --bin sock -- install-runtime --profile rocm --build-profile core --dry-run --format json
cargo run --bin sock -- install-runtime --profile auto --preflight-only
cargo run --bin sock -- install-runtime --profile cuda --build-profile gptq-marlin --recreate-venv
```

The JSON plan includes preflight status and SHA-256 digests for every selected
requirement file. `--preflight-only` is a fail-closed readiness gate, while
`--dry-run` remains a non-mutating plan capture mode.

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

## NVIDIA/CUDA Bring-Up: RTX 4090

| Field | Value |
| --- | --- |
| Machine | Vast.ai RTX 4090 rental |
| GPU | NVIDIA GeForce RTX 4090 |
| Compute capability | `8.9` |
| Driver | `580.119.02` |
| CUDA reported by torch | `13.0` |
| Build profile | `minimal-dev` |
| Torch | `2.11.0+cu130` |
| Vendored vLLM import | `0.0.0+sock.cu128` metadata, `0.0.0+sock` package version |
| Canonical CLI validation | `sock serve --help` reached vendored `vllm serve` |

Production fix validated on this host: optional model-family fused CUDA kernels
and their torch library registrations are now controlled by the same
`VLLM_BUILD_FAMILY_MODEL_FUSED_OPS` build flag, so slim CUDA builds do not
produce unresolved symbols while full builds still register the fused ops.

### CUDA Endpoint Validation: Qwen2.5-0.5B

This run validates the canonical `sock serve` path on a live RTX 4090 after a
fresh deterministic `sock install-runtime --profile cuda --build-profile
minimal-dev --recreate-venv`.

| Field | Value |
| --- | ---: |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Endpoint | `/v1/chat/completions` streaming |
| `max_model_len` | `1024` |
| `gpu_memory_utilization` | `0.70` |
| `enforce_eager` | `true` |
| Attention backend | `FLASHINFER` |
| FlashInfer decode backend | `flashinfer-native` |
| KV cache memory | 15.21 GiB |
| KV cache tokens | 1,328,848 |
| Max concurrency at 1024 tokens | 1297.70x |
| Health ready after clean restart | 18 s |
| Engine init after warm cache | 1.61 s |
| Prompt tokens | 75 |
| Completion tokens | 384 |
| Total tokens | 459 |
| Time to first token | 0.057 s |
| Wall clock | 3.681 s |
| Decode throughput | 105.98 completion tok/s |
| End-to-end throughput | 104.33 completion tok/s |

Result: live CUDA serving now reaches a healthy OpenAI-compatible endpoint from
the canonical sock CLI, selects FlashInfer instead of a missing vendored
FlashAttention extension, and completes a streamed long-form inference without
unknown vLLM environment warnings.

### CUDA Larger-Model Stress: Qwen3-4B And Qwen3-8B

These runs compare eager mode against the default compiled/CUDA-graph serving
path on the same RTX 4090. The suite uses streamed `/v1/chat/completions`, one
warmup per case/concurrency, and two measured batches per case/concurrency.

| Model | Mode | Ready s | Avg TTFT ms | Avg completion tok/s | Profile |
| --- | --- | ---: | ---: | ---: | --- |
| `Qwen/Qwen3-4B` | eager (`--enforce-eager`) | 50.08 | 41.1 | 69.23 | `max_model_len=2048`, concurrency 1/2/4 |
| `Qwen/Qwen3-4B` | compiled/CUDA graphs | 52.07 | 25.4 | 98.38 | `max_model_len=2048`, concurrency 1/2/4 |
| `Qwen/Qwen3-8B` | eager (`--enforce-eager`) | 38.05 | 44.9 | 54.34 | `max_model_len=1024`, concurrency 1/2 |
| `Qwen/Qwen3-8B` | compiled/CUDA graphs | 48.07 | 27.3 | 56.74 | `max_model_len=1024`, concurrency 1/2 |

| Model | Mode | Model memory | KV cache memory | KV cache tokens | Max concurrency |
| --- | --- | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-4B` | eager | 7.56 GiB | 11.47 GiB | 83,552 | 40.80x at 2048 tokens |
| `Qwen/Qwen3-4B` | compiled/CUDA graphs | 7.56 GiB | 10.89 GiB | 79,280 | 38.71x at 2048 tokens |
| `Qwen/Qwen3-8B` | eager | 15.27 GiB | 4.75 GiB | 34,592 | 33.78x at 1024 tokens |
| `Qwen/Qwen3-8B` | compiled/CUDA graphs | 15.27 GiB | 3.68 GiB | 26,816 | 26.19x at 1024 tokens |

Production readout: for CUDA, the default production serving path should be
compiled/CUDA-graph mode, not `--enforce-eager`, when memory headroom has been
validated for the target model/profile. The throughput and TTFT gains are
material on Qwen3-4B and still positive on Qwen3-8B. `--enforce-eager` remains
the correct fallback for deterministic bring-up, debugging, and tight-memory
profiles where compile/CUDA-graph reservations reduce KV headroom too far.

Observed caveat: compiled mode emitted a vLLM AOT cache-save warning on both
Qwen3 models (`NoneType` has no `submodule_bytes_store`) but completed startup,
health, and all endpoint traffic. Treat this as a production observability item,
not a blocker for the 4090 profile.

### CUDA 30B GPTQ/Marlin Validation: Qwen3-30B-A3B

This run validates the canonical `sock serve` path against the same large model
family used for AMD pressure testing, but on a 24 GiB RTX 4090. The required
production profile is `gptq-marlin`; `minimal-dev` is intentionally too slim for
this model because it omits the GPTQ Marlin repack operator and WNA16 MoE
kernel family.

| Field | Value |
| --- | --- |
| Model | `Qwen/Qwen3-30B-A3B-GPTQ-Int4` |
| Endpoint | `/v1/completions` |
| Build profile | `gptq-marlin` |
| `max_model_len` | `2048` |
| `gpu_memory_utilization` | `0.90` |
| `max_num_batched_tokens` | `1024` |
| `max_num_seqs` | `4` |
| `enforce_eager` | `true` |
| Attention backend | `FLASHINFER` |
| Linear kernel | `MarlinLinearKernel` |
| MoE backend | `MARLIN` WNA16 |
| Checkpoint size | 15.77 GiB |
| Model memory | 15.61 GiB |
| Standard suite wall clock | 766.92 s |
| TMH accounting suite wall clock | 870.62 s |
| Raw summaries | `benchmarks/2026-07-19-rtx4090-qwen3-30b/` |

Startup finding: copying the AMD memory setting (`gpu_memory_utilization=0.35`)
onto a 24 GiB 4090 is invalid for this 30B model. The model weights load, but
the KV cache budget is negative after weight allocation. `0.90` leaves enough
budget for the configured 2048-token endpoint and completed the full suite.

| Case | Concurrency | Standard completion tok/s | TMH completion tok/s | TMH delta | Standard wall s | TMH wall s | Wall delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tiny_fact_64` | 1 | 26.54 | 25.70 | -3.2% | 2.41 | 2.49 | +3.3% |
| `tiny_fact_64` | 2 | 52.73 | 50.63 | -4.0% | 2.43 | 2.53 | +4.1% |
| `tiny_fact_64` | 4 | 102.94 | 98.18 | -4.6% | 2.49 | 2.61 | +4.8% |
| `short_codegen_128` | 1 | 26.39 | 25.27 | -4.2% | 4.85 | 5.07 | +4.4% |
| `short_codegen_128` | 2 | 52.43 | 50.10 | -4.4% | 4.88 | 5.11 | +4.6% |
| `short_codegen_128` | 4 | 103.35 | 96.18 | -6.9% | 4.95 | 5.32 | +7.5% |
| `medium_architecture_256` | 1 | 26.23 | 25.07 | -4.4% | 9.76 | 10.21 | +4.6% |
| `medium_architecture_256` | 2 | 51.89 | 48.69 | -6.2% | 9.87 | 10.52 | +6.6% |
| `medium_architecture_256` | 4 | 102.98 | 93.28 | -9.4% | 9.94 | 10.98 | +10.4% |
| `long_cosmology_512` | 1 | 26.01 | 24.21 | -6.9% | 19.68 | 21.16 | +7.5% |
| `long_cosmology_512` | 2 | 51.76 | 47.04 | -9.1% | 19.79 | 21.77 | +10.0% |
| `long_cosmology_512` | 4 | 103.07 | 87.55 | -15.1% | 19.87 | 23.39 | +17.7% |
| `long_context_summary_256` | 1 | 25.54 | 23.12 | -9.5% | 10.03 | 11.07 | +10.5% |
| `long_context_summary_256` | 2 | 51.61 | 42.13 | -18.4% | 9.92 | 12.15 | +22.5% |
| `long_context_summary_256` | 4 | 102.38 | 71.04 | -30.6% | 10.00 | 14.41 | +44.1% |
| `extended_generation_768` | 1 | 25.93 | 23.64 | -8.8% | 29.61 | 32.49 | +9.7% |
| `extended_generation_768` | 2 | 51.74 | 44.68 | -13.6% | 29.69 | 34.43 | +16.0% |
| `extended_generation_768` | 4 | 102.09 | 82.82 | -18.9% | 30.09 | 37.09 | +23.3% |

Production readout: the CUDA 30B path is live and stable through `sock serve`,
but TMH on CUDA currently runs as an accounting layout rather than the physical
AMD/ROCm layout. In this 4090 matrix, TMH accounting is functionally correct but
slower than standard: mean completion throughput was 60.31 tok/s for standard
versus 53.30 tok/s for TMH (-11.6%), with suite wall clock +13.5%. Do not market
CUDA TMH as a throughput win from this run; the verified win is first-class
large-model CUDA support through the hermetic `gptq-marlin` build profile.

### CUDA TMH Accounting Probe: Qwen3-8B

This run isolates the CUDA TMH accounting behavior on a smaller dense model using
the same canonical `sock serve` path and the same prompt/concurrency suite shape
as the 30B run. The first TMH launch attempt was discarded because the previous
standard server still owned the GPU; the retained TMH artifact was rerun after
verifying the 4090 was at 0 MiB used and that `/v1/models` was served by the TMH
process.

| Field | Value |
| --- | --- |
| Model | `Qwen/Qwen3-8B` |
| Endpoint | `/v1/completions` |
| Runtime | RTX 4090, CUDA 13.0, `sock serve` |
| Build profile | `gptq-marlin` runtime already present from 30B validation |
| `max_model_len` | `1024` |
| `gpu_memory_utilization` | `0.80` |
| `max_num_batched_tokens` | `1024` |
| `max_num_seqs` | `4` |
| `enforce_eager` | `true` |
| Standard suite wall clock | 367.83 s |
| TMH accounting suite wall clock | 383.62 s |
| Suite wall delta | +4.29% |
| Mean completion throughput | 124.17 tok/s standard, 118.10 tok/s TMH |
| Geomean completion throughput delta | -3.51% |
| Raw summaries | `benchmarks/2026-07-19-rtx4090-qwen3-8b/` |

| Case | Concurrency | Standard completion tok/s | TMH completion tok/s | TMH delta | Standard wall s | TMH wall s | Wall delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tiny_fact_64` | 1 | 55.62 | 55.62 | +0.0% | 1.15 | 1.15 | -0.0% |
| `tiny_fact_64` | 2 | 105.68 | 106.10 | +0.4% | 1.21 | 1.21 | -0.4% |
| `tiny_fact_64` | 4 | 209.40 | 209.01 | -0.2% | 1.22 | 1.22 | +0.2% |
| `short_codegen_128` | 1 | 55.80 | 55.73 | -0.1% | 2.29 | 2.30 | +0.1% |
| `short_codegen_128` | 2 | 107.21 | 106.79 | -0.4% | 2.39 | 2.40 | +0.4% |
| `short_codegen_128` | 4 | 211.61 | 211.34 | -0.1% | 2.42 | 2.42 | +0.1% |
| `medium_architecture_256` | 1 | 55.68 | 55.67 | -0.0% | 4.60 | 4.60 | +0.0% |
| `medium_architecture_256` | 2 | 106.66 | 106.53 | -0.1% | 4.80 | 4.81 | +0.1% |
| `medium_architecture_256` | 4 | 211.63 | 211.53 | -0.0% | 4.84 | 4.84 | +0.0% |
| `long_cosmology_512` | 1 | 55.62 | 55.58 | -0.1% | 9.20 | 9.21 | +0.1% |
| `long_cosmology_512` | 2 | 106.66 | 106.61 | -0.0% | 9.60 | 9.60 | +0.0% |
| `long_cosmology_512` | 4 | 211.01 | 199.84 | -5.3% | 9.71 | 10.25 | +5.6% |
| `long_context_summary_256` | 1 | 55.23 | 55.21 | -0.0% | 4.63 | 4.64 | +0.0% |
| `long_context_summary_256` | 2 | 105.73 | 96.00 | -9.2% | 4.84 | 5.33 | +10.1% |
| `long_context_summary_256` | 4 | 209.52 | 148.84 | -29.0% | 4.89 | 6.88 | +40.8% |
| `extended_generation_768` | 1 | 55.52 | 55.52 | -0.0% | 13.83 | 13.83 | +0.0% |
| `extended_generation_768` | 2 | 106.41 | 104.92 | -1.4% | 14.43 | 14.64 | +1.4% |
| `extended_generation_768` | 4 | 210.15 | 185.00 | -12.0% | 14.62 | 16.61 | +13.6% |

Production readout: CUDA TMH accounting does not impose a broad fixed overhead on
the 8B eager path. It is effectively at parity for short prompts and concurrency
1/2, but it regresses under concurrency-4 pressure when the request mix is
prompt-heavy or long-context. This supports a narrower hypothesis than the 30B
result alone: the CUDA issue appears tied to high-concurrency accounting/cache
pressure rather than ordinary single-stream decode.

### CUDA TMH Accounting Probe: GB10 Qwen3-8B

This run repeats the same Qwen3-8B standard-vs-TMH endpoint suite on a GB10 CUDA
host. GB10 reports `sm121`, driver `595.71.05`, CUDA 13.2 at the driver layer,
and torch `2.11.0+cu130` inside the hermetic sock venv. The host's
`nvidia-smi` memory accounting reports memory as `N/A`, but vLLM observes 121.69
GiB total and only 38.14 GiB free inside the container at startup, so this run
uses `gpu_memory_utilization=0.25`.

| Field | Value |
| --- | --- |
| Model | `Qwen/Qwen3-8B` |
| Endpoint | `/v1/completions` |
| Runtime | NVIDIA GB10, CUDA, `sock serve` |
| Build profile | `gptq-marlin` |
| `max_model_len` | `1024` |
| `gpu_memory_utilization` | `0.25` |
| `max_num_batched_tokens` | `1024` |
| `max_num_seqs` | `4` |
| `enforce_eager` | `true` |
| Standard suite wall clock | 1414.71 s |
| TMH accounting suite wall clock | 1389.09 s |
| Suite wall delta | -1.81% |
| Mean completion throughput | 32.54 tok/s standard, 33.29 tok/s TMH |
| Geomean completion throughput delta | +1.87% |
| Raw summaries | `benchmarks/2026-07-19-gb10-qwen3-8b/` |

| Case | Concurrency | Standard completion tok/s | TMH completion tok/s | TMH delta | Standard wall s | TMH wall s | Wall delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tiny_fact_64` | 1 | 14.32 | 14.37 | +0.4% | 4.47 | 4.45 | -0.4% |
| `tiny_fact_64` | 2 | 27.98 | 28.71 | +2.6% | 4.58 | 4.46 | -2.6% |
| `tiny_fact_64` | 4 | 55.39 | 56.37 | +1.8% | 4.62 | 4.54 | -1.7% |
| `short_codegen_128` | 1 | 14.34 | 14.37 | +0.2% | 8.93 | 8.91 | -0.2% |
| `short_codegen_128` | 2 | 28.06 | 28.89 | +3.0% | 9.12 | 8.86 | -2.9% |
| `short_codegen_128` | 4 | 55.82 | 57.18 | +2.4% | 9.17 | 8.95 | -2.4% |
| `medium_architecture_256` | 1 | 14.28 | 14.31 | +0.2% | 17.93 | 17.90 | -0.2% |
| `medium_architecture_256` | 2 | 28.07 | 28.79 | +2.6% | 18.24 | 17.79 | -2.5% |
| `medium_architecture_256` | 4 | 55.79 | 57.24 | +2.6% | 18.35 | 17.89 | -2.5% |
| `long_cosmology_512` | 1 | 14.26 | 14.30 | +0.3% | 35.91 | 35.80 | -0.3% |
| `long_cosmology_512` | 2 | 28.01 | 28.80 | +2.8% | 36.56 | 35.56 | -2.7% |
| `long_cosmology_512` | 4 | 55.59 | 57.01 | +2.6% | 36.84 | 35.92 | -2.5% |
| `long_context_summary_256` | 1 | 14.19 | 14.24 | +0.4% | 18.05 | 17.98 | -0.4% |
| `long_context_summary_256` | 2 | 27.81 | 28.58 | +2.7% | 18.41 | 17.92 | -2.7% |
| `long_context_summary_256` | 4 | 55.35 | 56.70 | +2.4% | 18.50 | 18.06 | -2.4% |
| `extended_generation_768` | 1 | 14.26 | 14.29 | +0.2% | 53.84 | 53.76 | -0.2% |
| `extended_generation_768` | 2 | 27.97 | 28.75 | +2.8% | 54.91 | 53.43 | -2.7% |
| `extended_generation_768` | 4 | 54.21 | 56.36 | +4.0% | 56.73 | 54.51 | -3.9% |

Production readout: the 4090 8B regression does not reproduce on GB10. Under
the same prompt/concurrency matrix, GB10 TMH is slightly faster than standard in
every measured concurrency bucket, including the C4 long-context and
extended-generation cases that regressed on the 4090. That narrows the CUDA TMH
accounting bug from "CUDA-wide overhead" to an architecture/profile-sensitive
interaction, likely involving the 4090 `sm89` eager scheduling/cache path rather
than the abstract sock CLI or benchmark harness.

### CUDA TMH Accounting Fix Validation

Commit `c4c3788` removes the false hot-path shape from TMH accounting. The
accounting path now computes pressure from page spans instead of walking every
layer/page pair during scheduler allocation, caches live-block byte accounting by
request, and clears request accounting state on KV free. It also makes explicit
`tmh_kv_policy=physical` fail closed before layout normalization so physical TMH
cannot be silently downgraded to standard.

| Host | Standard wall s | TMH wall s | Wall delta | Standard geomean tok/s | TMH geomean tok/s | Geomean delta | Raw summaries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RTX 4090 before fix | 367.83 | 383.62 | +4.29% | 107.58 | 103.81 | -3.51% | `benchmarks/2026-07-19-rtx4090-qwen3-8b/` |
| RTX 4090 after fix | 368.20 | 368.13 | -0.02% | 107.48 | 107.48 | +0.00% | `benchmarks/2026-07-19-rtx4090-qwen3-8b-tmh-fix/` |
| GB10 before fix | 1414.71 | 1389.09 | -1.81% | 28.07 | 28.59 | +1.87% | `benchmarks/2026-07-19-gb10-qwen3-8b/` |
| GB10 after fix | 1405.59 | 1388.06 | -1.25% | 28.24 | 28.63 | +1.37% | `benchmarks/2026-07-19-gb10-qwen3-8b-tmh-fix/` |

| Host | Case | Concurrency | Before TMH delta | After TMH delta |
| --- | --- | ---: | ---: | ---: |
| RTX 4090 | `long_context_summary_256` | 4 | -29.0% | -0.1% |
| RTX 4090 | `extended_generation_768` | 4 | -12.0% | +0.2% |
| RTX 4090 | `long_cosmology_512` | 4 | -5.3% | +0.0% |
| GB10 | `long_context_summary_256` | 4 | +2.4% | +2.1% |
| GB10 | `extended_generation_768` | 4 | +4.0% | +0.8% |
| GB10 | `long_cosmology_512` | 4 | +2.6% | +2.0% |

Production readout: the RTX 4090 regression was not a CUDA attention-kernel
failure and not a real TMH memory-layout effect. It was CPU-side Python
accounting placed inside the scheduler allocation path. After the fix, the same
4090 suite returns to parity with standard while GB10 remains stable. At this
commit, physical TMH was still intentionally fail-closed; accounting mode became
production-safe for observability and benchmarking without imposing a throughput
tax.

### CUDA Physical TMH FlashInfer Validation: RTX 4090 Qwen3-8B

Commit `3b8661c` validates the first physical TMH path on CUDA through the
standard FlashInfer attention backend. The run includes the production fixes
needed for live serving: CUDA runner TMH runtime initialization, decoder-only
FlashInfer warmup metadata, FlashInfer TMH cache update/attention routing,
canonical FlashInfer metadata exposure, and canonical/overlay physical-slot
reclamation across request lifetimes.

| Field | Value |
| --- | --- |
| Model | `Qwen/Qwen3-8B` |
| Endpoint | `/v1/completions` |
| Runtime | RTX 4090, CUDA 13.0, `sock serve` |
| Attention backend | `FLASHINFER` with physical TMH branch |
| `max_model_len` | `1024` |
| `gpu_memory_utilization` | `0.80` |
| `max_num_batched_tokens` | `1024` |
| `max_num_seqs` | `4` |
| `enforce_eager` | `true` |
| Standard suite wall clock | 368.40 s |
| Physical TMH suite wall clock | 627.94 s |
| Suite wall delta | +70.45% |
| Mean completion throughput | 123.95 tok/s standard, 77.82 tok/s physical TMH |
| Mean completion throughput delta | -37.21% |
| Raw summaries | `benchmarks/2026-07-19-rtx4090-qwen3-8b-standard-backend-tmh/` |

| Case | Concurrency | Standard completion tok/s | Physical TMH completion tok/s | TMH delta | Standard wall s | TMH wall s | Wall delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tiny_fact_64` | 1 | 55.63 | 53.63 | -3.6% | 1.15 | 1.19 | +3.7% |
| `tiny_fact_64` | 2 | 105.27 | 98.17 | -6.7% | 1.22 | 1.30 | +7.2% |
| `tiny_fact_64` | 4 | 208.49 | 172.38 | -17.3% | 1.23 | 1.49 | +21.0% |
| `short_codegen_128` | 1 | 55.72 | 50.50 | -9.4% | 2.30 | 2.53 | +10.3% |
| `short_codegen_128` | 2 | 106.99 | 94.27 | -11.9% | 2.39 | 2.72 | +13.5% |
| `short_codegen_128` | 4 | 211.16 | 146.70 | -30.5% | 2.42 | 3.30 | +35.9% |
| `medium_architecture_256` | 1 | 55.64 | 51.56 | -7.3% | 4.60 | 4.96 | +7.9% |
| `medium_architecture_256` | 2 | 106.60 | 86.86 | -18.5% | 4.80 | 5.89 | +22.7% |
| `medium_architecture_256` | 4 | 211.32 | 127.07 | -39.9% | 4.85 | 8.06 | +66.3% |
| `long_cosmology_512` | 1 | 55.50 | 47.28 | -14.8% | 9.22 | 10.83 | +17.4% |
| `long_cosmology_512` | 2 | 106.45 | 73.17 | -31.3% | 9.62 | 13.99 | +45.5% |
| `long_cosmology_512` | 4 | 210.85 | 90.45 | -57.1% | 9.71 | 19.46 | +100.4% |
| `long_context_summary_256` | 1 | 55.15 | 35.18 | -36.2% | 4.64 | 7.28 | +56.8% |
| `long_context_summary_256` | 2 | 105.50 | 46.19 | -56.2% | 4.85 | 11.08 | +128.4% |
| `long_context_summary_256` | 4 | 209.17 | 50.75 | -75.7% | 4.90 | 20.18 | +312.1% |
| `extended_generation_768` | 1 | 55.45 | 43.10 | -22.3% | 13.85 | 17.82 | +28.7% |
| `extended_generation_768` | 2 | 106.17 | 57.79 | -45.6% | 14.47 | 21.80 | +50.7% |
| `extended_generation_768` | 4 | 209.96 | 75.73 | -63.9% | 14.63 | 40.57 | +177.3% |

Production readout: physical TMH on CUDA is now functionally real, not a
placeholder or accounting-only path. It starts, survives the full six-case
concurrency matrix, reclaims physical slots across request lifetimes, and serves
OpenAI-compatible completions through FlashInfer. It is not production-ready as
a CUDA performance feature yet. The current physical kernel path regresses
throughput sharply on RTX 4090, especially at concurrency 4 and prompt-heavy or
long-context cases. The next CUDA TMH optimization target is kernel efficiency
and warmup shape coverage, not allocator correctness or endpoint bring-up.

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
`--kv-layout tmh` is wired into live vendored-vLLM `KVCacheManager` allocation
and records TMH pressure during real traffic. Physical TMH currently fails
closed until mixed-fidelity warm-page tensors and attention kernels are
implemented.

| Field | Value |
| --- | --- |
| Model | `Qwen/Qwen3-30B-A3B-GPTQ-Int4` |
| Serve path | `sock serve` OpenAI-compatible endpoint |
| Serve profile | `max_model_len=2048`, `max_num_seqs=4`, `max_num_batched_tokens=1024`, `gpu_memory_utilization=0.35`, `enforce_eager=true` |
| Suite shape | 10 smoke pressure cases, concurrency 1/2/4, 1 warmup batch, 1 measured batch, 10 streaming TTFT probes |
| TMH layout | `--kv-layout tmh --tmh-hot-budget-pct 25` |

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

Readout: this older run proved the allocator/policy path without imposing a
large scheduler tax. It is retained as the pre-physical baseline; the physical
mixed-fidelity layout and attention-kernel path is now benchmarked separately
below.

## Physical TMH Runtime: Qwen3-30B-A3B MoE GPTQ Int4

Commit `e007cb4` wires physical TMH kernel materialization into startup through
the production scheduler path. The warmup now covers both cold single-request
serving and max safe batch serving for the allocated TMH KV cache, so endpoint
traffic no longer triggers TMH Triton JIT compilation. Commit `2c322d5` then
optimizes the physical attention path by splitting all-raw, all-warm, and mixed
tiles, aligning TMH attention tiles with the 16-token physical page, and reusing
the GPU request-row map across layers instead of rebuilding it per attention
call. The next kernel pass specializes the physical attention loop around the
page-aligned descriptor invariant: each 16-token tile maps to one physical TMH
page, so the kernel loads role/slot metadata once per tile instead of classifying
every token lane. These runs compare regular KV against physical TMH on the same
six-case endpoint suite.

| Field | Value |
| --- | --- |
| Model | `Qwen/Qwen3-30B-A3B-GPTQ-Int4` |
| Serve path | `sock serve` OpenAI-compatible endpoint |
| Serve profile | `max_model_len=2048`, `max_num_seqs=4`, `max_num_batched_tokens=1024`, `gpu_memory_utilization=0.35`, `enforce_eager=true` |
| Suite shape | 6 prompt classes, concurrency 1/2/4, 1 warmup batch, 2 measured batches |
| TMH layout | `--kv-layout tmh --tmh-hot-budget-pct 25` |
| Standard suite wall clock | 765.69 s |
| Physical TMH suite wall clock, first physical kernel | 1075.15 s |
| Physical TMH suite wall clock, optimized kernel | 945.04 s |
| Physical TMH suite wall clock, page-descriptor kernel | 935.04 s |
| Standard ready time | 73 s |
| Physical TMH ready time, first physical kernel | 65 s |
| Physical TMH ready time, optimized kernel | 75 s |
| Physical TMH ready time, page-descriptor kernel | 76 s |
| Standard request-time JIT warnings | 3, non-TMH kernels (`_fwd_kernel`, MoE/GPTQ) |
| Physical TMH request-time JIT warnings, first physical kernel | 2, MoE/GPTQ kernels; zero TMH kernel JIT warnings |
| Physical TMH request-time JIT warnings, optimized kernel | 2, MoE/GPTQ kernels; zero TMH kernel JIT warnings |
| Physical TMH request-time JIT warnings, page-descriptor kernel | 2, MoE/GPTQ kernels; zero TMH kernel JIT warnings |
| Standard geomean completion throughput | 36.70 tok/s |
| Physical TMH geomean completion throughput, first physical kernel | 26.44 tok/s |
| Physical TMH geomean completion throughput, optimized kernel | 29.76 tok/s |
| Physical TMH geomean completion throughput, page-descriptor kernel | 29.98 tok/s |
| First physical TMH geomean delta vs standard | -27.96% |
| Optimized physical TMH geomean delta vs first physical TMH | +12.56% |
| Optimized physical TMH geomean delta vs standard | -18.92% |
| Page-descriptor physical TMH geomean delta vs optimized physical TMH | +0.73% |
| Page-descriptor physical TMH geomean delta vs standard | -18.33% |
| Raw summaries | `benchmarks/2026-07-19-gmk-qwen3-30b-physical-tmh/`, `benchmarks/2026-07-19-gmk-qwen3-30b-physical-tmh-kernel-opt/`, `benchmarks/2026-07-19-gmk-qwen3-30b-physical-tmh-page-desc-opt/` |

| Runtime | Suite wall s | Ready s | Geomean completion tok/s | Delta vs standard | TMH JIT warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard KV baseline | 765.69 | 73 | 36.70 | baseline | n/a |
| Physical TMH, first physical kernel | 1075.15 | 65 | 26.44 | -27.96% | 0 |
| Physical TMH, optimized kernel | 945.04 | 75 | 29.76 | -18.92% | 0 |
| Physical TMH, page-descriptor kernel | 935.04 | 76 | 29.98 | -18.33% | 0 |

| Case | Concurrency | Standard completion tok/s | Physical TMH completion tok/s | TMH delta | Standard wall s | TMH wall s | Wall delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tiny_fact_64` | 1 | 28.23 | 24.15 | -14.4% | 2.27 | 2.66 | +17.3% |
| `tiny_fact_64` | 2 | 32.07 | 26.07 | -18.7% | 3.99 | 4.91 | +22.9% |
| `tiny_fact_64` | 4 | 49.88 | 38.80 | -22.2% | 5.13 | 6.60 | +28.6% |
| `short_codegen_128` | 1 | 28.10 | 22.93 | -18.4% | 4.56 | 5.58 | +22.6% |
| `short_codegen_128` | 2 | 39.30 | 28.41 | -27.7% | 6.52 | 9.01 | +38.2% |
| `short_codegen_128` | 4 | 48.56 | 36.68 | -24.5% | 10.54 | 13.96 | +32.4% |
| `medium_architecture_256` | 1 | 28.93 | 21.34 | -26.2% | 8.85 | 12.00 | +35.6% |
| `medium_architecture_256` | 2 | 32.04 | 26.82 | -16.3% | 15.99 | 19.11 | +19.5% |
| `medium_architecture_256` | 4 | 49.46 | 37.03 | -25.1% | 20.71 | 27.65 | +33.6% |
| `long_cosmology_512` | 1 | 28.87 | 20.41 | -29.3% | 17.73 | 25.09 | +41.5% |
| `long_cosmology_512` | 2 | 32.66 | 25.56 | -21.7% | 31.37 | 40.06 | +27.7% |
| `long_cosmology_512` | 4 | 48.52 | 35.76 | -26.3% | 42.21 | 57.31 | +35.8% |
| `long_context_summary_256` | 1 | 28.29 | 16.58 | -41.4% | 9.05 | 15.44 | +70.7% |
| `long_context_summary_256` | 2 | 34.88 | 20.87 | -40.2% | 14.68 | 24.54 | +67.2% |
| `long_context_summary_256` | 4 | 67.26 | 28.67 | -57.4% | 15.22 | 35.72 | +134.6% |
| `extended_generation_768` | 1 | 28.48 | 19.89 | -30.2% | 26.96 | 38.61 | +43.2% |
| `extended_generation_768` | 2 | 30.91 | 24.88 | -19.5% | 49.70 | 61.75 | +24.3% |
| `extended_generation_768` | 4 | 49.23 | 35.18 | -28.5% | 62.45 | 87.31 | +39.8% |

Optimized physical TMH kernel row-level throughput:

| Case | Concurrency | Standard tok/s | First physical TMH tok/s | Optimized physical TMH tok/s | Optimized vs first TMH | Optimized vs standard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `tiny_fact_64` | 1 | 28.23 | 24.15 | 25.97 | +7.5% | -8.0% |
| `tiny_fact_64` | 2 | 32.07 | 26.07 | 28.15 | +8.0% | -12.2% |
| `tiny_fact_64` | 4 | 49.88 | 38.80 | 42.09 | +8.5% | -15.6% |
| `short_codegen_128` | 1 | 28.10 | 22.93 | 24.94 | +8.8% | -11.2% |
| `short_codegen_128` | 2 | 39.30 | 28.41 | 30.97 | +9.0% | -21.2% |
| `short_codegen_128` | 4 | 48.56 | 36.68 | 41.71 | +13.7% | -14.1% |
| `medium_architecture_256` | 1 | 28.93 | 21.34 | 24.35 | +14.1% | -15.8% |
| `medium_architecture_256` | 2 | 32.04 | 26.82 | 29.36 | +9.5% | -8.4% |
| `medium_architecture_256` | 4 | 49.46 | 37.03 | 42.07 | +13.6% | -14.9% |
| `long_cosmology_512` | 1 | 28.87 | 20.41 | 23.35 | +14.4% | -19.1% |
| `long_cosmology_512` | 2 | 32.66 | 25.56 | 28.41 | +11.1% | -13.0% |
| `long_cosmology_512` | 4 | 48.52 | 35.76 | 40.57 | +13.5% | -16.4% |
| `long_context_summary_256` | 1 | 28.29 | 16.58 | 19.24 | +16.1% | -32.0% |
| `long_context_summary_256` | 2 | 34.88 | 20.87 | 24.13 | +15.6% | -30.8% |
| `long_context_summary_256` | 4 | 67.26 | 28.67 | 35.07 | +22.3% | -47.9% |
| `extended_generation_768` | 1 | 28.48 | 19.89 | 22.24 | +11.8% | -21.9% |
| `extended_generation_768` | 2 | 30.91 | 24.88 | 28.33 | +13.9% | -8.3% |
| `extended_generation_768` | 4 | 49.23 | 35.18 | 40.70 | +15.7% | -17.3% |

Page-descriptor physical TMH kernel row-level throughput:

| Case | Concurrency | Standard tok/s | Optimized physical TMH tok/s | Page-descriptor physical TMH tok/s | Page-desc vs optimized TMH | Page-desc vs standard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `tiny_fact_64` | 1 | 28.23 | 25.97 | 25.42 | -2.1% | -10.0% |
| `tiny_fact_64` | 2 | 32.07 | 28.15 | 27.83 | -1.2% | -13.2% |
| `tiny_fact_64` | 4 | 49.88 | 42.09 | 42.04 | -0.1% | -15.7% |
| `short_codegen_128` | 1 | 28.10 | 24.94 | 25.02 | +0.3% | -11.0% |
| `short_codegen_128` | 2 | 39.30 | 30.97 | 31.22 | +0.8% | -20.6% |
| `short_codegen_128` | 4 | 48.56 | 41.71 | 40.42 | -3.1% | -16.8% |
| `medium_architecture_256` | 1 | 28.93 | 24.35 | 24.76 | +1.7% | -14.4% |
| `medium_architecture_256` | 2 | 32.04 | 29.36 | 29.75 | +1.3% | -7.1% |
| `medium_architecture_256` | 4 | 49.46 | 42.07 | 42.59 | +1.2% | -13.9% |
| `long_cosmology_512` | 1 | 28.87 | 23.35 | 23.62 | +1.2% | -18.2% |
| `long_cosmology_512` | 2 | 32.66 | 28.41 | 28.61 | +0.7% | -12.4% |
| `long_cosmology_512` | 4 | 48.52 | 40.57 | 40.48 | -0.2% | -16.6% |
| `long_context_summary_256` | 1 | 28.29 | 19.24 | 20.24 | +5.2% | -28.4% |
| `long_context_summary_256` | 2 | 34.88 | 24.13 | 24.92 | +3.3% | -28.6% |
| `long_context_summary_256` | 4 | 67.26 | 35.07 | 35.31 | +0.7% | -47.5% |
| `extended_generation_768` | 1 | 28.48 | 22.24 | 22.63 | +1.8% | -20.6% |
| `extended_generation_768` | 2 | 30.91 | 28.33 | 28.52 | +0.7% | -7.7% |
| `extended_generation_768` | 4 | 49.23 | 40.70 | 41.23 | +1.3% | -16.2% |

Production readout: physical TMH is correct enough to serve the full 30B suite,
and the cold-start TMH JIT issue is fixed. The first optimized physical kernel
improves geomean completion throughput by +12.56% over the first physical
implementation and cuts suite wall clock by 12.10%. The page-descriptor pass is
smaller but still production-positive: +0.73% geomean over the prior optimized
kernel and another 10.00 s off suite wall clock. This is real progress, but not
yet a throughput win: page-descriptor physical TMH remains -18.33% behind
standard KV on the full endpoint suite. The next target is deeper physical
attention algorithm and kernel efficiency, not scheduler accounting, CLI wiring,
or warmup coverage.

### RTX 4090 TMH kernel optimization slice

Follow-up CUDA kernel work used the RTX 4090 Qwen3-8B focused slice that exposed
the largest physical TMH decode regressions: `tiny_fact_64`,
`long_context_summary_256`, and `extended_generation_768` at concurrency 1, 2,
and 4. The baseline for this slice is the physical TMH run in
`benchmarks/2026-07-19-rtx4090-qwen3-8b-standard-backend-tmh/tmh-suite.json`.

| Run | Attention backend | Production decision | Mean completion tok/s delta vs prior physical TMH slice | Notes |
| --- | --- | --- | ---: | --- |
| CUDA tile-shape pass (`fb04dd4`) | FlashInfer metadata, TMH physical 2D fallback | kept | +0.85% | Small net win; improves `extended_generation_768` c2 by +10.99%, mixed elsewhere. |
| Packed-V split accumulator (`7632647`) | FlashInfer metadata, TMH physical 2D fallback | reverted | n/a | Correct but slower overall; removed from production code. |
| Segmented decode (`a4b6134`) | Triton attention metadata, TMH physical 3D segment/reduce | gated to long contexts | -4.53% at 1k context | Correct and raises sampled GPU utilization to 83%, but segment/reduce overhead is not amortized at `max_model_len=1024`. |

Production readout: keep the CUDA tile-shape tuning and keep segmented decode
implemented, tested, and gated for longer contexts (`max_seq_len >= 1025`) where
additional sequence-parallelism can amortize the segment/reduce overhead. Do not
enable segmented Triton TMH for the current 1k-context endpoint path.

## Artifacts

| Artifact | Purpose |
| --- | --- |
| `benchmarks/2026-07-18-gmk-qwen3-4b/summary.json` | Qwen3-4B compact sock/upstream summary |
| `benchmarks/2026-07-18-gmk-qwen3-4b/suite-summary.json` | Qwen3-4B compact suite comparison |
| `benchmarks/2026-07-18-gmk-qwen3-32b-2bit-gptq/suite-summary.json` | Qwen3-32B 2-bit compact suite summary |
| `benchmarks/2026-07-18-gmk-qwen3-32b-4bit-gptq/suite-summary.json` | Qwen3-32B 4-bit compact suite summary |
| `benchmarks/2026-07-18-gmk-qwen3-30b-a3b-gptq-int4/suite-summary.json` | Qwen3-30B-A3B MoE compact suite summary |
| `benchmarks/2026-07-19-rtx4090-cuda-qwen3/summary.json` | RTX 4090 CUDA Qwen3-4B/8B eager vs compiled summary |
| `benchmarks/2026-07-19-rtx4090-qwen3-8b-tmh-kernel-fb04dd4/tmh-suite.json` | RTX 4090 focused TMH CUDA tile-shape optimization slice |
| `benchmarks/2026-07-19-rtx4090-qwen3-8b-tmh-kernel-7632647/tmh-suite.json` | RTX 4090 focused packed-V accumulator diagnostic slice |
| `benchmarks/2026-07-20-rtx4090-qwen3-8b-tmh-triton-segmented-a4b6134/tmh-suite.json` | RTX 4090 focused Triton segmented decode diagnostic slice |
| `benchmarks/2026-07-19-gmk-qwen3-30b-physical-tmh/suite-summary.json` | Qwen3-30B-A3B standard vs physical TMH full endpoint suite |
| `benchmarks/2026-07-19-gmk-qwen3-30b-physical-tmh-kernel-opt/suite-summary.json` | Qwen3-30B-A3B optimized physical TMH kernel full endpoint suite |
| `benchmarks/2026-07-19-gmk-qwen3-30b-physical-tmh-page-desc-opt/suite-summary.json` | Qwen3-30B-A3B page-descriptor physical TMH kernel full endpoint suite |
| `artifacts/tmh_runtime_integration/REPORT.md` | Matched regular vs TMH allocator-path endpoint comparison |
| `artifacts/tmh_runtime_integration/summary.json` | Machine-readable TMH runtime integration summary |
| `artifacts/tmh_runtime_integration/logs/tmh_accounting_server.log` | TMH allocation-pressure log with 14,016 live allocator records |
| `tmp/bench-suite-sock-fixed-qwen3-4b.json` | Raw sock Qwen3-4B endpoint suite |
| `tmp/bench-suite-upstream-rocm-wheel-qwen3-4b.json` | Raw upstream vLLM ROCm Qwen3-4B endpoint suite |
| `tmp/bench-suite-sock-qwen3-32b-2bit-gptq-full.json` | Raw sock Qwen3-32B 2-bit suite |
| `tmp/bench-suite-sock-qwen3-32b-4bit-gptq-full.json` | Raw sock Qwen3-32B 4-bit suite |
| `tmp/bench-suite-sock-qwen3-30b-a3b-gptq-int4-small.json` | Raw sock Qwen3-30B-A3B MoE suite |
