# CUDA Shim

`cuda-shim/` is sock's local NVIDIA/CUDA contract model. It is not a kernel
emulator and it does not pretend AMD hardware can execute CUDA. Instead, it
models the production decisions that real CUDA/NVIDIA runtimes force us to make
so sock can be developed against a CUDA-shaped target before final validation on
real NVIDIA hardware.

The shim is intentionally strict:

- invalid CUDA environment combinations fail closed;
- unsupported compute capabilities are explicit;
- CUDA graph capture rejects dynamic or synchronizing operations;
- NCCL/distributed plans validate topology and transport assumptions;
- KV cache and paged-attention metadata are validated like production serving
  code, not like loose unit-test fixtures;
- physical memory savings are modeled separately from accounting-only savings.

## Source Map

The contract shape is informed by current public runtime implementations:

| Concern | Reference |
| --- | --- |
| CUDA platform/device detection | `vllm-project/vllm:vllm/platforms/cuda.py` |
| vLLM paged attention and KV cache | `vllm-project/vllm:vllm/v1/attention/ops/paged_attn.py`, `csrc/libtorch_stable/cache_kernels.cu`, `csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu` |
| FlashInfer paged KV and attention kernels | `flashinfer-ai/flashinfer:benchmarks/bench_append_paged_kv_cache.py`, `csrc/fmha_v2/fmha/paged_kv_cache.h` |
| TensorRT-LLM KV manager and CUDA virtual memory | `NVIDIA/TensorRT-LLM:tensorrt_llm/runtime/kv_cache_manager_v2/_cuda_virt_mem.py`, `cpp/include/tensorrt_llm/batch_manager/kvCacheManager.h` |
| SGLang CUDA integration | `sgl-project/sglang:python/sglang/srt/platforms/cuda.py`, `python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py` |
| NVIDIA kernel architecture substrate | `NVIDIA/cutlass`, `Dao-AILab/flash-attention` |

The shim code is original sock test infrastructure. If we later vendor upstream
code, it should be done as an explicit dependency with license review rather
than by copying into this directory.

## Concern Files

| File | Concern |
| --- | --- |
| `sock_cuda_shim/device.py` | NVIDIA devices, compute capability, MIG, memory, feature gates |
| `sock_cuda_shim/environment.py` | CUDA/NVIDIA environment variables and invalid combinations |
| `sock_cuda_shim/build.py` | wheel/build compatibility, CUDA version, torch ABI, arch list |
| `sock_cuda_shim/memory.py` | CUDA virtual memory, pools, fragmentation, alignment |
| `sock_cuda_shim/kv_cache.py` | paged KV layout, slot mapping, TMH physical/accounting pressure |
| `sock_cuda_shim/attention.py` | attention backend dispatch and metadata requirements |
| `sock_cuda_shim/cuda_graphs.py` | CUDA graph capture/replay constraints |
| `sock_cuda_shim/distributed.py` | NCCL-like topology and transport validation |
| `sock_cuda_shim/quantization.py` | FP8/FP4/NVFP4/KV-quant feature gates |
| `sock_cuda_shim/diagnostics.py` | production readiness reports |
| `sock_cuda_shim/scenarios.py` | canonical adversarial NVIDIA scenarios |

## Run

```bash
cd /home/deepsaint/work/sock
python -m pytest cuda-shim/tests -q
python cuda-shim/run_matrix.py --json --inference-contract
```

Passing these tests does not prove CUDA kernel correctness. It proves sock is
threaded through the same CUDA-shaped constraints that we need to satisfy before
renting a 4090/H100/B200-class machine for live execution benchmarks.
