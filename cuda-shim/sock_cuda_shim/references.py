"""Source references that define the CUDA contract surface we model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceReference:
    concern: str
    repo: str
    path: str
    url: str
    note: str


REFERENCES: tuple[SourceReference, ...] = (
    SourceReference(
        concern="platform",
        repo="vllm-project/vllm",
        path="vllm/platforms/cuda.py",
        url="https://github.com/vllm-project/vllm/blob/main/vllm/platforms/cuda.py",
        note="CUDA availability, device capability, and environment normalization.",
    ),
    SourceReference(
        concern="paged_attention",
        repo="vllm-project/vllm",
        path="vllm/v1/attention/ops/paged_attn.py",
        url="https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/paged_attn.py",
        note="vLLM paged-attention call shape and metadata boundary.",
    ),
    SourceReference(
        concern="kv_cache_kernels",
        repo="vllm-project/vllm",
        path="csrc/libtorch_stable/cache_kernels.cu",
        url="https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/cache_kernels.cu",
        note="KV cache reshape/copy kernel boundary.",
    ),
    SourceReference(
        concern="nvfp4_kv_cache",
        repo="vllm-project/vllm",
        path="csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu",
        url="https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu",
        note="Blackwell-era low-bit KV cache gates.",
    ),
    SourceReference(
        concern="flashinfer_paged_kv",
        repo="flashinfer-ai/flashinfer",
        path="csrc/fmha_v2/fmha/paged_kv_cache.h",
        url="https://github.com/flashinfer-ai/flashinfer/blob/main/csrc/fmha_v2/fmha/paged_kv_cache.h",
        note="Paged KV table shape and kernel-facing metadata.",
    ),
    SourceReference(
        concern="cuda_virtual_memory",
        repo="NVIDIA/TensorRT-LLM",
        path="tensorrt_llm/runtime/kv_cache_manager_v2/_cuda_virt_mem.py",
        url="https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/runtime/kv_cache_manager_v2/_cuda_virt_mem.py",
        note="CUDA virtual memory allocation, mapping, rollback, and chunk constraints.",
    ),
    SourceReference(
        concern="kv_cache_manager",
        repo="NVIDIA/TensorRT-LLM",
        path="cpp/include/tensorrt_llm/batch_manager/kvCacheManager.h",
        url="https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/include/tensorrt_llm/batch_manager/kvCacheManager.h",
        note="Production KV manager lifecycle and allocation contracts.",
    ),
    SourceReference(
        concern="server_integration",
        repo="sgl-project/sglang",
        path="python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py",
        url="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py",
        note="Runtime attention backend selection and server setup gates.",
    ),
)
