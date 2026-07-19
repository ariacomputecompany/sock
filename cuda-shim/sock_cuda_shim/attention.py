"""CUDA attention backend selection contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sock_cuda_shim.capabilities import InvalidCudaConfiguration
from sock_cuda_shim.device import CudaDevice
from sock_cuda_shim.environment import CudaEnvironment
from sock_cuda_shim.kv_cache import KVLayout, Precision


class AttentionBackend(str, Enum):
    FLASHINFER = "FLASHINFER"
    FLASH_ATTN = "FLASH_ATTN"
    TRITON_ATTN = "TRITON_ATTN"
    CUTLASS_MLA = "CUTLASS_MLA"
    TORCH_SDPA = "TORCH_SDPA"


@dataclass(frozen=True)
class AttentionRequestShape:
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    max_model_len: int
    kv_layout: KVLayout
    kv_precision: Precision
    is_mla: bool = False
    sliding_window: int | None = None

    def validate(self) -> None:
        if self.num_query_heads <= 0 or self.num_kv_heads <= 0:
            raise InvalidCudaConfiguration("attention heads must be positive")
        if self.num_query_heads % self.num_kv_heads != 0:
            raise InvalidCudaConfiguration("GQA requires query heads divisible by KV heads")
        if self.head_size not in {64, 80, 96, 112, 120, 128, 160, 192, 256, 512}:
            raise InvalidCudaConfiguration(f"unsupported CUDA attention head size {self.head_size}")
        if self.max_model_len <= 0:
            raise InvalidCudaConfiguration("max_model_len must be positive")
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise InvalidCudaConfiguration("sliding_window must be positive")


def select_attention_backend(
    device: CudaDevice,
    env: CudaEnvironment,
    shape: AttentionRequestShape,
) -> AttentionBackend:
    shape.validate()
    requested = env.vllm_attention_backend.upper() if env.vllm_attention_backend else None
    if requested:
        backend = AttentionBackend(requested)
        _validate_backend(device, shape, backend)
        return backend
    if shape.is_mla and device.capability.supports("tma"):
        return AttentionBackend.CUTLASS_MLA
    if shape.kv_layout in {KVLayout.FLASHINFER_PAGED, KVLayout.TMH_FIDELITY_PAGED}:
        if device.capability.sm >= 89:
            return AttentionBackend.FLASHINFER
        return AttentionBackend.TRITON_ATTN
    if device.capability.sm >= 80:
        return AttentionBackend.FLASH_ATTN
    return AttentionBackend.TORCH_SDPA


def _validate_backend(
    device: CudaDevice,
    shape: AttentionRequestShape,
    backend: AttentionBackend,
) -> None:
    if backend is AttentionBackend.CUTLASS_MLA and not device.capability.supports("tma"):
        raise InvalidCudaConfiguration("CUTLASS MLA requires Hopper/Blackwell TMA support")
    if backend is AttentionBackend.FLASHINFER and shape.kv_precision is Precision.NVFP4:
        device.require("nvfp4")
    if backend is AttentionBackend.FLASH_ATTN and shape.kv_layout is KVLayout.TRTLLM_PAGED:
        raise InvalidCudaConfiguration("FlashAttention backend cannot consume TensorRT-LLM page tables directly")
