"""CUDA graph capture and replay constraints."""

from __future__ import annotations

from dataclasses import dataclass, field

from sock_cuda_shim.capabilities import InvalidCudaConfiguration
from sock_cuda_shim.device import CudaDevice


CAPTURE_FORBIDDEN_OPS = {
    "cudaMalloc",
    "cudaFree",
    "cudaDeviceSynchronize",
    "host_io",
    "nccl_comm_init",
    "shape_alloc",
}


@dataclass(frozen=True)
class CudaGraphPlan:
    batch_size: int
    max_tokens: int
    static_input_shapes: bool = True
    captures_decode_only: bool = True
    forbidden_ops_seen: tuple[str, ...] = field(default_factory=tuple)

    def validate(self, device: CudaDevice) -> None:
        device.require("cuda_graphs")
        if self.batch_size <= 0 or self.max_tokens <= 0:
            raise InvalidCudaConfiguration("CUDA graph batch and token sizes must be positive")
        if not self.static_input_shapes:
            raise InvalidCudaConfiguration("CUDA graph replay requires static input shapes")
        if not self.captures_decode_only:
            raise InvalidCudaConfiguration("prefill capture must be separately proven before graph replay")
        forbidden = sorted(set(self.forbidden_ops_seen) & CAPTURE_FORBIDDEN_OPS)
        if forbidden:
            raise InvalidCudaConfiguration(f"CUDA graph capture saw forbidden ops: {forbidden}")

    @property
    def cache_key(self) -> tuple[int, int]:
        return (self.batch_size, self.max_tokens)
