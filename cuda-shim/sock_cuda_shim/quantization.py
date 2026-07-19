"""CUDA quantization feature gates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sock_cuda_shim.capabilities import InvalidCudaConfiguration
from sock_cuda_shim.device import CudaDevice
from sock_cuda_shim.kv_cache import Precision


class QuantBackend(str, Enum):
    GPTQ = "gptq"
    AWQ = "awq"
    FP8 = "fp8"
    NVFP4 = "nvfp4"
    MXFP4 = "mxfp4"


@dataclass(frozen=True)
class QuantizationPlan:
    weight_backend: QuantBackend
    kv_precision: Precision
    activation_fp8: bool = False
    per_token_scale: bool = False

    def validate(self, device: CudaDevice) -> None:
        if self.weight_backend is QuantBackend.FP8 or self.activation_fp8:
            device.require("fp8")
        if self.weight_backend in {QuantBackend.NVFP4, QuantBackend.MXFP4}:
            device.require("nvfp4")
        if self.kv_precision is Precision.NVFP4:
            device.require("nvfp4")
        if self.kv_precision is Precision.FP8:
            device.require("fp8")
        if self.per_token_scale and self.weight_backend not in {
            QuantBackend.FP8,
            QuantBackend.NVFP4,
            QuantBackend.MXFP4,
        }:
            raise InvalidCudaConfiguration("per-token scaling is only modeled for FP8/FP4 paths")
