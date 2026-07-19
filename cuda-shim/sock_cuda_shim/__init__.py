"""CUDA-shaped contract models for sock NVIDIA development."""

from sock_cuda_shim.device import CudaDevice, ComputeCapability, DeviceClass
from sock_cuda_shim.diagnostics import CudaReadinessReport, evaluate_readiness
from sock_cuda_shim.scenarios import CANONICAL_SCENARIOS, CudaScenario

__all__ = [
    "CANONICAL_SCENARIOS",
    "ComputeCapability",
    "CudaDevice",
    "CudaReadinessReport",
    "CudaScenario",
    "DeviceClass",
    "evaluate_readiness",
]
