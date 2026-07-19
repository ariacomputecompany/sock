"""CUDA build and wheel compatibility contracts."""

from __future__ import annotations

from dataclasses import dataclass

from sock_cuda_shim.capabilities import ComputeCapability, InvalidCudaConfiguration
from sock_cuda_shim.device import CudaDevice
from sock_cuda_shim.environment import CudaEnvironment


@dataclass(frozen=True)
class CudaBuildContract:
    cuda_version: str
    torch_cuda_version: str
    python_abi: str
    compiled_arches: tuple[ComputeCapability, ...]
    uses_libtorch_stable_abi: bool = True
    has_flashinfer: bool = True
    has_cutlass: bool = True
    has_flash_attention: bool = True

    def validate_for(self, device: CudaDevice, env: CudaEnvironment) -> None:
        if _major(self.cuda_version) != _major(self.torch_cuda_version):
            raise InvalidCudaConfiguration(
                f"CUDA toolkit {self.cuda_version} and torch CUDA {self.torch_cuda_version} are incompatible"
            )
        if device.capability not in self.compiled_arches:
            raise InvalidCudaConfiguration(
                f"wheel lacks exact compiled CUDA arch {device.capability.arch}; "
                f"compiled arches are {self.arch_list}"
            )
        for arch_text in env.torch_cuda_arch_list:
            arch = ComputeCapability.parse(arch_text.replace("+PTX", ""))
            if arch not in self.compiled_arches:
                raise InvalidCudaConfiguration(
                    f"TORCH_CUDA_ARCH_LIST requests {arch.arch}, but wheel compiled {self.arch_list}"
                )
        if device.capability.supports("fp8") and not self.has_flashinfer:
            raise InvalidCudaConfiguration("Ada/Hopper FP8 runtime requires FlashInfer or equivalent kernels")
        if device.capability.supports("tma") and not self.has_cutlass:
            raise InvalidCudaConfiguration("Hopper/Blackwell TMA paths require CUTLASS-family kernels")

    @property
    def arch_list(self) -> tuple[str, ...]:
        return tuple(arch.arch for arch in self.compiled_arches)


def _major(version: str) -> int:
    return int(version.split(".", 1)[0])
