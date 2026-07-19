"""NCCL-like distributed CUDA topology contracts."""

from __future__ import annotations

from dataclasses import dataclass

from sock_cuda_shim.capabilities import InvalidCudaConfiguration
from sock_cuda_shim.device import CudaDevice
from sock_cuda_shim.environment import CudaEnvironment


@dataclass(frozen=True)
class DistributedPlan:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    requires_p2p: bool = True
    requires_ib: bool = False

    @property
    def world_size(self) -> int:
        return self.tensor_parallel_size * self.pipeline_parallel_size * self.data_parallel_size

    def validate(self, devices: tuple[CudaDevice, ...], env: CudaEnvironment) -> None:
        if self.world_size <= 0:
            raise InvalidCudaConfiguration("world size must be positive")
        if len(devices) < self.world_size:
            raise InvalidCudaConfiguration("not enough CUDA devices for distributed plan")
        if self.requires_p2p and env.nccl_p2p_disable:
            raise InvalidCudaConfiguration("plan requires P2P but NCCL_P2P_DISABLE is set")
        if self.requires_ib and env.nccl_ib_disable:
            raise InvalidCudaConfiguration("plan requires InfiniBand but NCCL_IB_DISABLE is set")
        selected = devices[: self.world_size]
        capabilities = {device.capability for device in selected}
        if len(capabilities) != 1:
            raise InvalidCudaConfiguration("mixed compute capability distributed runs need explicit kernels")
        if self.tensor_parallel_size > 1:
            for device in selected[: self.tensor_parallel_size]:
                peers = set(device.nvlink_peers)
                missing = {
                    other.ordinal
                    for other in selected[: self.tensor_parallel_size]
                    if other.ordinal != device.ordinal and other.ordinal not in peers
                }
                if missing and self.requires_p2p:
                    raise InvalidCudaConfiguration(
                        f"device {device.ordinal} lacks NVLink/P2P peers {sorted(missing)}"
                    )
