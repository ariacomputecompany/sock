"""NVIDIA device and compute-capability contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from sock_cuda_shim.capabilities import (
    ComputeCapability,
    InvalidCudaConfiguration,
    UnsupportedCudaFeature,
    gib,
)


class DeviceClass(str, Enum):
    CONSUMER = "consumer"
    DATACENTER = "datacenter"
    MIG_SLICE = "mig_slice"


@dataclass(frozen=True)
class CudaDevice:
    name: str
    capability: ComputeCapability
    total_memory_bytes: int
    device_class: DeviceClass
    ordinal: int = 0
    uuid: str | None = None
    mig_parent_uuid: str | None = None
    pci_bus_id: str | None = None
    clock_locked: bool = False
    ecc_enabled: bool = False
    nvlink_peers: tuple[int, ...] = field(default_factory=tuple)
    numa_node: int | None = None

    def __post_init__(self) -> None:
        if self.total_memory_bytes <= 0:
            raise InvalidCudaConfiguration("device memory must be positive")
        if self.device_class is DeviceClass.MIG_SLICE and not self.mig_parent_uuid:
            raise InvalidCudaConfiguration("MIG slices must include a parent UUID")
        if self.capability.sm < 80:
            raise UnsupportedCudaFeature("sock CUDA inference requires sm80+")

    @classmethod
    def rtx_4090(cls, total_gib: float = 24.0, ordinal: int = 0) -> "CudaDevice":
        return cls(
            name="NVIDIA GeForce RTX 4090",
            capability=ComputeCapability(8, 9),
            total_memory_bytes=gib(total_gib),
            device_class=DeviceClass.CONSUMER,
            ordinal=ordinal,
            pci_bus_id=f"0000:{ordinal:02x}:00.0",
        )

    @classmethod
    def h100(cls, total_gib: float = 80.0, ordinal: int = 0) -> "CudaDevice":
        return cls(
            name="NVIDIA H100",
            capability=ComputeCapability(9, 0),
            total_memory_bytes=gib(total_gib),
            device_class=DeviceClass.DATACENTER,
            ordinal=ordinal,
            uuid=f"GPU-H100-{ordinal}",
            pci_bus_id=f"0000:{ordinal:02x}:00.0",
            ecc_enabled=True,
        )

    @classmethod
    def b200(cls, total_gib: float = 180.0, ordinal: int = 0) -> "CudaDevice":
        return cls(
            name="NVIDIA B200",
            capability=ComputeCapability(10, 0),
            total_memory_bytes=gib(total_gib),
            device_class=DeviceClass.DATACENTER,
            ordinal=ordinal,
            uuid=f"GPU-B200-{ordinal}",
            pci_bus_id=f"0000:{ordinal:02x}:00.0",
            ecc_enabled=True,
        )

    @classmethod
    def blackwell_sm120(cls, total_gib: float = 180.0, ordinal: int = 0) -> "CudaDevice":
        return cls(
            name="NVIDIA Blackwell SM120",
            capability=ComputeCapability(12, 0),
            total_memory_bytes=gib(total_gib),
            device_class=DeviceClass.DATACENTER,
            ordinal=ordinal,
            uuid=f"GPU-SM120-{ordinal}",
            pci_bus_id=f"0000:{ordinal:02x}:00.0",
            ecc_enabled=True,
        )

    def require(self, feature: str) -> None:
        if not self.capability.supports(feature):
            raise UnsupportedCudaFeature(
                f"{feature} requires newer NVIDIA hardware than {self.capability.arch}"
            )

    @property
    def visible_name(self) -> str:
        suffix = f" MIG({self.mig_parent_uuid})" if self.device_class is DeviceClass.MIG_SLICE else ""
        return f"{self.ordinal}:{self.name}:{self.capability.arch}{suffix}"

    @property
    def supports_cuda_graphs(self) -> bool:
        return self.capability.supports("cuda_graphs")

    @property
    def supports_fp8(self) -> bool:
        return self.capability.supports("fp8")

    @property
    def supports_nvfp4(self) -> bool:
        return self.capability.supports("nvfp4")

    @property
    def supports_tma(self) -> bool:
        return self.capability.supports("tma")

    def memory_budget(self, utilization: float, reserve_bytes: int = 0) -> int:
        if not (0 < utilization <= 1):
            raise InvalidCudaConfiguration("gpu_memory_utilization must be in (0, 1]")
        budget = int(self.total_memory_bytes * utilization) - reserve_bytes
        if budget <= 0:
            raise InvalidCudaConfiguration("memory reserve exceeds usable device memory")
        return budget
