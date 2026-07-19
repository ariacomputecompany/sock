"""CUDA virtual-memory and pool behavior."""

from __future__ import annotations

from dataclasses import dataclass, field

from sock_cuda_shim.capabilities import InvalidCudaConfiguration, round_up
from sock_cuda_shim.device import CudaDevice


CUDA_VMM_GRANULARITY = 2 * 1024 * 1024


@dataclass(frozen=True)
class VirtualAllocation:
    base: int
    size: int
    mapped: bool = True


@dataclass
class CudaMemoryPool:
    device: CudaDevice
    usable_bytes: int
    chunk_bytes: int = 64 * 1024 * 1024
    allocations: list[VirtualAllocation] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.usable_bytes > self.device.total_memory_bytes:
            raise InvalidCudaConfiguration("pool usable bytes exceed device memory")
        if self.chunk_bytes % CUDA_VMM_GRANULARITY != 0:
            raise InvalidCudaConfiguration("CUDA VMM chunks must respect allocation granularity")

    @property
    def allocated_bytes(self) -> int:
        return sum(allocation.size for allocation in self.allocations if allocation.mapped)

    @property
    def free_bytes(self) -> int:
        return self.usable_bytes - self.allocated_bytes

    def reserve(self, requested_bytes: int) -> VirtualAllocation:
        if requested_bytes <= 0:
            raise InvalidCudaConfiguration("allocation size must be positive")
        size = round_up(requested_bytes, self.chunk_bytes)
        if size > self.free_bytes:
            raise MemoryError(
                f"CUDA OOM: requested {size} bytes with {self.free_bytes} bytes free"
            )
        base = 0 if not self.allocations else max(a.base + a.size for a in self.allocations)
        allocation = VirtualAllocation(base=base, size=size)
        self.allocations.append(allocation)
        return allocation

    def unmap(self, allocation: VirtualAllocation) -> None:
        if allocation not in self.allocations:
            raise InvalidCudaConfiguration("allocation is not owned by this pool")
        index = self.allocations.index(allocation)
        self.allocations[index] = VirtualAllocation(
            base=allocation.base, size=allocation.size, mapped=False
        )

    def fragmentation_ratio(self) -> float:
        holes = sum(allocation.size for allocation in self.allocations if not allocation.mapped)
        return 0.0 if self.usable_bytes == 0 else holes / self.usable_bytes
