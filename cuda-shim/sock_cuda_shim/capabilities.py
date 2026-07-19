"""Shared CUDA capability primitives."""

from __future__ import annotations

from dataclasses import dataclass


class CudaShimError(ValueError):
    """Base class for shim contract violations."""


class UnsupportedCudaFeature(CudaShimError):
    """Raised when a feature is requested on an unsupported device/runtime."""


class InvalidCudaConfiguration(CudaShimError):
    """Raised when an environment or runtime config cannot be production-valid."""


@dataclass(frozen=True, order=True)
class ComputeCapability:
    major: int
    minor: int

    @classmethod
    def parse(cls, value: str | int | tuple[int, int]) -> "ComputeCapability":
        if isinstance(value, tuple):
            return cls(*value)
        if isinstance(value, int):
            text = str(value)
        else:
            text = value.lower().replace("sm_", "").replace("sm", "")
        if "." in text:
            major, minor = text.split(".", 1)
            return cls(int(major), int(minor))
        if len(text) < 2:
            raise InvalidCudaConfiguration(f"invalid compute capability {value!r}")
        return cls(int(text[:-1]), int(text[-1]))

    @property
    def sm(self) -> int:
        return self.major * 10 + self.minor

    @property
    def arch(self) -> str:
        return f"sm_{self.sm}"

    @property
    def is_ampere(self) -> bool:
        return self.major == 8 and self.minor in {0, 6, 7}

    @property
    def is_ada(self) -> bool:
        return self.major == 8 and self.minor == 9

    @property
    def is_hopper(self) -> bool:
        return self.major == 9

    @property
    def is_blackwell(self) -> bool:
        return self.major >= 10

    def supports(self, feature: str) -> bool:
        feature = feature.lower()
        if feature in {"cuda_graphs", "paged_attention", "prefix_cache", "fp16", "bf16"}:
            return self.sm >= 80
        if feature in {"fp8", "w8a8", "flashinfer_fp8"}:
            return self.sm >= 89
        if feature in {"tma", "wgmma", "flash_mla", "cutlass_sm90"}:
            return self.sm >= 90
        if feature in {"fp4", "nvfp4", "mxfp4", "cutlass_sm100"}:
            return self.sm >= 100
        if feature in {"sparse_mla_sm120", "sm120"}:
            return self.sm >= 120
        return False


def gib(value: float) -> int:
    return int(value * 1024**3)


def round_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise InvalidCudaConfiguration("alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment
