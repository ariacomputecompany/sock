"""CUDA/NVIDIA environment contract parsing."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from sock_cuda_shim.capabilities import InvalidCudaConfiguration


CUDA_ORDER = "PCI_BUS_ID"
TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class CudaEnvironment:
    cuda_visible_devices: tuple[str, ...] | None = None
    cuda_device_order: str = CUDA_ORDER
    cuda_module_loading: str = "LAZY"
    torch_cuda_arch_list: tuple[str, ...] = field(default_factory=tuple)
    vllm_attention_backend: str | None = None
    vllm_use_v1: bool = True
    nccl_p2p_disable: bool = False
    nccl_ib_disable: bool = False
    cuda_launch_blocking: bool = False
    cudagraph_capture_sizes: tuple[int, ...] = field(default_factory=tuple)
    raw: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: dict[str, str] | None = None) -> "CudaEnvironment":
        env = dict(os.environ if mapping is None else mapping)
        visible = _parse_visible(env.get("CUDA_VISIBLE_DEVICES"))
        arch_list = _parse_arch_list(env.get("TORCH_CUDA_ARCH_LIST", ""))
        capture_sizes = _parse_int_list(env.get("VLLM_CUDAGRAPH_CAPTURE_SIZES", ""))
        out = cls(
            cuda_visible_devices=visible,
            cuda_device_order=env.get("CUDA_DEVICE_ORDER", CUDA_ORDER),
            cuda_module_loading=env.get("CUDA_MODULE_LOADING", "LAZY").upper(),
            torch_cuda_arch_list=arch_list,
            vllm_attention_backend=env.get("VLLM_ATTENTION_BACKEND"),
            vllm_use_v1=_parse_bool(env.get("VLLM_USE_V1", "1"), "VLLM_USE_V1"),
            nccl_p2p_disable=_parse_bool(env.get("NCCL_P2P_DISABLE", "0"), "NCCL_P2P_DISABLE"),
            nccl_ib_disable=_parse_bool(env.get("NCCL_IB_DISABLE", "0"), "NCCL_IB_DISABLE"),
            cuda_launch_blocking=_parse_bool(env.get("CUDA_LAUNCH_BLOCKING", "0"), "CUDA_LAUNCH_BLOCKING"),
            cudagraph_capture_sizes=capture_sizes,
            raw={k: v for k, v in env.items() if k.startswith(("CUDA", "NCCL", "VLLM", "TORCH"))},
        )
        out.validate()
        return out

    def validate(self) -> None:
        if self.cuda_device_order != CUDA_ORDER:
            raise InvalidCudaConfiguration(
                "CUDA_DEVICE_ORDER must be PCI_BUS_ID for stable production ordinals"
            )
        if self.cuda_module_loading not in {"LAZY", "EAGER"}:
            raise InvalidCudaConfiguration("CUDA_MODULE_LOADING must be LAZY or EAGER")
        if self.cuda_launch_blocking and self.cudagraph_capture_sizes:
            raise InvalidCudaConfiguration(
                "CUDA_LAUNCH_BLOCKING is incompatible with cudagraph capture benchmarking"
            )
        if self.vllm_attention_backend:
            allowed = {
                "FLASH_ATTN",
                "FLASHINFER",
                "TRITON_ATTN",
                "TRITON_MLA",
                "CUTLASS_MLA",
                "TORCH_SDPA",
            }
            if self.vllm_attention_backend.upper() not in allowed:
                raise InvalidCudaConfiguration(
                    f"unknown CUDA attention backend {self.vllm_attention_backend!r}"
                )
        if any(size <= 0 for size in self.cudagraph_capture_sizes):
            raise InvalidCudaConfiguration("cudagraph capture sizes must be positive")


def _parse_visible(value: str | None) -> tuple[str, ...] | None:
    if value is None or value.strip() == "":
        return None
    if value.strip() in {"-1", "none", "None"}:
        return tuple()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_bool(value: str | None, name: str) -> bool:
    text = "0" if value is None else value.strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    raise InvalidCudaConfiguration(f"{name} must be a boolean-like value")


def _parse_arch_list(value: str) -> tuple[str, ...]:
    if not value.strip():
        return tuple()
    normalized = value.replace(";", " ").replace(",", " ")
    return tuple(part.strip() for part in normalized.split() if part.strip())


def _parse_int_list(value: str) -> tuple[int, ...]:
    if not value.strip():
        return tuple()
    normalized = value.replace(";", ",")
    return tuple(int(part.strip()) for part in normalized.split(",") if part.strip())
