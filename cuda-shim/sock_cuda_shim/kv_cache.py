"""Paged KV cache contracts, including TMH physical/accounting pressure."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from sock_cuda_shim.capabilities import InvalidCudaConfiguration


class KVLayout(str, Enum):
    VLLM_BLOCK_MAJOR = "vllm_block_major"
    FLASHINFER_PAGED = "flashinfer_paged"
    TRTLLM_PAGED = "trtllm_paged"
    TMH_FIDELITY_PAGED = "tmh_fidelity_paged_kv"


class Precision(str, Enum):
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"
    NVFP4 = "nvfp4"


PRECISION_BYTES = {
    Precision.FP16: 2.0,
    Precision.BF16: 2.0,
    Precision.FP8: 1.0,
    Precision.INT8: 1.0,
    Precision.INT4: 0.5,
    Precision.NVFP4: 0.5,
}


@dataclass(frozen=True)
class KVPageSpec:
    block_tokens: int
    num_layers: int
    num_kv_heads: int
    head_size_k: int
    head_size_v: int
    precision: Precision = Precision.FP16
    layout: KVLayout = KVLayout.VLLM_BLOCK_MAJOR

    def __post_init__(self) -> None:
        for name, value in (
            ("block_tokens", self.block_tokens),
            ("num_layers", self.num_layers),
            ("num_kv_heads", self.num_kv_heads),
            ("head_size_k", self.head_size_k),
            ("head_size_v", self.head_size_v),
        ):
            if value <= 0:
                raise InvalidCudaConfiguration(f"{name} must be positive")

    @property
    def page_bytes(self) -> int:
        bytes_per = PRECISION_BYTES[self.precision]
        scalars = self.block_tokens * self.num_layers * self.num_kv_heads
        return math.ceil(scalars * (self.head_size_k + self.head_size_v) * bytes_per)

    def pages_for_tokens(self, tokens: int) -> int:
        if tokens < 0:
            raise InvalidCudaConfiguration("tokens cannot be negative")
        return max(1, math.ceil(max(1, tokens) / self.block_tokens))


@dataclass(frozen=True)
class PagedKVRequest:
    request_id: str
    prompt_tokens: int
    generated_tokens: int
    slot_mapping: tuple[int, ...]
    block_table: tuple[int, ...]

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.generated_tokens

    def validate(self, spec: KVPageSpec) -> None:
        if self.prompt_tokens < 0 or self.generated_tokens < 0:
            raise InvalidCudaConfiguration("request token counts cannot be negative")
        needed_pages = spec.pages_for_tokens(self.total_tokens)
        if len(self.block_table) < needed_pages:
            raise InvalidCudaConfiguration("block table is shorter than required pages")
        if len(self.slot_mapping) != self.total_tokens:
            raise InvalidCudaConfiguration("slot mapping must contain one entry per token")
        if any(slot < 0 for slot in self.slot_mapping):
            raise InvalidCudaConfiguration("slot mapping cannot contain negative slots")


@dataclass(frozen=True)
class TMHPhysicalPolicy:
    hot_budget_pct: float = 25.0
    anchor_pages: int = 1
    early_layer_k: Precision = Precision.INT8
    early_layer_v: Precision = Precision.INT4
    late_layer_k: Precision = Precision.INT8
    late_layer_v: Precision = Precision.INT8

    def __post_init__(self) -> None:
        if not (0 <= self.hot_budget_pct <= 100):
            raise InvalidCudaConfiguration("hot budget must be in [0, 100]")
        if self.anchor_pages < 0:
            raise InvalidCudaConfiguration("anchor pages cannot be negative")

    def pressure(self, spec: KVPageSpec, total_tokens: int) -> dict[str, float | int | str]:
        total_pages = spec.pages_for_tokens(total_tokens)
        hot_pages = min(
            total_pages,
            self.anchor_pages
            + math.ceil(max(0, total_pages - self.anchor_pages) * self.hot_budget_pct / 100.0),
        )
        old_pages = max(0, total_pages - hot_pages)
        regular_bytes = spec.page_bytes * total_pages
        hot_bytes = spec.page_bytes * hot_pages
        early_layers = (spec.num_layers * 2) // 3
        late_layers = spec.num_layers - early_layers
        old_tokens = old_pages * spec.block_tokens
        early_old = _bytes_for(old_tokens, early_layers, spec.num_kv_heads, spec.head_size_k, self.early_layer_k)
        early_old += _bytes_for(old_tokens, early_layers, spec.num_kv_heads, spec.head_size_v, self.early_layer_v)
        late_old = _bytes_for(old_tokens, late_layers, spec.num_kv_heads, spec.head_size_k, self.late_layer_k)
        late_old += _bytes_for(old_tokens, late_layers, spec.num_kv_heads, spec.head_size_v, self.late_layer_v)
        tmh_bytes = hot_bytes + early_old + late_old
        return {
            "layout": KVLayout.TMH_FIDELITY_PAGED.value,
            "total_pages": total_pages,
            "hot_pages": hot_pages,
            "old_pages": old_pages,
            "regular_bytes": regular_bytes,
            "tmh_effective_bytes": tmh_bytes,
            "reduction_pct": 0.0 if regular_bytes == 0 else 100.0 * (1.0 - tmh_bytes / regular_bytes),
        }


def _bytes_for(
    tokens: int,
    layers: int,
    heads: int,
    head_size: int,
    precision: Precision,
) -> int:
    return math.ceil(tokens * layers * heads * head_size * PRECISION_BYTES[precision])
