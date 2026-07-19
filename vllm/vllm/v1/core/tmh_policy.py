# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field

from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    UniformTypeKVCacheSpecs,
)

KV_LAYOUT = "tmh_fidelity_paged_kv"


@dataclass(frozen=True)
class TMHLayerShape:
    layer_name: str
    layer_index: int
    num_kv_heads: int
    head_size: int
    head_size_v: int
    raw_dtype_bytes: float


@dataclass(frozen=True)
class TMHRequestPressure:
    request_id: str
    kv_layout: str
    policy: str
    total_tokens: int
    prompt_tokens: int
    page_tokens: int
    total_pages: int
    prompt_pages: int
    hot_pages: int
    recent_start_page: int
    layer_count: int
    late_layer_start: int
    regular_live_bytes: int
    tmh_effective_bytes: int
    hot_bytes: int
    warm_bytes: int
    raw_equivalent_bytes: int
    same_hot_uniform_int8_bytes: int
    old_tokens: int
    warm_reduction_vs_uniform_int8_pct: float
    total_reduction_vs_same_hot_uniform_int8_pct: float
    physical: bool = False

    def as_log_fields(self) -> dict[str, int | float | str | bool]:
        return {
            "request_id": self.request_id,
            "kv_layout": self.kv_layout,
            "policy": self.policy,
            "physical": self.physical,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "page_tokens": self.page_tokens,
            "total_pages": self.total_pages,
            "hot_pages": self.hot_pages,
            "regular_live_bytes": self.regular_live_bytes,
            "tmh_effective_bytes": self.tmh_effective_bytes,
            "old_tokens": self.old_tokens,
            "warm_reduction_vs_uniform_int8_pct": round(
                self.warm_reduction_vs_uniform_int8_pct, 3
            ),
            "total_reduction_vs_same_hot_uniform_int8_pct": round(
                self.total_reduction_vs_same_hot_uniform_int8_pct, 3
            ),
        }


@dataclass
class TMHKVRuntimePolicy:
    policy: str
    hot_budget_pct: float
    page_tokens: int
    layers: list[TMHLayerShape]
    regular_page_bytes_by_group: list[int]
    latest_by_request: dict[str, TMHRequestPressure] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return self.policy != "off"

    @property
    def physical(self) -> bool:
        return self.policy == "physical"

    @classmethod
    def from_kv_cache_config(
        cls,
        kv_cache_config: KVCacheConfig,
        scheduler_block_size: int,
    ) -> "TMHKVRuntimePolicy":
        policy = kv_cache_config.tmh_kv_policy
        hot_budget_pct = kv_cache_config.tmh_hot_budget_pct
        if policy == "physical":
            raise RuntimeError(
                "TMH physical mode requires mixed-fidelity warm-page tensors "
                "and attention kernels. Refusing to run as standard KV."
            )
        return cls(
            policy=policy,
            hot_budget_pct=hot_budget_pct,
            page_tokens=scheduler_block_size,
            layers=_extract_layers(kv_cache_config.kv_cache_groups),
            regular_page_bytes_by_group=[
                _regular_page_bytes(group)
                for group in kv_cache_config.kv_cache_groups
            ],
        )

    def record_allocation(
        self,
        request_id: str,
        total_tokens: int,
        prompt_tokens: int,
        blocks_by_group: tuple[list[KVCacheBlock], ...],
    ) -> TMHRequestPressure | None:
        if not self.enabled or not self.layers:
            return None
        total_tokens = max(1, total_tokens)
        prompt_tokens = max(0, min(prompt_tokens, total_tokens))
        total_pages = max(1, math.ceil(total_tokens / self.page_tokens))
        prompt_pages = max(1, math.ceil(max(1, prompt_tokens) / self.page_tokens))
        hot_pages = (
            0
            if self.hot_budget_pct <= 0
            else min(total_pages, math.ceil(total_pages * self.hot_budget_pct / 100.0))
        )
        recent_start_page = total_pages if hot_pages <= 0 else max(0, total_pages - hot_pages)
        late_layer_start = (len(self.layers) * 2) // 3
        regular_live_bytes = self._regular_live_bytes(blocks_by_group)

        hot_bytes = 0
        warm_bytes = 0
        raw_equivalent_bytes = 0
        uniform_old_int8_bytes = 0
        old_tokens = _token_sum(1, recent_start_page - 1, total_tokens, self.page_tokens)
        for layer_pos, layer in enumerate(self.layers):
            for page_id in range(total_pages):
                tokens = _page_token_count(page_id, total_tokens, self.page_tokens)
                raw_equivalent_bytes += _bytes_for(layer, tokens, "raw", "k")
                raw_equivalent_bytes += _bytes_for(layer, tokens, "raw", "v")
                k_precision, v_precision, tier = _resolve_tmh_page(
                    page_id=page_id,
                    layer_pos=layer_pos,
                    recent_start_page=recent_start_page,
                    late_layer_start=late_layer_start,
                )
                page_bytes = _bytes_for(layer, tokens, k_precision, "k")
                page_bytes += _bytes_for(layer, tokens, v_precision, "v")
                if tier in {"pinned", "hot"}:
                    hot_bytes += page_bytes
                else:
                    warm_bytes += page_bytes
                    uniform_old_int8_bytes += _bytes_for(layer, tokens, "int8", "k")
                    uniform_old_int8_bytes += _bytes_for(layer, tokens, "int8", "v")

        tmh_effective_bytes = hot_bytes + warm_bytes
        same_hot_uniform_int8_bytes = hot_bytes + uniform_old_int8_bytes
        warm_reduction = (
            0.0
            if uniform_old_int8_bytes <= 0
            else 100.0 * (1.0 - (warm_bytes / uniform_old_int8_bytes))
        )
        total_reduction = (
            0.0
            if same_hot_uniform_int8_bytes <= 0
            else 100.0 * (1.0 - (tmh_effective_bytes / same_hot_uniform_int8_bytes))
        )
        pressure = TMHRequestPressure(
            request_id=request_id,
            kv_layout=KV_LAYOUT,
            policy=self.policy,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            page_tokens=self.page_tokens,
            total_pages=total_pages,
            prompt_pages=prompt_pages,
            hot_pages=hot_pages,
            recent_start_page=recent_start_page,
            layer_count=len(self.layers),
            late_layer_start=late_layer_start,
            regular_live_bytes=regular_live_bytes,
            tmh_effective_bytes=tmh_effective_bytes,
            hot_bytes=hot_bytes,
            warm_bytes=warm_bytes,
            raw_equivalent_bytes=raw_equivalent_bytes,
            same_hot_uniform_int8_bytes=same_hot_uniform_int8_bytes,
            old_tokens=old_tokens,
            warm_reduction_vs_uniform_int8_pct=warm_reduction,
            total_reduction_vs_same_hot_uniform_int8_pct=total_reduction,
            physical=self.physical,
        )
        self.latest_by_request[request_id] = pressure
        return pressure

    def _regular_live_bytes(
        self, blocks_by_group: tuple[list[KVCacheBlock], ...]
    ) -> int:
        total = 0
        for group_index, blocks in enumerate(blocks_by_group):
            if group_index >= len(self.regular_page_bytes_by_group):
                continue
            live_blocks = sum(1 for block in blocks if not block.is_null)
            total += live_blocks * self.regular_page_bytes_by_group[group_index]
        return total


def should_log_allocations() -> bool:
    return os.getenv("VLLM_TMH_LOG_ALLOCATIONS", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _extract_layers(groups: list[KVCacheGroupSpec]) -> list[TMHLayerShape]:
    layers: dict[str, TMHLayerShape] = {}
    for group in groups:
        spec_by_layer = _spec_by_layer(group)
        for layer_name in group.layer_names:
            spec = spec_by_layer.get(layer_name)
            if not isinstance(spec, AttentionSpec):
                continue
            layers[layer_name] = TMHLayerShape(
                layer_name=layer_name,
                layer_index=_layer_index(layer_name),
                num_kv_heads=spec.num_kv_heads,
                head_size=spec.head_size,
                head_size_v=getattr(spec, "head_size_v", spec.head_size),
                raw_dtype_bytes=float(get_dtype_size(spec.dtype)),
            )
    return sorted(layers.values(), key=lambda layer: (layer.layer_index, layer.layer_name))


def _spec_by_layer(group: KVCacheGroupSpec) -> dict[str, object]:
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return dict(spec.kv_cache_specs)
    return {layer_name: spec for layer_name in group.layer_names}


def _regular_page_bytes(group: KVCacheGroupSpec) -> int:
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return sum(layer_spec.page_size_bytes for layer_spec in spec.kv_cache_specs.values())
    return spec.page_size_bytes * len(group.layer_names)


def _layer_index(layer_name: str) -> int:
    matches = re.findall(r"\d+", layer_name)
    return int(matches[-1]) if matches else 0


def _resolve_tmh_page(
    *,
    page_id: int,
    layer_pos: int,
    recent_start_page: int,
    late_layer_start: int,
) -> tuple[str, str, str]:
    if page_id == 0:
        return "raw", "raw", "pinned"
    if page_id >= recent_start_page:
        return "raw", "raw", "hot"
    if layer_pos < late_layer_start:
        return "int8", "int4", "warm"
    return "int8", "int8", "warm"


def _bytes_for(
    layer: TMHLayerShape,
    tokens: int,
    precision: str,
    component: str,
) -> int:
    if precision == "raw":
        bytes_per_scalar = layer.raw_dtype_bytes
    elif precision == "int8":
        bytes_per_scalar = 1.0
    elif precision == "int4":
        bytes_per_scalar = 0.5
    else:
        raise ValueError(f"unknown TMH precision {precision!r}")
    head_size = layer.head_size if component == "k" else layer.head_size_v
    return int(math.ceil(tokens * layer.num_kv_heads * head_size * bytes_per_scalar))


def _page_token_count(page_id: int, total_tokens: int, page_tokens: int) -> int:
    start = page_id * page_tokens
    end = min(total_tokens, start + page_tokens)
    return max(0, end - start)


def _token_sum(
    start_page: int,
    end_page: int,
    total_tokens: int,
    page_tokens: int,
) -> int:
    if end_page < start_page:
        return 0
    return sum(
        _page_token_count(page_id, total_tokens, page_tokens)
        for page_id in range(start_page, end_page + 1)
    )
