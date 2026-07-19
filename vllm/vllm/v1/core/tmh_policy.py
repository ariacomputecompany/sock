# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from enum import IntEnum

from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    UniformTypeKVCacheSpecs,
)

KV_LAYOUT = "tmh_fidelity_paged_kv"


class TMHPageRole(IntEnum):
    PINNED_RAW = 0
    HOT_RAW = 1
    WARM_INT8_INT4 = 2
    WARM_INT8_INT8 = 3


class TMHStorageKind(IntEnum):
    CANONICAL = 0
    REQUEST_OVERLAY = 1


@dataclass(frozen=True)
class TMHPhysicalPageDescriptor:
    request_id: str
    layer_name: str
    logical_block_id: int
    page_index: int
    role: TMHPageRole
    storage: TMHStorageKind
    prefix_cached: bool
    k_quant_mode: str
    v_quant_mode: str

    @property
    def raw(self) -> bool:
        return self.role in (TMHPageRole.PINNED_RAW, TMHPageRole.HOT_RAW)


@dataclass(frozen=True)
class TMHPhysicalEvent:
    request_id: str
    descriptors: tuple[TMHPhysicalPageDescriptor, ...]
    total_pages: int
    recent_start_page: int
    hot_pages: int
    released_request_ids: tuple[str, ...] = ()


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
    _regular_live_bytes_cache: dict[str, tuple[tuple[int, ...], int]] = field(
        default_factory=dict
    )
    _physical_descriptors: dict[
        tuple[str, str, int], TMHPhysicalPageDescriptor
    ] = field(default_factory=dict)
    _pending_physical_events: list[TMHPhysicalEvent] = field(default_factory=list)
    _early_layers: list[TMHLayerShape] = field(init=False, repr=False)
    _late_layers: list[TMHLayerShape] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        late_layer_start = (len(self.layers) * 2) // 3
        self._early_layers = self.layers[:late_layer_start]
        self._late_layers = self.layers[late_layer_start:]

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
        regular_live_bytes = self._regular_live_bytes(request_id, blocks_by_group)

        hot_bytes = 0
        warm_bytes = 0
        raw_equivalent_bytes = 0
        uniform_old_int8_bytes = 0
        old_tokens = _token_count_for_page_span(
            start_page=1,
            end_page=recent_start_page,
            total_tokens=total_tokens,
            page_tokens=self.page_tokens,
        )
        for layer in self.layers:
            raw_equivalent_bytes += _bytes_for_page_span(
                layer=layer,
                start_page=0,
                end_page=total_pages,
                total_tokens=total_tokens,
                page_tokens=self.page_tokens,
                precision="raw",
                component="k",
            )
            raw_equivalent_bytes += _bytes_for_page_span(
                layer=layer,
                start_page=0,
                end_page=total_pages,
                total_tokens=total_tokens,
                page_tokens=self.page_tokens,
                precision="raw",
                component="v",
            )
            for start_page, end_page in ((0, 1), (recent_start_page, total_pages)):
                hot_bytes += _bytes_for_page_span(
                    layer=layer,
                    start_page=start_page,
                    end_page=end_page,
                    total_tokens=total_tokens,
                    page_tokens=self.page_tokens,
                    precision="raw",
                    component="k",
                )
                hot_bytes += _bytes_for_page_span(
                    layer=layer,
                    start_page=start_page,
                    end_page=end_page,
                    total_tokens=total_tokens,
                    page_tokens=self.page_tokens,
                    precision="raw",
                    component="v",
                )

        for layer in self._early_layers:
            warm_bytes += _bytes_for_page_span(
                layer=layer,
                start_page=1,
                end_page=recent_start_page,
                total_tokens=total_tokens,
                page_tokens=self.page_tokens,
                precision="int8",
                component="k",
            )
            warm_bytes += _bytes_for_page_span(
                layer=layer,
                start_page=1,
                end_page=recent_start_page,
                total_tokens=total_tokens,
                page_tokens=self.page_tokens,
                precision="int4",
                component="v",
            )

        for layer in self._late_layers:
            warm_bytes += _bytes_for_page_span(
                layer=layer,
                start_page=1,
                end_page=recent_start_page,
                total_tokens=total_tokens,
                page_tokens=self.page_tokens,
                precision="int8",
                component="k",
            )
            warm_bytes += _bytes_for_page_span(
                layer=layer,
                start_page=1,
                end_page=recent_start_page,
                total_tokens=total_tokens,
                page_tokens=self.page_tokens,
                precision="int8",
                component="v",
            )

        for layer in self.layers:
            uniform_old_int8_bytes += _bytes_for_page_span(
                layer=layer,
                start_page=1,
                end_page=recent_start_page,
                total_tokens=total_tokens,
                page_tokens=self.page_tokens,
                precision="int8",
                component="k",
            )
            uniform_old_int8_bytes += _bytes_for_page_span(
                layer=layer,
                start_page=1,
                end_page=recent_start_page,
                total_tokens=total_tokens,
                page_tokens=self.page_tokens,
                precision="int8",
                component="v",
            )

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
            late_layer_start=len(self._early_layers),
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
        if self.physical:
            self._record_physical_descriptors(
                request_id=request_id,
                total_pages=total_pages,
                recent_start_page=recent_start_page,
                hot_pages=hot_pages,
                blocks_by_group=blocks_by_group,
            )
        return pressure

    def _regular_live_bytes(
        self,
        request_id: str,
        blocks_by_group: tuple[list[KVCacheBlock], ...],
    ) -> int:
        signature = tuple(
            sum(1 for block in blocks if not block.is_null)
            for blocks in blocks_by_group
        )
        cached = self._regular_live_bytes_cache.get(request_id)
        if cached is not None and cached[0] == signature:
            return cached[1]
        total = 0
        for group_index, live_blocks in enumerate(signature):
            if group_index >= len(self.regular_page_bytes_by_group):
                continue
            total += live_blocks * self.regular_page_bytes_by_group[group_index]
        self._regular_live_bytes_cache[request_id] = (signature, total)
        return total

    def forget_request(self, request_id: str) -> None:
        self.latest_by_request.pop(request_id, None)
        self._regular_live_bytes_cache.pop(request_id, None)
        removed = [
            key
            for key in self._physical_descriptors
            if key[0] == request_id
        ]
        for key in removed:
            self._physical_descriptors.pop(key, None)
        if self.physical and removed:
            self._pending_physical_events.append(
                TMHPhysicalEvent(
                    request_id=request_id,
                    descriptors=(),
                    total_pages=0,
                    recent_start_page=0,
                    hot_pages=0,
                    released_request_ids=(request_id,),
                )
            )

    def take_physical_events(self) -> list[TMHPhysicalEvent]:
        events = self._pending_physical_events
        self._pending_physical_events = []
        return events

    def _record_physical_descriptors(
        self,
        request_id: str,
        total_pages: int,
        recent_start_page: int,
        hot_pages: int,
        blocks_by_group: tuple[list[KVCacheBlock], ...],
    ) -> None:
        if not blocks_by_group:
            return
        logical_pages = [
            (block.block_id, block.block_hash is not None or block.ref_cnt > 1)
            for block in blocks_by_group[0]
            if not block.is_null
        ]
        self._record_physical_descriptors_for_pages(
            request_id=request_id,
            total_pages=total_pages,
            recent_start_page=recent_start_page,
            hot_pages=hot_pages,
            logical_pages=logical_pages,
        )

    def record_physical_descriptors_from_block_ids(
        self,
        *,
        request_id: str,
        total_tokens: int,
        logical_block_ids: list[int] | tuple[int, ...],
        prefix_cached_page_indices: set[int] | frozenset[int] = frozenset(),
    ) -> None:
        if not self.physical or not logical_block_ids:
            return
        total_tokens = max(1, total_tokens)
        total_pages = max(1, math.ceil(total_tokens / self.page_tokens))
        hot_pages = (
            0
            if self.hot_budget_pct <= 0
            else min(total_pages, math.ceil(total_pages * self.hot_budget_pct / 100.0))
        )
        recent_start_page = total_pages if hot_pages <= 0 else max(0, total_pages - hot_pages)
        logical_pages = [
            (block_id, page_index in prefix_cached_page_indices)
            for page_index, block_id in enumerate(logical_block_ids[:total_pages])
        ]
        self._record_physical_descriptors_for_pages(
            request_id=request_id,
            total_pages=total_pages,
            recent_start_page=recent_start_page,
            hot_pages=hot_pages,
            logical_pages=logical_pages,
        )

    def _record_physical_descriptors_for_pages(
        self,
        *,
        request_id: str,
        total_pages: int,
        recent_start_page: int,
        hot_pages: int,
        logical_pages: list[tuple[int, bool]],
    ) -> None:
        descriptors: list[TMHPhysicalPageDescriptor] = []
        for page_index, (logical_block_id, prefix_cached) in enumerate(
            logical_pages[:total_pages]
        ):
            for layer in self.layers:
                role = _physical_role_for_page(
                    layer=layer,
                    early_layers=self._early_layers,
                    page_index=page_index,
                    recent_start_page=recent_start_page,
                )
                k_quant_mode, v_quant_mode = _quant_modes_for_role(role)
                storage = _storage_kind_for_role(role, prefix_cached)
                descriptor = TMHPhysicalPageDescriptor(
                    request_id=request_id,
                    layer_name=layer.layer_name,
                    logical_block_id=logical_block_id,
                    page_index=page_index,
                    role=role,
                    storage=storage,
                    prefix_cached=prefix_cached,
                    k_quant_mode=k_quant_mode,
                    v_quant_mode=v_quant_mode,
                )
                key = (request_id, layer.layer_name, page_index)
                if self._physical_descriptors.get(key) != descriptor:
                    self._physical_descriptors[key] = descriptor
                    descriptors.append(descriptor)
        if descriptors:
            self._pending_physical_events.append(
                TMHPhysicalEvent(
                    request_id=request_id,
                    descriptors=tuple(descriptors),
                    total_pages=total_pages,
                    recent_start_page=recent_start_page,
                    hot_pages=hot_pages,
                )
            )


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
                head_size_v=spec.head_size_v,
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
        return sum(
            layer_spec.page_size_bytes for layer_spec in spec.kv_cache_specs.values()
        )
    return spec.page_size_bytes * len(group.layer_names)


def _physical_role_for_page(
    *,
    layer: TMHLayerShape,
    early_layers: list[TMHLayerShape],
    page_index: int,
    recent_start_page: int,
) -> TMHPageRole:
    if page_index == 0:
        return TMHPageRole.PINNED_RAW
    if page_index >= recent_start_page:
        return TMHPageRole.HOT_RAW
    if layer in early_layers:
        return TMHPageRole.WARM_INT8_INT4
    return TMHPageRole.WARM_INT8_INT8


def _quant_modes_for_role(role: TMHPageRole) -> tuple[str, str]:
    if role in (TMHPageRole.PINNED_RAW, TMHPageRole.HOT_RAW):
        return "raw", "raw"
    if role == TMHPageRole.WARM_INT8_INT4:
        return "int8_per_token_head", "int4_per_token_head"
    if role == TMHPageRole.WARM_INT8_INT8:
        return "int8_per_token_head", "int8_per_token_head"
    raise ValueError(f"unknown TMH physical role: {role!r}")


def _storage_kind_for_role(
    role: TMHPageRole,
    prefix_cached: bool,
) -> TMHStorageKind:
    if role in (TMHPageRole.WARM_INT8_INT4, TMHPageRole.WARM_INT8_INT8):
        return TMHStorageKind.CANONICAL
    if role == TMHPageRole.PINNED_RAW:
        return TMHStorageKind.CANONICAL
    if prefix_cached:
        return TMHStorageKind.REQUEST_OVERLAY
    return TMHStorageKind.CANONICAL


def _layer_index(layer_name: str) -> int:
    matches = re.findall(r"\d+", layer_name)
    return int(matches[-1]) if matches else 0


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


def _bytes_for_page_span(
    *,
    layer: TMHLayerShape,
    start_page: int,
    end_page: int,
    total_tokens: int,
    page_tokens: int,
    precision: str,
    component: str,
) -> int:
    token_count = _token_count_for_page_span(
        start_page=start_page,
        end_page=end_page,
        total_tokens=total_tokens,
        page_tokens=page_tokens,
    )
    full_pages, partial_tokens = divmod(token_count, page_tokens)
    total = full_pages * _bytes_for(layer, page_tokens, precision, component)
    if partial_tokens:
        total += _bytes_for(layer, partial_tokens, precision, component)
    return total


def _token_count_for_page_span(
    *,
    start_page: int,
    end_page: int,
    total_tokens: int,
    page_tokens: int,
) -> int:
    if end_page <= start_page:
        return 0
    start = start_page * page_tokens
    end = min(total_tokens, end_page * page_tokens)
    return max(0, end - start)
