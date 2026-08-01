# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field, replace as dataclass_replace
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
TMH_EVENT_SCHEMA_VERSION = 1


class TMHPageRole(IntEnum):
    PINNED_RAW = 0
    HOT_RAW = 1
    WARM_INT8_INT4 = 2
    WARM_INT8_INT8 = 3


class TMHStorageKind(IntEnum):
    CANONICAL = 0
    REQUEST_OVERLAY = 1


class TMHPhysicalEventKind(IntEnum):
    """Wire-level semantics for scheduler-to-worker physical updates."""

    DELTA = 0
    RELEASE = 1


class TMHStorageLocation(IntEnum):
    DEVICE = 0
    CPU_OFFLOAD = 1
    RECOMPUTE = 2


class TMHMaterializationState(IntEnum):
    PLANNED = 0
    TRANSITIONING = 1
    READY = 2


class TMHTransitionOperation(IntEnum):
    NONE = 0
    COPY = 1
    QUANTIZE = 2
    DEQUANTIZE = 3
    REENCODE = 4
    LOAD = 5
    RECOMPUTE = 6


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
    cache_group_id: int = 0
    allocation_generation: int = 0
    valid_tokens: int = 0
    storage_location: TMHStorageLocation = TMHStorageLocation.DEVICE
    materialization_state: TMHMaterializationState = TMHMaterializationState.READY
    transition_operation: TMHTransitionOperation = TMHTransitionOperation.NONE
    retention_priority: float = 0.0
    placement_score: float | None = None
    policy_metadata: tuple[tuple[str, str], ...] = ()

    @property
    def raw(self) -> bool:
        return self.role in (TMHPageRole.PINNED_RAW, TMHPageRole.HOT_RAW)


@dataclass(frozen=True)
class TMHPhysicalEvent:
    schema_version: int
    event_kind: TMHPhysicalEventKind
    request_id: str
    request_generation: int
    sequence: int
    expected_base_version: int
    target_version: int
    commit_id: str
    descriptors: tuple[TMHPhysicalPageDescriptor, ...]
    total_pages: int
    recent_start_page: int
    hot_pages: int
    released_request_ids: tuple[str, ...] = ()
    released_descriptors: tuple[TMHPhysicalPageDescriptor, ...] = ()
    dependency_commit_ids: tuple[str, ...] = ()
    planner_policy: str = "anchor_recent_fixed_layer_split_v1"


@dataclass(frozen=True)
class TMHLayerShape:
    layer_name: str
    layer_index: int
    num_kv_heads: int
    head_size: int
    head_size_v: int
    raw_dtype_bytes: float
    cache_group_id: int = 0
    late_layer: bool | None = None


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
    _canonical_descriptor_refcounts: dict[tuple[str, int, int, int], int] = field(
        default_factory=dict
    )
    _resident_canonical_descriptors: dict[
        tuple[str, int, int], TMHPhysicalPageDescriptor
    ] = field(default_factory=dict)
    _request_event_state: dict[str, tuple[int, int, int]] = field(
        default_factory=dict
    )
    _request_generation_counters: dict[str, int] = field(default_factory=dict)
    _request_total_pages: dict[str, int] = field(default_factory=dict)
    _request_physical_tokens: dict[str, int] = field(default_factory=dict)
    _request_physical_fingerprint: dict[
        str, tuple[tuple[int, tuple[tuple[int, int, bool], ...]], ...]
    ] = field(default_factory=dict)
    _pending_physical_events: list[TMHPhysicalEvent] = field(default_factory=list)
    _early_layers: list[TMHLayerShape] = field(init=False, repr=False)
    _late_layers: list[TMHLayerShape] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        late_layer_start = (len(self.layers) * 2) // 3
        resolved_layers: list[TMHLayerShape] = []
        for position, layer in enumerate(self.layers):
            resolved_layers.append(
                layer
                if layer.late_layer is not None
                else dataclass_replace(layer, late_layer=position >= late_layer_start)
            )
        self.layers = resolved_layers
        self._early_layers = [layer for layer in self.layers if not layer.late_layer]
        self._late_layers = [layer for layer in self.layers if layer.late_layer]

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

    def _effective_hot_budget_pct(self, *, total_pages: int) -> float:
        pct = self.hot_budget_pct
        if self.physical and total_pages >= 32:
            pct = max(pct, 50.0)
        return pct

    def _effective_hot_pages(self, *, total_pages: int) -> int:
        hot_budget_pct = self._effective_hot_budget_pct(total_pages=total_pages)
        if hot_budget_pct <= 0:
            return 0
        return min(total_pages, math.ceil(total_pages * hot_budget_pct / 100.0))

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
        hot_pages = self._effective_hot_pages(total_pages=total_pages)
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
                total_tokens=total_tokens,
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
        removed_descriptors: list[TMHPhysicalPageDescriptor] = []
        for key, descriptor in list(self._physical_descriptors.items()):
            if key[0] != request_id:
                continue
            removed_descriptors.append(descriptor)
            self._physical_descriptors.pop(key, None)
            self._untrack_canonical_descriptor(descriptor)
        if self.physical and removed_descriptors:
            self._pending_physical_events.append(
                self._next_physical_event(
                    request_id=request_id,
                    event_kind=TMHPhysicalEventKind.RELEASE,
                    descriptors=(),
                    total_pages=0,
                    recent_start_page=0,
                    hot_pages=0,
                    released_request_ids=(request_id,),
                )
            )
        self._request_total_pages.pop(request_id, None)
        self._request_physical_tokens.pop(request_id, None)
        self._request_physical_fingerprint.pop(request_id, None)

    def take_physical_events(self) -> list[TMHPhysicalEvent]:
        events = self._pending_physical_events
        self._pending_physical_events = []
        return events

    def _record_physical_descriptors(
        self,
        request_id: str,
        total_tokens: int,
        total_pages: int,
        recent_start_page: int,
        hot_pages: int,
        blocks_by_group: tuple[list[KVCacheBlock], ...],
    ) -> None:
        if not blocks_by_group:
            return
        logical_pages_by_group = {
            group_id: [
                (
                    block.block_id,
                    block.allocation_generation,
                    block.block_hash is not None or block.ref_cnt > 1,
                )
                for block in blocks
                if not block.is_null
            ]
            for group_id, blocks in enumerate(blocks_by_group)
        }
        self._record_physical_descriptors_for_pages(
            request_id=request_id,
            total_tokens=total_tokens,
            total_pages=total_pages,
            recent_start_page=recent_start_page,
            hot_pages=hot_pages,
            logical_pages_by_group=logical_pages_by_group,
        )

    def record_physical_descriptors_from_block_ids(
        self,
        *,
        request_id: str,
        total_tokens: int,
        logical_block_ids: list[int] | tuple[int, ...],
        prefix_cached_page_indices: set[int] | frozenset[int] = frozenset(),
    ) -> None:
        self.record_physical_descriptors_from_group_block_ids(
            request_id=request_id,
            total_tokens=total_tokens,
            logical_block_ids_by_group={0: logical_block_ids},
            prefix_cached_page_indices_by_group={0: prefix_cached_page_indices},
        )

    def record_physical_descriptors_from_group_block_ids(
        self,
        *,
        request_id: str,
        total_tokens: int,
        logical_block_ids_by_group: dict[int, list[int] | tuple[int, ...]],
        prefix_cached_page_indices_by_group: dict[
            int, set[int] | frozenset[int]
        ] | None = None,
    ) -> None:
        if not self.physical or not logical_block_ids_by_group:
            return
        prefix_cached_page_indices_by_group = (
            prefix_cached_page_indices_by_group or {}
        )
        total_tokens = max(1, total_tokens)
        total_pages = max(1, math.ceil(total_tokens / self.page_tokens))
        hot_pages = self._effective_hot_pages(total_pages=total_pages)
        recent_start_page = total_pages if hot_pages <= 0 else max(0, total_pages - hot_pages)
        logical_pages_by_group = {
            group_id: [
                (
                    block_id,
                    0,
                    page_index
                    in prefix_cached_page_indices_by_group.get(
                        group_id, frozenset()
                    ),
                )
                for page_index, block_id in enumerate(block_ids[:total_pages])
            ]
            for group_id, block_ids in logical_block_ids_by_group.items()
        }
        self._record_physical_descriptors_for_pages(
            request_id=request_id,
            total_tokens=total_tokens,
            total_pages=total_pages,
            recent_start_page=recent_start_page,
            hot_pages=hot_pages,
            logical_pages_by_group=logical_pages_by_group,
        )

    def _record_physical_descriptors_for_pages(
        self,
        *,
        request_id: str,
        total_tokens: int,
        total_pages: int,
        recent_start_page: int,
        hot_pages: int,
        logical_pages_by_group: dict[int, list[tuple[int, int, bool]]],
    ) -> None:
        previous_total_pages = self._request_total_pages.get(request_id, 0)
        previous_total_tokens = self._request_physical_tokens.get(request_id)
        physical_fingerprint = tuple(
            (group_id, tuple(pages[:total_pages]))
            for group_id, pages in sorted(logical_pages_by_group.items())
        )
        previous_fingerprint = self._request_physical_fingerprint.get(request_id)
        self._request_physical_tokens[request_id] = total_tokens
        self._request_physical_fingerprint[request_id] = physical_fingerprint
        if (
            previous_total_tokens is not None
            and total_tokens > previous_total_tokens
            and total_tokens % self.page_tokens != 0
            and total_pages == previous_total_pages
            and physical_fingerprint == previous_fingerprint
        ):
            # The fused writer advances the current page's device-side valid
            # range. With identical block identities/generations and prefix
            # state, monotonic in-page progress cannot change placement.
            return
        descriptors: list[TMHPhysicalPageDescriptor] = []
        released_descriptors: list[TMHPhysicalPageDescriptor] = []
        for layer in self.layers:
            logical_pages = logical_pages_by_group.get(layer.cache_group_id, [])
            for page_index, (
                logical_block_id,
                allocation_generation,
                prefix_cached,
            ) in enumerate(logical_pages[:total_pages]):
                role = _physical_role_for_page(
                    layer=layer,
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
                    cache_group_id=layer.cache_group_id,
                    allocation_generation=allocation_generation,
                    valid_tokens=min(
                        self.page_tokens,
                        max(0, total_tokens - page_index * self.page_tokens),
                    ),
                    retention_priority=_retention_priority(role, page_index),
                )
                key = (request_id, layer.layer_name, page_index)
                old_descriptor = self._physical_descriptors.get(key)
                if old_descriptor != descriptor:
                    only_intermediate_valid_growth = (
                        old_descriptor is not None
                        and old_descriptor.valid_tokens < descriptor.valid_tokens
                        and descriptor.valid_tokens < self.page_tokens
                        and dataclass_replace(
                            descriptor,
                            valid_tokens=old_descriptor.valid_tokens,
                        )
                        == old_descriptor
                    )
                    if old_descriptor is not None:
                        old_key = _canonical_descriptor_key(old_descriptor)
                        new_key = _canonical_descriptor_key(descriptor)
                        if old_key != new_key:
                            self._untrack_canonical_descriptor(old_descriptor)
                        if old_key != new_key:
                            released_descriptors.extend(
                                self._track_canonical_descriptor(descriptor)
                            )
                    else:
                        released_descriptors.extend(
                            self._track_canonical_descriptor(descriptor)
                        )
                    self._physical_descriptors[key] = descriptor
                    # Monotonic progress inside an already-published page is
                    # written into the device valid-token table by the fused KV
                    # writer. Publish at allocation, representation changes,
                    # shrink, and page completion; do not build and consume a
                    # 24-layer transaction for every single decode token.
                    if not only_intermediate_valid_growth:
                        descriptors.append(descriptor)
        if total_pages < previous_total_pages:
            for key, old_descriptor in list(self._physical_descriptors.items()):
                if key[0] != request_id or old_descriptor.page_index < total_pages:
                    continue
                self._physical_descriptors.pop(key, None)
                self._untrack_canonical_descriptor(old_descriptor)
        total_pages_changed = previous_total_pages != total_pages
        self._request_total_pages[request_id] = total_pages
        if descriptors or released_descriptors or total_pages_changed:
            self._pending_physical_events.append(
                self._next_physical_event(
                    request_id=request_id,
                    event_kind=TMHPhysicalEventKind.DELTA,
                    descriptors=tuple(descriptors),
                    total_pages=total_pages,
                    recent_start_page=recent_start_page,
                    hot_pages=hot_pages,
                    released_descriptors=tuple(released_descriptors),
                )
            )

    def _track_canonical_descriptor(
        self,
        descriptor: TMHPhysicalPageDescriptor,
    ) -> list[TMHPhysicalPageDescriptor]:
        descriptor_key = _canonical_descriptor_key(descriptor)
        if descriptor_key is None:
            return []
        resident_key = _canonical_resident_key(descriptor)
        replaced: list[TMHPhysicalPageDescriptor] = []
        resident = self._resident_canonical_descriptors.get(resident_key)
        if resident is not None and (
            resident.allocation_generation != descriptor.allocation_generation
        ):
            old_key = _canonical_descriptor_key(resident)
            if old_key is not None:
                self._canonical_descriptor_refcounts.pop(old_key, None)
            replaced.append(resident)
        self._resident_canonical_descriptors[resident_key] = descriptor
        self._canonical_descriptor_refcounts[descriptor_key] = (
            self._canonical_descriptor_refcounts.get(descriptor_key, 0) + 1
        )
        return replaced

    def _untrack_canonical_descriptor(
        self,
        descriptor: TMHPhysicalPageDescriptor,
    ) -> bool:
        descriptor_key = _canonical_descriptor_key(descriptor)
        if descriptor_key is None:
            return False
        count = self._canonical_descriptor_refcounts.get(descriptor_key, 0)
        if count <= 1:
            self._canonical_descriptor_refcounts.pop(descriptor_key, None)
            # Canonical storage remains resident while its logical cache block
            # generation is resident. Reuse of the block id emits its release.
            return False
        self._canonical_descriptor_refcounts[descriptor_key] = count - 1
        return False

    def _next_physical_event(
        self,
        *,
        request_id: str,
        event_kind: TMHPhysicalEventKind,
        descriptors: tuple[TMHPhysicalPageDescriptor, ...],
        total_pages: int,
        recent_start_page: int,
        hot_pages: int,
        released_request_ids: tuple[str, ...] = (),
        released_descriptors: tuple[TMHPhysicalPageDescriptor, ...] = (),
    ) -> TMHPhysicalEvent:
        state = self._request_event_state.get(request_id)
        if state is None:
            generation = self._request_generation_counters.get(request_id, 0) + 1
            self._request_generation_counters[request_id] = generation
            sequence = 1
            base_version = 0
        else:
            generation, previous_sequence, base_version = state
            sequence = previous_sequence + 1
        target_version = base_version + 1
        self._request_event_state[request_id] = (
            generation,
            sequence,
            target_version,
        )
        if event_kind == TMHPhysicalEventKind.RELEASE:
            self._request_event_state.pop(request_id, None)
        return TMHPhysicalEvent(
            schema_version=TMH_EVENT_SCHEMA_VERSION,
            event_kind=event_kind,
            request_id=request_id,
            request_generation=generation,
            sequence=sequence,
            expected_base_version=base_version,
            target_version=target_version,
            commit_id=f"{request_id}:{generation}:{sequence}:{target_version}",
            descriptors=descriptors,
            total_pages=total_pages,
            recent_start_page=recent_start_page,
            hot_pages=hot_pages,
            released_request_ids=released_request_ids,
            released_descriptors=released_descriptors,
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
    for group_id, group in enumerate(groups):
        spec_by_layer = _spec_by_layer(group)
        for layer_name in group.layer_names:
            spec = spec_by_layer.get(layer_name)
            if not isinstance(spec, AttentionSpec):
                continue
            layer_index = _layer_index(layer_name)
            late_layer_start = getattr(spec, "tmh_late_layer_start", 0)
            late_layer = (
                layer_index >= late_layer_start
                if late_layer_start > 0
                else getattr(spec, "tmh_late_layer", None)
            )
            layers[layer_name] = TMHLayerShape(
                layer_name=layer_name,
                layer_index=layer_index,
                num_kv_heads=spec.num_kv_heads,
                head_size=spec.head_size,
                head_size_v=spec.head_size_v,
                raw_dtype_bytes=float(get_dtype_size(spec.dtype)),
                cache_group_id=group_id,
                late_layer=late_layer,
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
    page_index: int,
    recent_start_page: int,
) -> TMHPageRole:
    if page_index == 0:
        return TMHPageRole.PINNED_RAW
    if page_index >= recent_start_page:
        return TMHPageRole.HOT_RAW
    if not layer.late_layer:
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


def _retention_priority(role: TMHPageRole, page_index: int) -> float:
    if role == TMHPageRole.PINNED_RAW:
        return 1_000_000.0
    if role == TMHPageRole.HOT_RAW:
        return 100_000.0 + page_index
    return 1_000.0 + page_index


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


def _canonical_descriptor_key(
    descriptor: TMHPhysicalPageDescriptor,
) -> tuple[str, int, int, int] | None:
    if descriptor.storage != TMHStorageKind.CANONICAL:
        return None
    return (
        descriptor.layer_name,
        descriptor.cache_group_id,
        descriptor.logical_block_id,
        descriptor.allocation_generation,
    )


def _canonical_resident_key(
    descriptor: TMHPhysicalPageDescriptor,
) -> tuple[str, int, int]:
    return (
        descriptor.layer_name,
        descriptor.cache_group_id,
        descriptor.logical_block_id,
    )


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
    payload = int(
        math.ceil(
            tokens * layer.num_kv_heads * head_size * bytes_per_scalar
        )
    )
    scale_bytes = 0 if precision == "raw" else tokens * layer.num_kv_heads * 4
    return payload + scale_bytes


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
