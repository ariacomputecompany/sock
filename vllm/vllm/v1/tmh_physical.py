# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from vllm.v1.core.tmh_policy import (
    TMHPageRole,
    TMHPhysicalEvent,
    TMHPhysicalPageDescriptor,
    TMHStorageKind,
)
from vllm.v1.kv_cache_interface import TMHFullAttentionSpec


@dataclass
class TMHPhysicalKVCache:
    """Physical TMH cache tensors for one attention layer."""

    spec: TMHFullAttentionSpec
    num_logical_blocks: int
    raw_kv_cache: torch.Tensor
    raw_key: torch.Tensor
    raw_value: torch.Tensor
    warm_key: torch.Tensor
    warm_value: torch.Tensor
    warm_k_scale: torch.Tensor
    warm_v_scale: torch.Tensor
    canonical_role_by_logical_block: torch.Tensor
    canonical_slot_by_logical_block: torch.Tensor
    request_slot_by_row_page: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.raw_key.device

    @property
    def dtype(self) -> torch.dtype:
        return self.raw_key.dtype

    def numel(self) -> int:
        return (
            self.raw_key.numel()
            + self.raw_value.numel()
            + self.warm_key.numel()
            + self.warm_value.numel()
        )


def reshape_tmh_physical_kv_cache(
    kv_raw_tensor: torch.Tensor,
    spec: TMHFullAttentionSpec,
    num_logical_blocks: int,
) -> TMHPhysicalKVCache:
    raw_pages, warm_pages = spec.physical_pool_page_counts(num_logical_blocks)
    byte_view = kv_raw_tensor.view(torch.uint8)
    offset = 0

    def take(num_bytes: int, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
        nonlocal offset
        dtype_size = torch.empty((), dtype=dtype).element_size()
        if offset % dtype_size:
            offset += dtype_size - (offset % dtype_size)
        end = offset + num_bytes
        if end > byte_view.numel():
            raise ValueError(
                "TMH physical cache allocation is too small for the planned "
                f"layout: need byte {end}, have {byte_view.numel()}."
            )
        tensor = byte_view[offset:end].view(dtype).view(shape)
        offset = end
        return tensor

    if spec.head_size_v != spec.head_size:
        raise ValueError(
            "TMH physical raw-native cache requires head_size_v == head_size"
        )
    raw_kv_shape = (2, raw_pages, spec.block_size, spec.num_kv_heads, spec.head_size)
    warm_v_head = spec.head_size_v if spec.tmh_late_layer else (spec.head_size_v + 1) // 2
    warm_k_shape = (warm_pages, spec.block_size, spec.num_kv_heads, spec.head_size)
    warm_v_shape = (warm_pages, spec.block_size, spec.num_kv_heads, warm_v_head)
    warm_scale_shape = (warm_pages, spec.block_size, spec.num_kv_heads)

    raw_dtype_size = torch.empty((), dtype=spec.dtype).element_size()
    raw_kv_cache = take(
        2 * raw_pages * spec.block_size * spec.num_kv_heads * spec.head_size * raw_dtype_size,
        spec.dtype,
        raw_kv_shape,
    )
    x = 16 // raw_kv_cache.element_size()
    raw_key = raw_kv_cache[0].view(
        raw_pages, spec.num_kv_heads, spec.head_size // x, spec.block_size, x
    )
    raw_value = raw_kv_cache[1].view(
        raw_pages, spec.num_kv_heads, spec.head_size, spec.block_size
    )
    warm_key = take(
        warm_pages * spec.block_size * spec.num_kv_heads * spec.head_size,
        torch.int8,
        warm_k_shape,
    )
    warm_value = take(
        warm_pages * spec.block_size * spec.num_kv_heads * warm_v_head,
        torch.int8,
        warm_v_shape,
    )
    scale_bytes = warm_pages * spec.block_size * spec.num_kv_heads * 4
    warm_k_scale = take(scale_bytes, torch.float32, warm_scale_shape)
    warm_v_scale = take(scale_bytes, torch.float32, warm_scale_shape)

    canonical_role_by_logical_block = torch.full(
        (num_logical_blocks,),
        fill_value=-1,
        dtype=torch.int16,
        device=kv_raw_tensor.device,
    )
    canonical_slot_by_logical_block = torch.full(
        (num_logical_blocks,),
        fill_value=-1,
        dtype=torch.int32,
        device=kv_raw_tensor.device,
    )
    request_shape = (spec.tmh_max_num_seqs, spec.tmh_max_model_pages)
    request_slot_by_row_page = torch.full(
        request_shape,
        fill_value=-1,
        dtype=torch.int32,
        device=kv_raw_tensor.device,
    )
    return TMHPhysicalKVCache(
        spec=spec,
        num_logical_blocks=num_logical_blocks,
        raw_kv_cache=raw_kv_cache,
        raw_key=raw_key,
        raw_value=raw_value,
        warm_key=warm_key,
        warm_value=warm_value,
        warm_k_scale=warm_k_scale,
        warm_v_scale=warm_v_scale,
        canonical_role_by_logical_block=canonical_role_by_logical_block,
        canonical_slot_by_logical_block=canonical_slot_by_logical_block,
        request_slot_by_row_page=request_slot_by_row_page,
    )


def _recent_start_page(total_pages: int, hot_budget_pct: float) -> int:
    hot_ratio = max(0.0, min(1.0, hot_budget_pct / 100.0))
    hot_pages = (
        0
        if hot_ratio <= 0.0
        else min(total_pages, math.ceil(total_pages * hot_ratio))
    )
    return total_pages if hot_pages <= 0 else max(0, total_pages - hot_pages)


def tmh_rocm_native_raw_attention_args(
    cache: TMHPhysicalKVCache,
    attn_metadata,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return ROCm-native raw KV/block-table views when TMH has no warm pages."""
    max_seq_len = int(getattr(attn_metadata, "max_seq_len", 0) or 0)
    if max_seq_len <= 0:
        return None
    block_size = cache.spec.block_size
    max_pages = math.ceil(max_seq_len / block_size)
    if max_pages > cache.request_slot_by_row_page.shape[1]:
        return None
    if _recent_start_page(max_pages, cache.spec.tmh_hot_budget_pct) > 1:
        return None
    num_seqs = len(attn_metadata.seq_lens)
    block_table = cache.request_slot_by_row_page[
        :num_seqs, : attn_metadata.block_table.shape[1]
    ]
    return cache.raw_kv_cache, block_table


class TMHPhysicalRuntime:
    """Device-side TMH descriptor state for model runners."""

    def __init__(self) -> None:
        self._caches: dict[str, TMHPhysicalKVCache] = {}
        self._raw_free_slots: dict[str, list[int]] = {}
        self._warm_free_slots: dict[str, list[int]] = {}
        self._canonical_slots: dict[tuple[str, int, int], int] = {}
        self._overlay_slots: dict[tuple[str, str, int], int] = {}

    def register_cache(self, layer_name: str, cache: TMHPhysicalKVCache) -> None:
        self._caches[layer_name] = cache
        self._raw_free_slots[layer_name] = list(
            range(cache.raw_key.shape[0] - 1, -1, -1)
        )
        self._warm_free_slots[layer_name] = list(
            range(cache.warm_key.shape[0] - 1, -1, -1)
        )

    def apply_events(
        self,
        events: list[TMHPhysicalEvent] | None,
        req_id_to_index: dict[str, int],
    ) -> None:
        if not events:
            return
        for event in events:
            for req_id in event.released_request_ids:
                self.release_request(req_id)
            for descriptor in event.released_descriptors:
                self.release_descriptor(descriptor)
            if event.descriptors:
                self.release_request(event.request_id)
                self._clear_request_rows(event.descriptors, req_id_to_index)
            for descriptor in event.descriptors:
                cache = self._caches.get(descriptor.layer_name)
                if cache is None:
                    raise RuntimeError(
                        "TMH physical scheduler event targets layer "
                        f"{descriptor.layer_name!r}, but the worker has no "
                        "registered TMH physical cache for that layer."
                    )
                req_index = req_id_to_index.get(descriptor.request_id)
                if req_index is None:
                    raise RuntimeError(
                        "TMH physical descriptor targets request "
                        f"{descriptor.request_id!r}, but the worker has no "
                        "active request row for it."
                    )
                self._apply_descriptor(
                    cache,
                    descriptor.layer_name,
                    descriptor,
                    req_index,
                )

    def release_request(self, request_id: str) -> None:
        released = [
            key for key in self._overlay_slots
            if key[1] == request_id
        ]
        for key in released:
            layer_name, _, _ = key
            self._raw_free_slots[layer_name].append(self._overlay_slots.pop(key))

    def release_descriptor(self, descriptor: TMHPhysicalPageDescriptor) -> None:
        if descriptor.storage == TMHStorageKind.REQUEST_OVERLAY:
            key = (descriptor.layer_name, descriptor.request_id, descriptor.page_index)
            slot = self._overlay_slots.pop(key, None)
            if slot is not None:
                self._raw_free_slots[descriptor.layer_name].append(slot)
            return

        key = (
            descriptor.layer_name,
            descriptor.logical_block_id,
            int(descriptor.role),
        )
        slot = self._canonical_slots.pop(key, None)
        if slot is None:
            return
        wants_raw = descriptor.role in (TMHPageRole.PINNED_RAW, TMHPageRole.HOT_RAW)
        free_slots = (
            self._raw_free_slots[descriptor.layer_name]
            if wants_raw
            else self._warm_free_slots[descriptor.layer_name]
        )
        free_slots.append(slot)
        self._refresh_logical_descriptor(
            descriptor.layer_name,
            descriptor.logical_block_id,
        )

    def _refresh_logical_descriptor(
        self,
        layer_name: str,
        logical_block_id: int,
    ) -> None:
        cache = self._caches[layer_name]
        replacement = next(
            (
                (role, slot)
                for (candidate_layer, candidate_block, role), slot
                in self._canonical_slots.items()
                if candidate_layer == layer_name
                and candidate_block == logical_block_id
            ),
            None,
        )
        if replacement is None:
            cache.canonical_role_by_logical_block[logical_block_id] = -1
            cache.canonical_slot_by_logical_block[logical_block_id] = -1
            return
        role, slot = replacement
        cache.canonical_role_by_logical_block[logical_block_id] = role
        cache.canonical_slot_by_logical_block[logical_block_id] = slot

    def _clear_request_rows(
        self,
        descriptors: tuple[TMHPhysicalPageDescriptor, ...],
        req_id_to_index: dict[str, int],
    ) -> None:
        cleared: set[tuple[str, int]] = set()
        for descriptor in descriptors:
            req_index = req_id_to_index.get(descriptor.request_id)
            if req_index is None:
                raise RuntimeError(
                    "TMH physical descriptor targets request "
                    f"{descriptor.request_id!r}, but the worker has no active "
                    "request row for it."
                )
            key = (descriptor.layer_name, req_index)
            if key in cleared:
                continue
            cache = self._caches.get(descriptor.layer_name)
            if cache is None:
                raise RuntimeError(
                    "TMH physical scheduler event targets layer "
                    f"{descriptor.layer_name!r}, but the worker has no "
                    "registered TMH physical cache for that layer."
                )
            cache.request_slot_by_row_page[req_index].fill_(-1)
            cleared.add(key)

    def _apply_descriptor(self, cache, layer_name, descriptor, req_index: int) -> None:
        logical_block_id = descriptor.logical_block_id
        if logical_block_id < 0 or logical_block_id >= cache.num_logical_blocks:
            raise RuntimeError(
                f"TMH logical block id {logical_block_id} is outside the "
                f"allocated descriptor table ({cache.num_logical_blocks})."
            )
        if descriptor.storage == TMHStorageKind.REQUEST_OVERLAY:
            slot = self._overlay_slot(layer_name, descriptor.request_id, descriptor.page_index)
        else:
            slot = self._canonical_slot(layer_name, logical_block_id, descriptor.role)

        page_index = descriptor.page_index
        if page_index >= cache.request_slot_by_row_page.shape[1]:
            raise RuntimeError(
                f"TMH page index {page_index} exceeds descriptor row width "
                f"{cache.request_slot_by_row_page.shape[1]} for layer {layer_name!r}."
            )
        cache.request_slot_by_row_page[req_index, page_index] = slot

    def _canonical_slot(
        self,
        layer_name: str,
        logical_block_id: int,
        role: TMHPageRole,
    ) -> int:
        key = (layer_name, logical_block_id, int(role))
        slot = self._canonical_slots.get(key)
        if slot is not None:
            return slot
        wants_raw = role in (TMHPageRole.PINNED_RAW, TMHPageRole.HOT_RAW)
        slot = self._take_slot(layer_name, wants_raw)
        self._canonical_slots[key] = slot
        cache = self._caches[layer_name]
        cache.canonical_role_by_logical_block[logical_block_id] = int(role)
        cache.canonical_slot_by_logical_block[logical_block_id] = slot
        return slot

    def _overlay_slot(self, layer_name: str, request_id: str, page_index: int) -> int:
        key = (layer_name, request_id, page_index)
        slot = self._overlay_slots.get(key)
        if slot is not None:
            return slot
        slot = self._take_slot(layer_name, wants_raw=True)
        self._overlay_slots[key] = slot
        return slot

    def _take_slot(self, layer_name: str, wants_raw: bool) -> int:
        free_slots = (
            self._raw_free_slots[layer_name]
            if wants_raw
            else self._warm_free_slots[layer_name]
        )
        if not free_slots:
            raise RuntimeError(
                f"TMH physical {('raw' if wants_raw else 'warm')} pool for "
                f"layer {layer_name!r} is exhausted. Increase the hot budget "
                "reserve or reduce concurrency/max context."
            )
        return free_slots.pop()


def build_tmh_physical_runtime(
    kv_caches: dict[str, object],
) -> TMHPhysicalRuntime | None:
    runtime = TMHPhysicalRuntime()
    found = False
    for layer_name, cache in kv_caches.items():
        if isinstance(cache, TMHPhysicalKVCache):
            runtime.register_cache(layer_name, cache)
            found = True
    return runtime if found else None
