# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable
import os
import time
import torch

from vllm.logger import init_logger

from vllm.v1.core.tmh_policy import (
    TMHPageRole,
    TMH_EVENT_SCHEMA_VERSION,
    TMHMaterializationState,
    TMHPhysicalEvent,
    TMHPhysicalEventKind,
    TMHPhysicalPageDescriptor,
    TMHStorageKind,
    TMHStorageLocation,
)
from vllm.v1.kv_cache_interface import TMHFullAttentionSpec

logger = init_logger(__name__)
_TMH_RUNTIME_TIMING_LOG_COUNT = 0


def _tmh_runtime_timing_enabled() -> bool:
    return os.getenv("VLLM_TMH_LOG_RUNTIME_TIMING", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _tmh_runtime_timing_limit() -> int:
    try:
        return int(os.getenv("VLLM_TMH_RUNTIME_TIMING_LIMIT", "256"))
    except ValueError:
        return 256


def _tmh_runtime_timing_sync() -> None:
    if os.getenv("VLLM_TMH_RUNTIME_TIMING_SYNC", "0").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _tmh_log_runtime_timing(*, stage: str, elapsed_ms: float, num_events: int, num_descriptors: int) -> None:
    global _TMH_RUNTIME_TIMING_LOG_COUNT
    if not _tmh_runtime_timing_enabled():
        return
    if _TMH_RUNTIME_TIMING_LOG_COUNT >= _tmh_runtime_timing_limit():
        return
    _TMH_RUNTIME_TIMING_LOG_COUNT += 1
    logger.info(
        "TMH runtime timing: stage=%s elapsed_ms=%.3f num_events=%d num_descriptors=%d",
        stage, elapsed_ms, num_events, num_descriptors,
    )


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
    canonical_generation_by_logical_block: torch.Tensor
    request_slot_by_row_page: torch.Tensor
    request_role_by_row_page: torch.Tensor
    request_valid_tokens_by_row_page: torch.Tensor
    request_materialized_by_row_page: torch.Tensor
    request_slot_generation_by_row_page: torch.Tensor
    raw_slot_generation: torch.Tensor
    warm_slot_generation: torch.Tensor
    native_block_table_by_seq: torch.Tensor
    native_block_valid_by_seq: torch.Tensor
    native_block_table_gather: torch.Tensor
    native_seq_to_request_row: torch.Tensor
    gathered_role_by_seq_page: torch.Tensor
    gathered_slot_by_seq_page: torch.Tensor
    page_index_by_model_page: torch.Tensor
    identity_scale: torch.Tensor

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
    if spec.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            "TMH physical cache supports raw torch.float16 and torch.bfloat16 "
            f"only; received {spec.dtype}."
        )
    if not spec.tmh_late_layer and spec.head_size_v % 2:
        raise ValueError(
            "TMH early-layer INT4 values require an even head_size_v; "
            f"received {spec.head_size_v}."
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

    canonical_role_by_logical_block = take(
        num_logical_blocks * 2, torch.int16, (num_logical_blocks,)
    ).fill_(-1)
    canonical_slot_by_logical_block = take(
        num_logical_blocks * 4, torch.int32, (num_logical_blocks,)
    ).fill_(-1)
    canonical_generation_by_logical_block = take(
        num_logical_blocks * 8, torch.int64, (num_logical_blocks,)
    ).fill_(-1)
    request_shape = (spec.tmh_max_num_seqs, spec.tmh_max_model_pages)
    request_elements = request_shape[0] * request_shape[1]
    request_slot_by_row_page = take(
        request_elements * 4, torch.int32, request_shape
    ).fill_(-1)
    request_role_by_row_page = take(
        request_elements * 2, torch.int16, request_shape
    ).fill_(-1)
    request_valid_tokens_by_row_page = take(
        request_elements * 2, torch.int16, request_shape
    ).zero_()
    request_materialized_by_row_page = take(
        request_elements, torch.bool, request_shape
    ).zero_()
    request_slot_generation_by_row_page = take(
        request_elements * 8, torch.int64, request_shape
    ).fill_(-1)
    raw_slot_generation = take(
        raw_pages * 8, torch.int64, (raw_pages,)
    ).zero_()
    warm_slot_generation = take(
        warm_pages * 8, torch.int64, (warm_pages,)
    ).zero_()
    native_block_table_by_seq = take(
        request_elements * 4, torch.int32, request_shape
    ).fill_(-1)
    native_block_valid_by_seq = take(
        request_elements, torch.bool, request_shape
    ).zero_()
    native_block_table_gather = take(
        request_elements * 4, torch.int32, request_shape
    )
    native_seq_to_request_row = take(
        spec.tmh_max_num_seqs * 8,
        torch.int64,
        (spec.tmh_max_num_seqs,),
    )
    gathered_role_by_seq_page = take(
        request_elements * 2, torch.int16, request_shape
    )
    gathered_slot_by_seq_page = take(
        request_elements * 4, torch.int32, request_shape
    )
    page_index_by_model_page = take(
        spec.tmh_max_model_pages * 4,
        torch.int32,
        (spec.tmh_max_model_pages,),
    )
    page_index_by_model_page.copy_(
        torch.arange(
            spec.tmh_max_model_pages,
            dtype=torch.int32,
            device=kv_raw_tensor.device,
        )
    )
    identity_scale = take(4, torch.float32, ()).fill_(1)
    if offset != spec.physical_allocation_bytes(num_logical_blocks):
        raise RuntimeError(
            "TMH physical memory ledger disagrees with the carved allocation: "
            f"ledger={spec.physical_allocation_bytes(num_logical_blocks)}, "
            f"carved={offset}"
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
        canonical_generation_by_logical_block=canonical_generation_by_logical_block,
        request_slot_by_row_page=request_slot_by_row_page,
        request_role_by_row_page=request_role_by_row_page,
        request_valid_tokens_by_row_page=request_valid_tokens_by_row_page,
        request_materialized_by_row_page=request_materialized_by_row_page,
        request_slot_generation_by_row_page=request_slot_generation_by_row_page,
        raw_slot_generation=raw_slot_generation,
        warm_slot_generation=warm_slot_generation,
        native_block_table_by_seq=native_block_table_by_seq,
        native_block_valid_by_seq=native_block_valid_by_seq,
        native_block_table_gather=native_block_table_gather,
        native_seq_to_request_row=native_seq_to_request_row,
        gathered_role_by_seq_page=gathered_role_by_seq_page,
        gathered_slot_by_seq_page=gathered_slot_by_seq_page,
        page_index_by_model_page=page_index_by_model_page,
        identity_scale=identity_scale,
    )


@dataclass(frozen=True)
class _TMHAllocation:
    role: TMHPageRole
    slot: int
    slot_generation: int

    @property
    def raw(self) -> bool:
        return self.role in (TMHPageRole.PINNED_RAW, TMHPageRole.HOT_RAW)


@dataclass(frozen=True)
class _TMHBinding:
    descriptor: TMHPhysicalPageDescriptor
    allocation: _TMHAllocation


@dataclass(frozen=True)
class _TMHStagedBinding:
    descriptor: TMHPhysicalPageDescriptor
    allocation: _TMHAllocation
    source: _TMHBinding | None
    newly_allocated: bool


def _round_half_away_from_zero(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.floor(torch.abs(value) + 0.5)


def quantize_tmh_int4(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference affine INT4 contract used by migration and tests.

    The range always contains zero. NaN/Inf are rejected instead of being
    converted into an apparently valid but corrupt page.
    """

    if values.shape[-1] % 2:
        raise ValueError("TMH INT4 packing requires an even final dimension")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("TMH INT4 quantization rejects NaN or Inf values")
    values_f = values.float()
    zeros = torch.zeros_like(values_f[..., :1])
    lower = torch.minimum(values_f.amin(dim=-1, keepdim=True), zeros)
    upper = torch.maximum(values_f.amax(dim=-1, keepdim=True), zeros)
    span = upper - lower
    scale = torch.where(span > 0, span / 15.0, torch.ones_like(span))
    zero_point = _round_half_away_from_zero(-lower / scale).clamp(0, 15)
    quantized = _round_half_away_from_zero(values_f / scale + zero_point).clamp(
        0, 15
    )
    even = quantized[..., 0::2].to(torch.uint8)
    odd = quantized[..., 1::2].to(torch.uint8)
    packed = (even | (odd << 4)).to(torch.int8)
    return packed, scale.squeeze(-1), zero_point.squeeze(-1).to(torch.int32)


def dequantize_tmh_int4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    low = (packed.to(torch.uint8) & 0xF).float()
    high = ((packed.to(torch.uint8) >> 4) & 0xF).float()
    unpacked = torch.stack((low, high), dim=-1).flatten(start_dim=-2)
    return (unpacked - zero_point[..., None].float()) * scale[..., None].float()


def _pack_scale_zero_point(scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    bits = scale.contiguous().view(torch.int32)
    packed_bits = (bits & -16) | (zero_point.to(torch.int32) & 0xF)
    return packed_bits.contiguous().view(torch.float32)


def _unpack_scale_zero_point(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bits = packed.contiguous().view(torch.int32)
    scale = (bits & -16).contiguous().view(torch.float32)
    zero_point = bits & 0xF
    return scale, zero_point


class TMHPhysicalRuntime:
    """Transactional physical-page state shared by TMH writers and readers."""

    def __init__(
        self,
        failure_injector: Callable[[str], None] | None = None,
    ) -> None:
        self._caches: dict[str, TMHPhysicalKVCache] = {}
        self._raw_free_slots: dict[str, list[int]] = {}
        self._warm_free_slots: dict[str, list[int]] = {}
        self._raw_slot_generations: dict[str, list[int]] = {}
        self._warm_slot_generations: dict[str, list[int]] = {}
        self._canonical_allocations: dict[
            tuple[str, int, int, int], _TMHAllocation
        ] = {}
        self._canonical_binding_keys: dict[
            tuple[str, int, int, int], set[tuple[str, str, int]]
        ] = {}
        self._canonical_resident_key: dict[
            tuple[str, int, int], tuple[str, int, int, int]
        ] = {}
        self._pending_canonical_releases: dict[
            tuple[str, int, int, int], TMHPhysicalPageDescriptor
        ] = {}
        self._overlay_allocations: dict[
            tuple[str, str, int, int], _TMHAllocation
        ] = {}
        self._request_binding_keys: dict[
            str, set[tuple[str, str, int]]
        ] = {}
        self._request_bindings: dict[tuple[str, str, int], _TMHBinding] = {}
        self._request_rows: dict[tuple[str, str], int] = {}
        self._request_total_pages: dict[str, int] = {}
        self._request_versions: dict[str, tuple[int, int, int]] = {}
        self._last_request_generation: dict[str, int] = {}
        self._committed_ids: set[str] = set()
        self._failure_injector = failure_injector
        self.counters: dict[str, int] = {
            "events_committed": 0,
            "events_duplicate": 0,
            "events_rejected": 0,
            "migrations": 0,
            "migration_bytes": 0,
            "overlay_pages": 0,
            "failed_admissions": 0,
            "slot_generation_mismatch": 0,
            "quantized_values": 0,
            "quantization_saturated_values": 0,
            "metadata_descriptors_published": 0,
            "metadata_bytes_published": 0,
            "metadata_logical_updates": 0,
        }

    def register_cache(self, layer_name: str, cache: TMHPhysicalKVCache) -> None:
        self._caches[layer_name] = cache
        self._raw_free_slots[layer_name] = list(
            range(cache.raw_key.shape[0] - 1, -1, -1)
        )
        self._warm_free_slots[layer_name] = list(
            range(cache.warm_key.shape[0] - 1, -1, -1)
        )
        self._raw_slot_generations[layer_name] = [0] * cache.raw_key.shape[0]
        self._warm_slot_generations[layer_name] = [0] * cache.warm_key.shape[0]

    def batch_is_all_raw(self, request_ids: list[str]) -> bool:
        """Host-side regime bit maintained by committed page metadata.

        Model runners pass this bit into steady-state dispatch so choosing the
        native/raw family never reads a device condition back to the host.
        """

        active = set(request_ids)
        saw_page = False
        for (request_id, _layer_name, _page), binding in self._request_bindings.items():
            if request_id not in active:
                continue
            saw_page = True
            if not binding.allocation.raw:
                return False
        return saw_page

    def diagnostics(self) -> dict[str, object]:
        pages_by_representation = {role.name: 0 for role in TMHPageRole}
        for binding in self._request_bindings.values():
            pages_by_representation[binding.allocation.role.name] += 1
        pools = {
            layer_name: {
                "raw_capacity": cache.raw_key.shape[0],
                "raw_free": len(self._raw_free_slots[layer_name]),
                "warm_capacity": cache.warm_key.shape[0],
                "warm_free": len(self._warm_free_slots[layer_name]),
            }
            for layer_name, cache in self._caches.items()
        }
        return {
            "pages_by_representation": pages_by_representation,
            "valid_pages": len(self._request_bindings),
            "unmaterialized_pages": 0,
            "transitional_pages": 0,
            "overlay_pages": len(self._overlay_allocations),
            "prefix_canonical_pages": len(self._canonical_allocations),
            "pools": pools,
            "counters": self.counters.copy(),
        }

    def prepare_graph_capture(self, num_rows: int) -> tuple[str, ...]:
        """Install safe all-raw dummy mappings at stable request-table rows."""

        if num_rows < 1:
            return ()
        max_rows = min(
            cache.request_slot_by_row_page.shape[0]
            for cache in self._caches.values()
        )
        if num_rows > max_rows:
            raise RuntimeError(
                f"TMH graph capture requests {num_rows} rows but only "
                f"{max_rows} are configured"
            )
        request_ids: list[str] = []
        for row in range(num_rows):
            request_id = f"_tmh_graph_capture_{row}"
            descriptors = tuple(
                TMHPhysicalPageDescriptor(
                    request_id=request_id,
                    layer_name=layer_name,
                    logical_block_id=row % cache.num_logical_blocks,
                    page_index=0,
                    role=TMHPageRole.HOT_RAW,
                    storage=TMHStorageKind.REQUEST_OVERLAY,
                    prefix_cached=True,
                    k_quant_mode="raw",
                    v_quant_mode="raw",
                    allocation_generation=0,
                    valid_tokens=cache.spec.block_size,
                )
                for layer_name, cache in self._caches.items()
            )
            self.apply_events(
                [
                    TMHPhysicalEvent(
                        schema_version=TMH_EVENT_SCHEMA_VERSION,
                        event_kind=TMHPhysicalEventKind.DELTA,
                        request_id=request_id,
                        request_generation=1,
                        sequence=1,
                        expected_base_version=0,
                        target_version=1,
                        commit_id=f"{request_id}:1:1:graph-capture",
                        descriptors=descriptors,
                        total_pages=1,
                        recent_start_page=0,
                        hot_pages=1,
                    )
                ],
                {request_id: row},
            )
            request_ids.append(request_id)
        return tuple(request_ids)

    def finish_graph_capture(self, request_ids: tuple[str, ...]) -> None:
        """Release graph-capture overlays after graph construction."""

        for request_id in request_ids:
            self.apply_events(
                [
                    TMHPhysicalEvent(
                        schema_version=TMH_EVENT_SCHEMA_VERSION,
                        event_kind=TMHPhysicalEventKind.RELEASE,
                        request_id=request_id,
                        request_generation=1,
                        sequence=2,
                        expected_base_version=1,
                        target_version=2,
                        commit_id=f"{request_id}:1:2:graph-capture-release",
                        descriptors=(),
                        total_pages=0,
                        recent_start_page=0,
                        hot_pages=0,
                        released_request_ids=(request_id,),
                    )
                ],
                {},
            )

    def apply_events(
        self,
        events: list[TMHPhysicalEvent] | None,
        req_id_to_index: dict[str, int],
    ) -> None:
        if not events:
            return
        timing_enabled = _tmh_runtime_timing_enabled()
        apply_start = None
        if timing_enabled:
            _tmh_runtime_timing_sync()
            apply_start = time.perf_counter()
        total_descriptors = sum(
            len(event.descriptors) + len(event.released_descriptors)
            for event in events
        )
        try:
            for event in events:
                if event.commit_id in self._committed_ids:
                    self.counters["events_duplicate"] += 1
                    continue
                snapshot = self._snapshot() if self._failure_injector is not None else None
                staged_new: list[tuple[str, _TMHAllocation]] = []
                try:
                    req_index = self._validate_event(event, req_id_to_index)
                    self._checkpoint("validated")
                    staged, pending_canonical, pending_overlay = self._stage_event(
                        event, staged_new
                    )
                    self._checkpoint("reserved")
                    self._materialize_staged(staged)
                    self._checkpoint("materialized")
                    self._publish_event(
                        event,
                        req_index,
                        staged,
                        pending_canonical,
                        pending_overlay,
                    )
                    self._checkpoint("published")
                    self._release_after_commit(event, staged)
                    self._checkpoint("released")
                    self._request_versions[event.request_id] = (
                        event.request_generation,
                        event.sequence,
                        event.target_version,
                    )
                    self._last_request_generation[event.request_id] = max(
                        event.request_generation,
                        self._last_request_generation.get(event.request_id, 0),
                    )
                    if event.event_kind == TMHPhysicalEventKind.RELEASE:
                        self._request_versions.pop(event.request_id, None)
                    self._committed_ids.add(event.commit_id)
                    self.counters["events_committed"] += 1
                except Exception:
                    self.counters["events_rejected"] += 1
                    if snapshot is not None:
                        self._restore(snapshot)
                    else:
                        for layer_name, allocation in reversed(staged_new):
                            self._return_slot(layer_name, allocation)
                    raise
        finally:
            if apply_start is not None:
                _tmh_runtime_timing_sync()
                _tmh_log_runtime_timing(
                    stage="apply_events",
                    elapsed_ms=(time.perf_counter() - apply_start) * 1000.0,
                    num_events=len(events),
                    num_descriptors=total_descriptors,
                )


    def _validate_event(
        self,
        event: TMHPhysicalEvent,
        req_id_to_index: dict[str, int],
    ) -> int | None:
        if event.schema_version != TMH_EVENT_SCHEMA_VERSION:
            raise RuntimeError(
                f"unsupported TMH event schema {event.schema_version}; "
                f"expected {TMH_EVENT_SCHEMA_VERSION}"
            )
        if not event.commit_id:
            raise RuntimeError("TMH physical event commit_id must be non-empty")
        if event.sequence < 1 or event.target_version != event.expected_base_version + 1:
            raise RuntimeError("TMH physical event has a malformed version transition")
        missing_dependencies = set(event.dependency_commit_ids) - self._committed_ids
        if missing_dependencies:
            raise RuntimeError(
                "TMH physical event has uncommitted dependencies: "
                f"{sorted(missing_dependencies)}"
            )
        if not event.planner_policy:
            raise RuntimeError("TMH physical event planner_policy must be non-empty")
        current = self._request_versions.get(event.request_id)
        if current is None:
            last_generation = self._last_request_generation.get(event.request_id, 0)
            if (
                event.request_generation <= last_generation
                or event.sequence != 1
                or event.expected_base_version != 0
            ):
                raise RuntimeError(
                    "stale or out-of-order TMH event for inactive request "
                    f"{event.request_id!r}: generation={event.request_generation}, "
                    f"sequence={event.sequence}, base={event.expected_base_version}"
                )
        else:
            generation, sequence, version = current
            if (
                event.request_generation != generation
                or event.sequence != sequence + 1
                or event.expected_base_version != version
            ):
                raise RuntimeError(
                    "stale or out-of-order TMH event for request "
                    f"{event.request_id!r}: expected generation={generation}, "
                    f"sequence={sequence + 1}, base={version}"
                )
        if event.total_pages < 0:
            raise RuntimeError("TMH event total_pages cannot be negative")
        if event.event_kind == TMHPhysicalEventKind.RELEASE:
            if event.descriptors or event.total_pages != 0:
                raise RuntimeError("TMH release events cannot publish page descriptors")
            return None
        req_index = req_id_to_index.get(event.request_id)
        if req_index is None:
            raise RuntimeError(
                f"TMH event targets request {event.request_id!r} without an active row"
            )
        seen: set[tuple[str, int]] = set()
        for descriptor in event.descriptors:
            cache = self._caches.get(descriptor.layer_name)
            if cache is None:
                raise RuntimeError(
                    f"TMH event targets unregistered layer {descriptor.layer_name!r}"
                )
            if descriptor.request_id != event.request_id:
                raise RuntimeError("TMH descriptor request_id disagrees with its event")
            if not (0 <= req_index < cache.request_slot_by_row_page.shape[0]):
                raise RuntimeError(f"TMH request row {req_index} is outside the table")
            if not (0 <= descriptor.page_index < event.total_pages):
                raise RuntimeError(
                    f"TMH descriptor page {descriptor.page_index} is outside total_pages "
                    f"{event.total_pages}"
                )
            if descriptor.page_index >= cache.request_slot_by_row_page.shape[1]:
                raise RuntimeError("TMH descriptor page exceeds configured model pages")
            if not (0 <= descriptor.logical_block_id < cache.num_logical_blocks):
                raise RuntimeError("TMH descriptor logical block id is out of range")
            if not (1 <= descriptor.valid_tokens <= cache.spec.block_size):
                raise RuntimeError("TMH descriptor valid_tokens is out of range")
            expected_modes = {
                TMHPageRole.PINNED_RAW: ("raw", "raw"),
                TMHPageRole.HOT_RAW: ("raw", "raw"),
                TMHPageRole.WARM_INT8_INT4: (
                    "int8_per_token_head",
                    "int4_per_token_head",
                ),
                TMHPageRole.WARM_INT8_INT8: (
                    "int8_per_token_head",
                    "int8_per_token_head",
                ),
            }[descriptor.role]
            if (descriptor.k_quant_mode, descriptor.v_quant_mode) != expected_modes:
                raise RuntimeError("TMH descriptor role and encoding disagree")
            if descriptor.storage == TMHStorageKind.REQUEST_OVERLAY and not descriptor.raw:
                raise RuntimeError("TMH request overlays must use raw storage")
            if descriptor.storage_location != TMHStorageLocation.DEVICE:
                raise RuntimeError(
                    "TMH CPU-offload and recompute locations are not implemented"
                )
            if descriptor.materialization_state != TMHMaterializationState.READY:
                raise RuntimeError(
                    "TMH events may publish READY target descriptors only"
                )
            identity = (descriptor.layer_name, descriptor.page_index)
            if identity in seen:
                raise RuntimeError("TMH event contains a duplicate layer/page descriptor")
            seen.add(identity)
        return req_index

    def _stage_event(
        self,
        event: TMHPhysicalEvent,
        staged_new: list[tuple[str, _TMHAllocation]],
    ) -> tuple[
        list[_TMHStagedBinding],
        dict[tuple[str, int, int, int], _TMHAllocation],
        dict[tuple[str, str, int, int], _TMHAllocation],
    ]:
        pending_canonical: dict[tuple[str, int, int, int], _TMHAllocation] = {}
        pending_overlay: dict[tuple[str, str, int, int], _TMHAllocation] = {}
        staged: list[_TMHStagedBinding] = []
        for descriptor in event.descriptors:
            cache = self._caches[descriptor.layer_name]
            if descriptor.role in (
                TMHPageRole.WARM_INT8_INT4,
                TMHPageRole.WARM_INT8_INT8,
            ):
                expected_role = (
                    TMHPageRole.WARM_INT8_INT8
                    if cache.spec.tmh_late_layer
                    else TMHPageRole.WARM_INT8_INT4
                )
                if descriptor.role != expected_role:
                    raise RuntimeError(
                        "TMH policy/cache fidelity mismatch for "
                        f"{descriptor.layer_name!r}: policy={descriptor.role.name}, "
                        f"cache_late_layer={cache.spec.tmh_late_layer}, "
                        f"warm_value_width={cache.warm_value.shape[-1]}."
                    )
            binding_key = (
                descriptor.request_id,
                descriptor.layer_name,
                descriptor.page_index,
            )
            source = self._request_bindings.get(binding_key)
            if descriptor.storage == TMHStorageKind.CANONICAL:
                key = self._canonical_key(descriptor)
                allocation = pending_canonical.get(key)
                if allocation is None:
                    allocation = self._canonical_allocations.get(key)
                target_map = pending_canonical
            else:
                key = self._overlay_key(descriptor, event.request_generation)
                allocation = pending_overlay.get(key)
                if allocation is None:
                    allocation = self._overlay_allocations.get(key)
                target_map = pending_overlay
            newly_allocated = allocation is None or allocation.role != descriptor.role
            if newly_allocated:
                allocation = self._take_slot(descriptor.layer_name, descriptor.role)
                staged_new.append((descriptor.layer_name, allocation))
                target_map[key] = allocation
            staged.append(
                _TMHStagedBinding(
                    descriptor=descriptor,
                    allocation=allocation,
                    source=source,
                    newly_allocated=newly_allocated,
                )
            )
        return staged, pending_canonical, pending_overlay

    def _materialize_staged(self, staged: list[_TMHStagedBinding]) -> None:
        transitioned = False
        for item in staged:
            cache = self._caches[item.descriptor.layer_name]
            if item.newly_allocated:
                self._clear_slot(cache, item.allocation)
                if item.source is not None:
                    valid_tokens = min(
                        item.source.descriptor.valid_tokens,
                        item.descriptor.valid_tokens,
                    )
                    key, value = self._read_page(
                        cache, item.source.allocation, valid_tokens
                    )
                    self._write_page(
                        cache, item.allocation, key, value, valid_tokens
                    )
                    transitioned = True
                    self.counters["migrations"] += 1
                    self.counters["migration_bytes"] += (
                        key.numel() + value.numel()
                    ) * key.element_size()
            elif item.source is not None and (
                item.descriptor.valid_tokens < item.source.descriptor.valid_tokens
            ):
                self._zero_page_tail(
                    cache,
                    item.allocation,
                    item.descriptor.valid_tokens,
                )
        if transitioned:
            cuda_devices = {
                self._caches[item.descriptor.layer_name].device
                for item in staged
                if self._caches[item.descriptor.layer_name].device.type == "cuda"
            }
            for cuda_device in cuda_devices:
                torch.cuda.synchronize(cuda_device)

    def _publish_event(
        self,
        event: TMHPhysicalEvent,
        req_index: int | None,
        staged: list[_TMHStagedBinding],
        pending_canonical: dict[tuple[str, int, int, int], _TMHAllocation],
        pending_overlay: dict[tuple[str, str, int, int], _TMHAllocation],
    ) -> None:
        self._canonical_allocations.update(pending_canonical)
        self._overlay_allocations.update(pending_overlay)
        if event.event_kind == TMHPhysicalEventKind.RELEASE:
            for request_id in event.released_request_ids or (event.request_id,):
                self.release_request(request_id)
            return
        assert req_index is not None
        request_publications: dict[
            str, list[tuple[TMHPhysicalPageDescriptor, _TMHAllocation]]
        ] = {}
        valid_token_publications: dict[
            str, list[TMHPhysicalPageDescriptor]
        ] = {}
        for item in staged:
            descriptor = item.descriptor
            cache = self._caches[descriptor.layer_name]
            self._request_rows[(descriptor.request_id, descriptor.layer_name)] = req_index
            binding_key = (
                descriptor.request_id,
                descriptor.layer_name,
                descriptor.page_index,
            )
            previous_binding = self._request_bindings.get(binding_key)
            if previous_binding is not None:
                self._unregister_binding(binding_key, previous_binding)
            self._request_bindings[binding_key] = _TMHBinding(
                descriptor=descriptor,
                allocation=item.allocation,
            )
            self._register_binding(binding_key, descriptor)
            mapping_changed = (
                item.source is None
                or item.source.allocation != item.allocation
                or item.source.descriptor.role != descriptor.role
                or item.source.descriptor.storage != descriptor.storage
                or item.source.descriptor.allocation_generation
                != descriptor.allocation_generation
            )
            if mapping_changed:
                request_publications.setdefault(descriptor.layer_name, []).append(
                    (descriptor, item.allocation)
                )
            else:
                self.counters["metadata_logical_updates"] += 1
                if (
                    item.source is not None
                    and descriptor.valid_tokens
                    < item.source.descriptor.valid_tokens
                ):
                    valid_token_publications.setdefault(
                        descriptor.layer_name, []
                    ).append(descriptor)
            if mapping_changed and descriptor.storage == TMHStorageKind.CANONICAL:
                key = self._canonical_key(descriptor)
                resident_key = self._canonical_resident_identity(descriptor)
                self._canonical_resident_key[resident_key] = key
                cache.canonical_role_by_logical_block[
                    descriptor.logical_block_id
                ] = int(descriptor.role)
                cache.canonical_slot_by_logical_block[
                    descriptor.logical_block_id
                ] = item.allocation.slot
                cache.canonical_generation_by_logical_block[
                    descriptor.logical_block_id
                ] = descriptor.allocation_generation
                for other_key in tuple(self._canonical_binding_keys.get(key, ())):
                    if other_key == binding_key:
                        continue
                    other_binding = self._request_bindings.get(other_key)
                    if other_binding is None:
                        continue
                    other_descriptor = replace(
                        other_binding.descriptor,
                        role=descriptor.role,
                        k_quant_mode=descriptor.k_quant_mode,
                        v_quant_mode=descriptor.v_quant_mode,
                    )
                    self._request_bindings[other_key] = _TMHBinding(
                        other_descriptor, item.allocation
                    )
                    other_row = self._request_rows[(other_key[0], other_key[1])]
                    self._publish_request_page(
                        cache, other_row, other_descriptor, item.allocation
                    )
        for layer_name, publications in request_publications.items():
            self._publish_request_pages(
                self._caches[layer_name], req_index, publications
            )
        for layer_name, publications in valid_token_publications.items():
            self._publish_valid_tokens(
                self._caches[layer_name], req_index, publications
            )
        self._truncate_request(event.request_id, event.total_pages)

    def _release_after_commit(
        self,
        event: TMHPhysicalEvent,
        staged: list[_TMHStagedBinding],
    ) -> None:
        for item in staged:
            source = item.source
            if source is None or source.allocation == item.allocation:
                continue
            if source.descriptor.storage == TMHStorageKind.REQUEST_OVERLAY:
                old_key = self._overlay_key(
                    source.descriptor, event.request_generation
                )
                if self._overlay_allocations.get(old_key) == source.allocation:
                    self._overlay_allocations.pop(old_key, None)
                    self._return_slot(source.descriptor.layer_name, source.allocation)
            elif self._canonical_key(source.descriptor) == self._canonical_key(
                item.descriptor
            ):
                self._return_slot(source.descriptor.layer_name, source.allocation)
        for descriptor in event.released_descriptors:
            self.release_descriptor(descriptor)
        self._flush_pending_canonical_releases()
        self.counters["overlay_pages"] = len(self._overlay_allocations)

    def _publish_request_page(
        self,
        cache: TMHPhysicalKVCache,
        req_index: int,
        descriptor: TMHPhysicalPageDescriptor,
        allocation: _TMHAllocation,
    ) -> None:
        page = descriptor.page_index
        cache.request_slot_by_row_page[req_index, page] = allocation.slot
        cache.request_role_by_row_page[req_index, page] = int(descriptor.role)
        cache.request_valid_tokens_by_row_page[req_index, page] = descriptor.valid_tokens
        cache.request_slot_generation_by_row_page[
            req_index, page
        ] = allocation.slot_generation
        cache.request_materialized_by_row_page[req_index, page] = True
        cache.native_block_table_by_seq[req_index, page] = (
            allocation.slot if allocation.raw else -1
        )
        cache.native_block_valid_by_seq[req_index, page] = allocation.raw


    def _publish_request_pages(
        self,
        cache: TMHPhysicalKVCache,
        req_index: int,
        publications: list[
            tuple[TMHPhysicalPageDescriptor, _TMHAllocation]
        ],
    ) -> None:
        """Publish one layer's touched delta with compact device tensors."""

        if not publications:
            return
        if len(publications) == 1:
            descriptor, allocation = publications[0]
            self._publish_request_page(cache, req_index, descriptor, allocation)
            self.counters["metadata_descriptors_published"] += 1
            self.counters["metadata_bytes_published"] += 22
            return
        device = cache.device
        pages = torch.tensor(
            [descriptor.page_index for descriptor, _ in publications],
            dtype=torch.int64,
            device=device,
        )
        slots = torch.tensor(
            [allocation.slot for _, allocation in publications],
            dtype=torch.int32,
            device=device,
        )
        roles = torch.tensor(
            [int(descriptor.role) for descriptor, _ in publications],
            dtype=torch.int16,
            device=device,
        )
        valid_tokens = torch.tensor(
            [descriptor.valid_tokens for descriptor, _ in publications],
            dtype=torch.int16,
            device=device,
        )
        slot_generations = torch.tensor(
            [allocation.slot_generation for _, allocation in publications],
            dtype=torch.int64,
            device=device,
        )
        raw = torch.tensor(
            [allocation.raw for _, allocation in publications],
            dtype=torch.bool,
            device=device,
        )
        native_slots = torch.where(raw, slots, torch.full_like(slots, -1))
        cache.request_slot_by_row_page[req_index].index_copy_(0, pages, slots)
        cache.request_role_by_row_page[req_index].index_copy_(0, pages, roles)
        cache.request_valid_tokens_by_row_page[req_index].index_copy_(
            0, pages, valid_tokens
        )
        cache.request_slot_generation_by_row_page[req_index].index_copy_(
            0, pages, slot_generations
        )
        cache.request_materialized_by_row_page[req_index].index_fill_(
            0, pages, True
        )
        cache.native_block_table_by_seq[req_index].index_copy_(0, pages, native_slots)
        cache.native_block_valid_by_seq[req_index].index_copy_(0, pages, raw)
        self.counters["metadata_descriptors_published"] += len(publications)
        self.counters["metadata_bytes_published"] += len(publications) * 22

    @staticmethod
    def _publish_valid_tokens(
        cache: TMHPhysicalKVCache,
        req_index: int,
        descriptors: list[TMHPhysicalPageDescriptor],
    ) -> None:
        """Publish rare shrink updates; growth is fused into the KV writer."""

        if len(descriptors) == 1:
            descriptor = descriptors[0]
            cache.request_valid_tokens_by_row_page[
                req_index, descriptor.page_index
            ] = descriptor.valid_tokens
            return
        pages = torch.tensor(
            [descriptor.page_index for descriptor in descriptors],
            dtype=torch.int64,
            device=cache.device,
        )
        valid_tokens = torch.tensor(
            [descriptor.valid_tokens for descriptor in descriptors],
            dtype=torch.int16,
            device=cache.device,
        )
        cache.request_valid_tokens_by_row_page[req_index].index_copy_(
            0, pages, valid_tokens
        )

    def _truncate_request(self, request_id: str, total_pages: int) -> None:
        binding_keys = self._request_binding_keys.get(request_id)
        if not binding_keys:
            self._request_total_pages.pop(request_id, None)
            return
        previous_total_pages = self._request_total_pages.get(request_id)
        if previous_total_pages is not None and total_pages >= previous_total_pages:
            self._request_total_pages[request_id] = total_pages
            return
        if total_pages > 0:
            self._request_total_pages[request_id] = total_pages
        else:
            self._request_total_pages.pop(request_id, None)
        for key in tuple(binding_keys):
            if key[2] < total_pages:
                continue
            binding = self._request_bindings.pop(key, None)
            if binding is None:
                continue
            self._unregister_binding(key, binding)
            if binding.descriptor.storage == TMHStorageKind.REQUEST_OVERLAY:
                overlay_candidates = [
                    overlay_key
                    for overlay_key, allocation in self._overlay_allocations.items()
                    if allocation == binding.allocation
                    and overlay_key[0] == binding.descriptor.layer_name
                    and overlay_key[1] == request_id
                    and overlay_key[3] == binding.descriptor.page_index
                ]
                for overlay_key in overlay_candidates:
                    self._overlay_allocations.pop(overlay_key, None)
                self._return_slot(binding.descriptor.layer_name, binding.allocation)
            row = self._request_rows.get((request_id, binding.descriptor.layer_name))
            if row is not None:
                self._clear_request_page(
                    self._caches[binding.descriptor.layer_name], row, key[2]
                )

    def release_request(self, request_id: str) -> None:
        for key in tuple(self._request_binding_keys.get(request_id, ())):
            binding = self._request_bindings.pop(key, None)
            if binding is None:
                continue
            self._unregister_binding(key, binding)
            if binding.descriptor.storage == TMHStorageKind.REQUEST_OVERLAY:
                overlay_candidates = [
                    overlay_key
                    for overlay_key, allocation in self._overlay_allocations.items()
                    if overlay_key[1] == request_id and allocation == binding.allocation
                ]
                for overlay_key in overlay_candidates:
                    self._overlay_allocations.pop(overlay_key, None)
                self._return_slot(binding.descriptor.layer_name, binding.allocation)
        self._request_total_pages.pop(request_id, None)
        for (candidate_request, layer_name), row in list(self._request_rows.items()):
            if candidate_request != request_id:
                continue
            self._clear_request_row(self._caches[layer_name], row)
            self._request_rows.pop((candidate_request, layer_name), None)
        self._flush_pending_canonical_releases()

    def release_descriptor(self, descriptor: TMHPhysicalPageDescriptor) -> None:
        if descriptor.storage == TMHStorageKind.REQUEST_OVERLAY:
            return
        key = self._canonical_key(descriptor)
        allocation = self._canonical_allocations.get(key)
        if allocation is None:
            self._pending_canonical_releases.pop(key, None)
            return
        if self._canonical_binding_keys.get(key):
            self._pending_canonical_releases[key] = descriptor
            return
        self._pending_canonical_releases.pop(key, None)
        self._canonical_allocations.pop(key, None)
        self._return_slot(descriptor.layer_name, allocation)
        resident_key = self._canonical_resident_identity(descriptor)
        if self._canonical_resident_key.get(resident_key) == key:
            self._canonical_resident_key.pop(resident_key, None)
        cache = self._caches[descriptor.layer_name]
        block = descriptor.logical_block_id
        if cache.canonical_generation_by_logical_block[block].item() == (
            descriptor.allocation_generation
        ):
            cache.canonical_role_by_logical_block[block] = -1
            cache.canonical_slot_by_logical_block[block] = -1
            cache.canonical_generation_by_logical_block[block] = -1

    def _flush_pending_canonical_releases(self) -> None:
        for descriptor in list(self._pending_canonical_releases.values()):
            self.release_descriptor(descriptor)

    def _take_slot(
        self, layer_name: str, role: TMHPageRole
    ) -> _TMHAllocation:
        wants_raw = role in (TMHPageRole.PINNED_RAW, TMHPageRole.HOT_RAW)
        free_slots = (
            self._raw_free_slots[layer_name]
            if wants_raw
            else self._warm_free_slots[layer_name]
        )
        if not free_slots:
            self.counters["failed_admissions"] += 1
            raise RuntimeError(
                f"TMH physical {('raw' if wants_raw else 'warm')} pool for "
                f"layer {layer_name!r} is exhausted. Increase the hot budget "
                "reserve or reduce concurrency/max context."
            )
        slot = free_slots.pop()
        generations = (
            self._raw_slot_generations[layer_name]
            if wants_raw
            else self._warm_slot_generations[layer_name]
        )
        generations[slot] += 1
        allocation = _TMHAllocation(
            role=role,
            slot=slot,
            slot_generation=generations[slot],
        )
        cache = self._caches[layer_name]
        generation_tensor = (
            cache.raw_slot_generation if wants_raw else cache.warm_slot_generation
        )
        generation_tensor[slot] = allocation.slot_generation
        return allocation

    def _register_binding(
        self,
        binding_key: tuple[str, str, int],
        descriptor: TMHPhysicalPageDescriptor,
    ) -> None:
        self._request_binding_keys.setdefault(descriptor.request_id, set()).add(
            binding_key
        )
        if descriptor.storage == TMHStorageKind.CANONICAL:
            self._canonical_binding_keys.setdefault(
                self._canonical_key(descriptor), set()
            ).add(binding_key)

    def _unregister_binding(
        self,
        binding_key: tuple[str, str, int],
        binding: _TMHBinding,
    ) -> None:
        request_keys = self._request_binding_keys.get(binding_key[0])
        if request_keys is not None:
            request_keys.discard(binding_key)
            if not request_keys:
                self._request_binding_keys.pop(binding_key[0], None)
                self._request_total_pages.pop(binding_key[0], None)
        if binding.descriptor.storage == TMHStorageKind.CANONICAL:
            canonical_key = self._canonical_key(binding.descriptor)
            siblings = self._canonical_binding_keys.get(canonical_key)
            if siblings is not None:
                siblings.discard(binding_key)
                if not siblings:
                    self._canonical_binding_keys.pop(canonical_key, None)

    def _warm_role(self, layer_name: str) -> TMHPageRole:
        return (
            TMHPageRole.WARM_INT8_INT8
            if self._caches[layer_name].spec.tmh_late_layer
            else TMHPageRole.WARM_INT8_INT4
        )

    def _return_slot(self, layer_name: str, allocation: _TMHAllocation) -> None:
        free = (
            self._raw_free_slots[layer_name]
            if allocation.raw
            else self._warm_free_slots[layer_name]
        )
        if allocation.slot not in free:
            free.append(allocation.slot)

    @staticmethod
    def _canonical_key(
        descriptor: TMHPhysicalPageDescriptor,
    ) -> tuple[str, int, int, int]:
        return (
            descriptor.layer_name,
            descriptor.cache_group_id,
            descriptor.logical_block_id,
            descriptor.allocation_generation,
        )

    @staticmethod
    def _canonical_resident_identity(
        descriptor: TMHPhysicalPageDescriptor,
    ) -> tuple[str, int, int]:
        return (
            descriptor.layer_name,
            descriptor.cache_group_id,
            descriptor.logical_block_id,
        )

    @staticmethod
    def _overlay_key(
        descriptor: TMHPhysicalPageDescriptor,
        request_generation: int,
    ) -> tuple[str, str, int, int]:
        return (
            descriptor.layer_name,
            descriptor.request_id,
            request_generation,
            descriptor.page_index,
        )

    def _clear_slot(
        self, cache: TMHPhysicalKVCache, allocation: _TMHAllocation
    ) -> None:
        if allocation.raw:
            cache.raw_key[allocation.slot].zero_()
            cache.raw_value[allocation.slot].zero_()
        else:
            cache.warm_key[allocation.slot].zero_()
            cache.warm_value[allocation.slot].zero_()
            cache.warm_k_scale[allocation.slot].zero_()
            cache.warm_v_scale[allocation.slot].zero_()

    def _read_page(
        self,
        cache: TMHPhysicalKVCache,
        allocation: _TMHAllocation,
        valid_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if allocation.raw:
            key = (
                cache.raw_key[allocation.slot]
                .permute(2, 0, 1, 3)
                .reshape(cache.spec.block_size, cache.spec.num_kv_heads, -1)
            )
            value = cache.raw_value[allocation.slot].permute(2, 0, 1)
            return key[:valid_tokens].float(), value[:valid_tokens].float()
        key = cache.warm_key[allocation.slot, :valid_tokens].float()
        key *= cache.warm_k_scale[allocation.slot, :valid_tokens, :, None]
        if allocation.role == TMHPageRole.WARM_INT8_INT8:
            value = cache.warm_value[allocation.slot, :valid_tokens].float()
            value *= cache.warm_v_scale[allocation.slot, :valid_tokens, :, None]
        else:
            scale, zero_point = _unpack_scale_zero_point(
                cache.warm_v_scale[allocation.slot, :valid_tokens]
            )
            value = dequantize_tmh_int4(
                cache.warm_value[allocation.slot, :valid_tokens],
                scale,
                zero_point,
            )
        return key, value

    def _write_page(
        self,
        cache: TMHPhysicalKVCache,
        allocation: _TMHAllocation,
        key: torch.Tensor,
        value: torch.Tensor,
        valid_tokens: int,
    ) -> None:
        if allocation.raw:
            raw_key = (
                key[:valid_tokens]
                .to(cache.dtype)
                .reshape(valid_tokens, cache.spec.num_kv_heads, -1, 16 // cache.raw_key.element_size())
                .permute(1, 2, 0, 3)
            )
            cache.raw_key[allocation.slot, :, :, :valid_tokens].copy_(raw_key)
            cache.raw_value[allocation.slot, :, :, :valid_tokens].copy_(
                value[:valid_tokens].to(cache.dtype).permute(1, 2, 0)
            )
            return
        key_f = key[:valid_tokens].float()
        k_absmax = key_f.abs().amax(dim=-1)
        k_scale = torch.clamp(k_absmax / 127.0, min=1e-6)
        k_q = _round_half_away_from_zero(key_f / k_scale[..., None]).clamp(
            -128, 127
        )
        cache.warm_key[allocation.slot, :valid_tokens].copy_(k_q.to(torch.int8))
        cache.warm_k_scale[allocation.slot, :valid_tokens].copy_(k_scale)
        if allocation.role == TMHPageRole.WARM_INT8_INT8:
            value_f = value[:valid_tokens].float()
            v_absmax = value_f.abs().amax(dim=-1)
            v_scale = torch.clamp(v_absmax / 127.0, min=1e-6)
            v_q = _round_half_away_from_zero(value_f / v_scale[..., None]).clamp(
                -128, 127
            )
            cache.warm_value[allocation.slot, :valid_tokens].copy_(v_q.to(torch.int8))
            cache.warm_v_scale[allocation.slot, :valid_tokens].copy_(v_scale)
        else:
            packed, scale, zero_point = quantize_tmh_int4(value[:valid_tokens])
            unpacked_q = torch.stack(
                (
                    packed.to(torch.uint8) & 0xF,
                    (packed.to(torch.uint8) >> 4) & 0xF,
                ),
                dim=-1,
            )
            self.counters["quantized_values"] += unpacked_q.numel()
            self.counters["quantization_saturated_values"] += int(
                ((unpacked_q == 0) | (unpacked_q == 15)).sum().item()
            )
            cache.warm_value[allocation.slot, :valid_tokens].copy_(packed)
            cache.warm_v_scale[allocation.slot, :valid_tokens].copy_(
                _pack_scale_zero_point(scale, zero_point)
            )

    def _zero_page_tail(
        self,
        cache: TMHPhysicalKVCache,
        allocation: _TMHAllocation,
        valid_tokens: int,
    ) -> None:
        if allocation.raw:
            cache.raw_key[allocation.slot, :, :, valid_tokens:].zero_()
            cache.raw_value[allocation.slot, :, :, valid_tokens:].zero_()
        else:
            cache.warm_key[allocation.slot, valid_tokens:].zero_()
            cache.warm_value[allocation.slot, valid_tokens:].zero_()
            cache.warm_k_scale[allocation.slot, valid_tokens:].zero_()
            cache.warm_v_scale[allocation.slot, valid_tokens:].zero_()

    @staticmethod
    def _clear_request_page(
        cache: TMHPhysicalKVCache, row: int, page: int
    ) -> None:
        cache.request_slot_by_row_page[row, page] = -1
        cache.request_role_by_row_page[row, page] = -1
        cache.request_valid_tokens_by_row_page[row, page] = 0
        cache.request_materialized_by_row_page[row, page] = False
        cache.request_slot_generation_by_row_page[row, page] = -1
        cache.native_block_table_by_seq[row, page] = -1
        cache.native_block_valid_by_seq[row, page] = False

    def _clear_request_row(self, cache: TMHPhysicalKVCache, row: int) -> None:
        cache.request_slot_by_row_page[row].fill_(-1)
        cache.request_role_by_row_page[row].fill_(-1)
        cache.request_valid_tokens_by_row_page[row].zero_()
        cache.request_materialized_by_row_page[row].zero_()
        cache.request_slot_generation_by_row_page[row].fill_(-1)
        cache.native_block_table_by_seq[row].fill_(-1)
        cache.native_block_valid_by_seq[row].zero_()

    def _checkpoint(self, stage: str) -> None:
        if self._failure_injector is not None:
            self._failure_injector(stage)

    def _snapshot(self) -> dict[str, object]:
        tensor_names = (
            "canonical_role_by_logical_block",
            "canonical_slot_by_logical_block",
            "canonical_generation_by_logical_block",
            "request_slot_by_row_page",
            "request_role_by_row_page",
            "request_valid_tokens_by_row_page",
            "request_materialized_by_row_page",
            "request_slot_generation_by_row_page",
            "native_block_table_by_seq",
            "native_block_valid_by_seq",
            "raw_slot_generation",
            "warm_slot_generation",
        )
        return {
            "raw_free": {key: value.copy() for key, value in self._raw_free_slots.items()},
            "warm_free": {key: value.copy() for key, value in self._warm_free_slots.items()},
            "raw_gen": {key: value.copy() for key, value in self._raw_slot_generations.items()},
            "warm_gen": {key: value.copy() for key, value in self._warm_slot_generations.items()},
            "canonical": self._canonical_allocations.copy(),
            "canonical_binding_keys": {
                key: value.copy() for key, value in self._canonical_binding_keys.items()
            },
            "request_binding_keys": {
                key: value.copy() for key, value in self._request_binding_keys.items()
            },
            "resident": self._canonical_resident_key.copy(),
            "pending_canonical_releases": self._pending_canonical_releases.copy(),
            "overlay": self._overlay_allocations.copy(),
            "bindings": self._request_bindings.copy(),
            "rows": self._request_rows.copy(),
            "request_total_pages": self._request_total_pages.copy(),
            "versions": self._request_versions.copy(),
            "last_generation": self._last_request_generation.copy(),
            "committed": self._committed_ids.copy(),
            "tensors": {
                layer: {name: getattr(cache, name).clone() for name in tensor_names}
                for layer, cache in self._caches.items()
            },
        }

    def _restore(self, snapshot: dict[str, object]) -> None:
        self._raw_free_slots = snapshot["raw_free"]  # type: ignore[assignment]
        self._warm_free_slots = snapshot["warm_free"]  # type: ignore[assignment]
        self._raw_slot_generations = snapshot["raw_gen"]  # type: ignore[assignment]
        self._warm_slot_generations = snapshot["warm_gen"]  # type: ignore[assignment]
        self._canonical_allocations = snapshot["canonical"]  # type: ignore[assignment]
        self._canonical_binding_keys = snapshot["canonical_binding_keys"]  # type: ignore[assignment]
        self._request_binding_keys = snapshot["request_binding_keys"]  # type: ignore[assignment]
        self._canonical_resident_key = snapshot["resident"]  # type: ignore[assignment]
        self._pending_canonical_releases = snapshot["pending_canonical_releases"]  # type: ignore[assignment]
        self._overlay_allocations = snapshot["overlay"]  # type: ignore[assignment]
        self._request_bindings = snapshot["bindings"]  # type: ignore[assignment]
        self._request_rows = snapshot["rows"]  # type: ignore[assignment]
        self._request_total_pages = snapshot["request_total_pages"]  # type: ignore[assignment]
        self._request_versions = snapshot["versions"]  # type: ignore[assignment]
        self._last_request_generation = snapshot["last_generation"]  # type: ignore[assignment]
        self._committed_ids = snapshot["committed"]  # type: ignore[assignment]
        tensors = snapshot["tensors"]
        assert isinstance(tensors, dict)
        for layer, by_name in tensors.items():
            assert isinstance(layer, str) and isinstance(by_name, dict)
            cache = self._caches[layer]
            for name, value in by_name.items():
                getattr(cache, name).copy_(value)


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
