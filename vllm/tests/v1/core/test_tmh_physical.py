# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.ops.paged_attn import PagedAttention
from vllm.v1.core.tmh_policy import (
    TMHKVRuntimePolicy,
    TMHLayerShape,
    TMHPageRole,
    TMHPhysicalEvent as TMHPhysicalEventData,
    TMHPhysicalEventKind,
    TMHPhysicalPageDescriptor,
    TMHStorageKind,
)
from vllm.v1.kv_cache_interface import TMHFullAttentionSpec
from vllm.v1.tmh_physical import (
    TMHPhysicalRuntime,
    dequantize_tmh_int4,
    quantize_tmh_int4,
    reshape_tmh_physical_kv_cache,
)


def make_physical_cache():
    spec = TMHFullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=16,
        head_size_v=16,
        dtype=torch.float16,
        tmh_hot_budget_pct=25.0,
        tmh_max_num_seqs=2,
        tmh_max_model_pages=8,
    )
    backing = torch.empty(spec.physical_allocation_bytes(8), dtype=torch.uint8)
    return reshape_tmh_physical_kv_cache(backing, spec, num_logical_blocks=8)


def test_tmh_pool_planning_preserves_warm_capacity_with_high_scheduler_concurrency():
    spec = TMHFullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=16,
        head_size_v=16,
        dtype=torch.float16,
        tmh_hot_budget_pct=25.0,
        tmh_max_num_seqs=1024,
        tmh_max_model_pages=16,
    )

    raw_pages, warm_pages = spec.physical_pool_page_counts(29693)

    assert raw_pages > 1024
    assert warm_pages > 0
    assert raw_pages + warm_pages == 29693


def test_tmh_request_descriptor_tables_are_bounded_by_request_pages():
    spec = TMHFullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=16,
        head_size_v=16,
        dtype=torch.float16,
        tmh_hot_budget_pct=25.0,
        tmh_max_num_seqs=3,
        tmh_max_model_pages=11,
    )
    backing = torch.empty(spec.physical_allocation_bytes(4096), dtype=torch.uint8)

    cache = reshape_tmh_physical_kv_cache(
        backing,
        spec,
        num_logical_blocks=4096,
    )

    assert cache.canonical_role_by_logical_block.shape == (4096,)
    raw_pages, _ = spec.physical_pool_page_counts(4096)
    assert cache.raw_kv_cache.shape == (2, raw_pages, 16, 2, 16)
    assert cache.raw_key.data_ptr() == cache.raw_kv_cache[0].data_ptr()
    assert cache.raw_value.data_ptr() == cache.raw_kv_cache[1].data_ptr()
    assert cache.request_slot_by_row_page.shape == (3, 11)
    assert backing.numel() == spec.physical_allocation_bytes(4096)
    assert backing.numel() == sum(spec.physical_memory_ledger(4096).values())
    backing_ptr = backing.untyped_storage().data_ptr()
    assert cache.request_slot_by_row_page.untyped_storage().data_ptr() == backing_ptr
    assert cache.raw_slot_generation.untyped_storage().data_ptr() == backing_ptr
    assert cache.native_block_table_gather.untyped_storage().data_ptr() == backing_ptr


def descriptor(
    *,
    request_id: str = "req-1",
    page_index: int,
    logical_block_id: int,
    role: TMHPageRole,
    storage: TMHStorageKind = TMHStorageKind.CANONICAL,
    prefix_cached: bool | None = None,
    allocation_generation: int = 1,
    valid_tokens: int = 16,
) -> TMHPhysicalPageDescriptor:
    if prefix_cached is None:
        prefix_cached = storage == TMHStorageKind.REQUEST_OVERLAY
    return TMHPhysicalPageDescriptor(
        request_id=request_id,
        layer_name="model.layers.0.self_attn",
        logical_block_id=logical_block_id,
        page_index=page_index,
        role=role,
        storage=storage,
        prefix_cached=prefix_cached,
        k_quant_mode="raw" if role in (TMHPageRole.PINNED_RAW, TMHPageRole.HOT_RAW) else "int8_per_token_head",
        v_quant_mode=(
            "raw"
            if role in (TMHPageRole.PINNED_RAW, TMHPageRole.HOT_RAW)
            else "int4_per_token_head"
            if role == TMHPageRole.WARM_INT8_INT4
            else "int8_per_token_head"
        ),
        allocation_generation=allocation_generation,
        valid_tokens=valid_tokens,
    )


def event(
    *,
    request_id: str,
    descriptors: tuple[TMHPhysicalPageDescriptor, ...],
    total_pages: int,
    recent_start_page: int,
    hot_pages: int,
    sequence: int = 1,
    request_generation: int = 1,
    released_request_ids: tuple[str, ...] = (),
    released_descriptors: tuple[TMHPhysicalPageDescriptor, ...] = (),
    event_kind: TMHPhysicalEventKind | None = None,
) -> TMHPhysicalEventData:
    if event_kind is None:
        event_kind = (
            TMHPhysicalEventKind.RELEASE
            if released_request_ids
            else TMHPhysicalEventKind.DELTA
        )
    return TMHPhysicalEventData(
        schema_version=1,
        event_kind=event_kind,
        request_id=request_id,
        request_generation=request_generation,
        sequence=sequence,
        expected_base_version=sequence - 1,
        target_version=sequence,
        commit_id=f"{request_id}:{request_generation}:{sequence}:{event_kind.name}",
        descriptors=descriptors,
        total_pages=total_pages,
        recent_start_page=recent_start_page,
        hot_pages=hot_pages,
        released_request_ids=released_request_ids,
        released_descriptors=released_descriptors,
    )


def test_tmh_physical_runtime_respects_resolved_page_roles():
    cache = make_physical_cache()
    runtime = TMHPhysicalRuntime()
    runtime.register_cache("model.layers.0.self_attn", cache)

    runtime.apply_events(
        [
            event(
                request_id="req-raw",
                descriptors=(
                    descriptor(
                        request_id="req-raw",
                        page_index=0,
                        logical_block_id=0,
                        role=TMHPageRole.PINNED_RAW,
                    ),
                    descriptor(
                        request_id="req-raw",
                        page_index=1,
                        logical_block_id=1,
                        role=TMHPageRole.WARM_INT8_INT4,
                    ),
                ),
                total_pages=2,
                recent_start_page=1,
                hot_pages=1,
            )
        ],
        {"req-raw": 0},
    )

    assert cache.request_slot_by_row_page[0, :2].tolist() == [0, 0]
    assert cache.request_role_by_row_page[0, :2].tolist() == [
        int(TMHPageRole.PINNED_RAW),
        int(TMHPageRole.WARM_INT8_INT4),
    ]
    assert cache.native_block_table_by_seq[0, 0].item() == 0
    assert cache.native_block_table_by_seq[0, 1:].tolist() == [-1] * 7
    assert cache.canonical_role_by_logical_block[1].item() == int(
        TMHPageRole.WARM_INT8_INT4
    )


def test_tmh_logical_progress_does_not_republish_unchanged_physical_mapping():
    cache = make_physical_cache()
    runtime = TMHPhysicalRuntime()
    runtime.register_cache("model.layers.0.self_attn", cache)
    initial = descriptor(
        page_index=0,
        logical_block_id=0,
        role=TMHPageRole.PINNED_RAW,
        valid_tokens=4,
    )
    runtime.apply_events(
        [
            event(
                request_id="req-1",
                descriptors=(initial,),
                total_pages=1,
                recent_start_page=0,
                hot_pages=1,
            )
        ],
        {"req-1": 0},
    )
    published = runtime.counters["metadata_descriptors_published"]

    grown = descriptor(
        page_index=0,
        logical_block_id=0,
        role=TMHPageRole.PINNED_RAW,
        valid_tokens=8,
    )
    runtime.apply_events(
        [
            event(
                request_id="req-1",
                descriptors=(grown,),
                total_pages=1,
                recent_start_page=0,
                hot_pages=1,
                sequence=2,
            )
        ],
        {"req-1": 0},
    )

    binding = runtime._request_bindings[(
        "req-1",
        "model.layers.0.self_attn",
        0,
    )]
    assert binding.descriptor.valid_tokens == 8
    assert runtime.counters["metadata_descriptors_published"] == published
    assert runtime.counters["metadata_logical_updates"] == 1

    shrunk = descriptor(
        page_index=0,
        logical_block_id=0,
        role=TMHPageRole.PINNED_RAW,
        valid_tokens=2,
    )
    runtime.apply_events(
        [
            event(
                request_id="req-1",
                descriptors=(shrunk,),
                total_pages=1,
                recent_start_page=0,
                hot_pages=1,
                sequence=3,
            )
        ],
        {"req-1": 0},
    )
    assert cache.request_valid_tokens_by_row_page[0, 0].item() == 2
    assert runtime.counters["metadata_descriptors_published"] == published


def test_tmh_graph_capture_rows_are_safe_and_released() -> None:
    cache = make_physical_cache()
    runtime = TMHPhysicalRuntime()
    runtime.register_cache("model.layers.0.self_attn", cache)
    initial_raw_free = runtime.diagnostics()["pools"][
        "model.layers.0.self_attn"
    ]["raw_free"]

    request_ids = runtime.prepare_graph_capture(2)

    assert runtime.batch_is_all_raw(list(request_ids))
    assert cache.request_role_by_row_page[:2, 0].tolist() == [
        int(TMHPageRole.HOT_RAW),
        int(TMHPageRole.HOT_RAW),
    ]
    runtime.finish_graph_capture(request_ids)
    assert cache.request_slot_by_row_page[:2].eq(-1).all()
    assert runtime.diagnostics()["pools"]["model.layers.0.self_attn"][
        "raw_free"
    ] == initial_raw_free


def test_tmh_physical_runtime_maps_request_pages_to_canonical_and_overlay_slots():
    cache = make_physical_cache()
    runtime = TMHPhysicalRuntime()
    runtime.register_cache("model.layers.0.self_attn", cache)

    runtime.apply_events(
        [
            event(
                request_id="req-1",
                descriptors=(
                    descriptor(
                        page_index=1,
                        logical_block_id=1,
                        role=TMHPageRole.WARM_INT8_INT4,
                    ),
                    descriptor(
                        page_index=3,
                        logical_block_id=3,
                        role=TMHPageRole.HOT_RAW,
                        storage=TMHStorageKind.REQUEST_OVERLAY,
                    ),
                ),
                total_pages=4,
                recent_start_page=3,
                hot_pages=1,
            )
        ],
        {"req-1": 0},
    )

    assert cache.request_slot_by_row_page[0, 1].item() == 0
    assert cache.request_role_by_row_page[0, 1].item() == int(
        TMHPageRole.WARM_INT8_INT4
    )
    assert cache.canonical_role_by_logical_block[1].item() == int(
        TMHPageRole.WARM_INT8_INT4
    )
    overlay_slot = cache.request_slot_by_row_page[0, 3].item()
    assert overlay_slot >= 0
    assert cache.request_role_by_row_page[0, 3].item() == int(TMHPageRole.HOT_RAW)
    assert cache.native_block_table_by_seq[0, 1].item() == -1
    assert cache.native_block_table_by_seq[0, 2].item() == -1
    assert cache.native_block_table_by_seq[0, 3].item() == overlay_slot

    runtime.apply_events(
        [
            event(
                request_id="req-1",
                descriptors=(),
                total_pages=0,
                recent_start_page=0,
                hot_pages=0,
                released_request_ids=("req-1",),
                sequence=2,
            )
        ],
        {},
    )

    runtime.apply_events(
        [
            event(
                request_id="req-2",
                descriptors=(
                    descriptor(
                        request_id="req-2",
                        page_index=3,
                        logical_block_id=3,
                        role=TMHPageRole.HOT_RAW,
                        storage=TMHStorageKind.REQUEST_OVERLAY,
                    ),
                ),
                total_pages=4,
                recent_start_page=3,
                hot_pages=1,
            )
        ],
        {"req-2": 1},
    )
    assert cache.request_slot_by_row_page[1, 3].item() == overlay_slot
    assert cache.request_role_by_row_page[1, 3].item() == int(TMHPageRole.HOT_RAW)
    assert cache.native_block_table_by_seq[1, 3].item() == overlay_slot


def test_tmh_physical_runtime_shares_prefix_cached_hot_raw_pages():
    cache = make_physical_cache()
    runtime = TMHPhysicalRuntime()
    runtime.register_cache("model.layers.0.self_attn", cache)

    runtime.apply_events(
        [
            event(
                request_id="req-1",
                descriptors=(
                    descriptor(
                        request_id="req-1",
                        page_index=3,
                        logical_block_id=3,
                        role=TMHPageRole.HOT_RAW,
                        storage=TMHStorageKind.CANONICAL,
                        prefix_cached=True,
                    ),
                ),
                total_pages=4,
                recent_start_page=3,
                hot_pages=1,
            ),
            event(
                request_id="req-2",
                descriptors=(
                    descriptor(
                        request_id="req-2",
                        page_index=3,
                        logical_block_id=3,
                        role=TMHPageRole.HOT_RAW,
                        storage=TMHStorageKind.CANONICAL,
                        prefix_cached=True,
                    ),
                ),
                total_pages=4,
                recent_start_page=3,
                hot_pages=1,
            ),
        ],
        {"req-1": 0, "req-2": 1},
    )

    shared_slot = cache.request_slot_by_row_page[0, 3].item()
    assert shared_slot >= 0
    assert cache.request_slot_by_row_page[1, 3].item() == shared_slot
    assert cache.canonical_role_by_logical_block[3].item() == int(TMHPageRole.HOT_RAW)
    assert cache.canonical_slot_by_logical_block[3].item() == shared_slot


def test_tmh_physical_runtime_propagates_canonical_role_updates_to_shared_siblings():
    cache = make_physical_cache()
    runtime = TMHPhysicalRuntime()
    runtime.register_cache("model.layers.0.self_attn", cache)

    runtime.apply_events(
        [
            event(
                request_id="req-1",
                descriptors=(
                    descriptor(
                        request_id="req-1",
                        page_index=3,
                        logical_block_id=3,
                        role=TMHPageRole.HOT_RAW,
                        storage=TMHStorageKind.CANONICAL,
                        prefix_cached=True,
                    ),
                ),
                total_pages=4,
                recent_start_page=3,
                hot_pages=1,
            ),
            event(
                request_id="req-2",
                descriptors=(
                    descriptor(
                        request_id="req-2",
                        page_index=3,
                        logical_block_id=3,
                        role=TMHPageRole.HOT_RAW,
                        storage=TMHStorageKind.CANONICAL,
                        prefix_cached=True,
                    ),
                ),
                total_pages=4,
                recent_start_page=3,
                hot_pages=1,
            ),
        ],
        {"req-1": 0, "req-2": 1},
    )

    runtime.apply_events(
        [
            event(
                request_id="req-1",
                descriptors=(
                    descriptor(
                        request_id="req-1",
                        page_index=3,
                        logical_block_id=3,
                        role=TMHPageRole.WARM_INT8_INT4,
                        storage=TMHStorageKind.CANONICAL,
                        prefix_cached=True,
                    ),
                ),
                total_pages=4,
                recent_start_page=3,
                hot_pages=0,
                sequence=2,
            ),
        ],
        {"req-1": 0, "req-2": 1},
    )

    req1_slot = cache.request_slot_by_row_page[0, 3].item()
    req2_slot = cache.request_slot_by_row_page[1, 3].item()
    assert req1_slot >= 0
    assert req2_slot == req1_slot
    assert cache.request_role_by_row_page[0, 3].item() == int(TMHPageRole.WARM_INT8_INT4)
    assert cache.request_role_by_row_page[1, 3].item() == int(TMHPageRole.WARM_INT8_INT4)
    assert cache.native_block_valid_by_seq[0, 3].item() is False
    assert cache.native_block_valid_by_seq[1, 3].item() is False
    assert cache.canonical_role_by_logical_block[3].item() == int(TMHPageRole.WARM_INT8_INT4)
    assert cache.canonical_slot_by_logical_block[3].item() == req1_slot


def test_tmh_policy_retains_canonical_pages_after_last_active_owner():
    policy = TMHKVRuntimePolicy(
        policy="physical",
        hot_budget_pct=25.0,
        page_tokens=16,
        layers=[
            TMHLayerShape(
                layer_name="model.layers.0.self_attn",
                layer_index=0,
                num_kv_heads=2,
                head_size=16,
                head_size_v=16,
                raw_dtype_bytes=2.0,
            )
        ],
        regular_page_bytes_by_group=[512],
    )

    policy.record_physical_descriptors_from_block_ids(
        request_id="req-1",
        total_tokens=64,
        logical_block_ids=[0, 1, 2, 3],
    )
    policy.take_physical_events()

    policy.forget_request("req-1")
    events = policy.take_physical_events()

    assert len(events) == 1
    assert events[0].released_request_ids == ("req-1",)
    assert events[0].released_descriptors == ()


def test_tmh_physical_runtime_reuses_released_canonical_raw_slots():
    cache = make_physical_cache()
    runtime = TMHPhysicalRuntime()
    runtime.register_cache("model.layers.0.self_attn", cache)
    raw_capacity = cache.raw_key.shape[0]
    first_descriptors = tuple(
        descriptor(
            page_index=page,
            logical_block_id=page,
            role=TMHPageRole.HOT_RAW,
        )
        for page in range(raw_capacity)
    )
    runtime.apply_events(
        [
            event(
                request_id="req-1",
                descriptors=first_descriptors,
                total_pages=raw_capacity,
                recent_start_page=0,
                hot_pages=raw_capacity,
            )
        ],
        {"req-1": 0},
    )

    runtime.apply_events(
        [
            event(
                request_id="req-1",
                descriptors=(),
                total_pages=0,
                recent_start_page=0,
                hot_pages=0,
                released_request_ids=("req-1",),
                released_descriptors=first_descriptors,
                sequence=2,
            )
        ],
        {},
    )

    runtime.apply_events(
        [
            event(
                request_id="req-2",
                descriptors=tuple(
                    descriptor(
                        request_id="req-2",
                        page_index=page,
                        logical_block_id=page + raw_capacity,
                        role=TMHPageRole.HOT_RAW,
                    )
                    for page in range(raw_capacity)
                ),
                total_pages=raw_capacity,
                recent_start_page=0,
                hot_pages=raw_capacity,
            )
        ],
        {"req-2": 1},
    )

    assert cache.request_slot_by_row_page[1, 0].item() in range(raw_capacity)


def test_tmh_physical_runtime_rejects_raw_pool_exhaustion_without_mutation():
    cache = make_physical_cache()
    runtime = TMHPhysicalRuntime()
    runtime.register_cache("model.layers.0.self_attn", cache)
    raw_capacity = cache.raw_key.shape[0]
    raw_descriptors = tuple(
        descriptor(
            page_index=page,
            logical_block_id=page,
            role=TMHPageRole.HOT_RAW,
        )
        for page in range(raw_capacity)
    )
    fallback_descriptor = descriptor(
        page_index=raw_capacity,
        logical_block_id=raw_capacity,
        role=TMHPageRole.HOT_RAW,
    )

    with pytest.raises(RuntimeError, match="raw.*pool.*exhausted"):
        runtime.apply_events(
            [
                event(
                    request_id="req-1",
                    descriptors=raw_descriptors + (fallback_descriptor,),
                    total_pages=raw_capacity + 1,
                    recent_start_page=0,
                    hot_pages=raw_capacity + 1,
                )
            ],
            {"req-1": 0},
        )

    assert cache.request_slot_by_row_page[0].tolist() == [-1] * 8
    assert len(runtime._raw_free_slots["model.layers.0.self_attn"]) == raw_capacity


def test_tmh_delta_preserves_unchanged_pages_and_payloads():
    cache = make_physical_cache()
    runtime = TMHPhysicalRuntime()
    runtime.register_cache("model.layers.0.self_attn", cache)
    initial = tuple(
        descriptor(
            page_index=page,
            logical_block_id=page,
            role=TMHPageRole.HOT_RAW,
        )
        for page in range(3)
    )
    runtime.apply_events(
        [event(
            request_id="req-1",
            descriptors=initial,
            total_pages=3,
            recent_start_page=0,
            hot_pages=3,
        )],
        {"req-1": 0},
    )
    slots_before = cache.request_slot_by_row_page[0, :3].clone()
    page_zero = int(slots_before[0])
    page_one = int(slots_before[1])
    cache.raw_key[page_zero].fill_(1.25)
    cache.raw_value[page_zero].fill_(-2.5)
    cache.raw_key[page_one].fill_(3.5)
    cache.raw_value[page_one].fill_(4.5)
    raw_zero_before = cache.raw_key[page_zero].clone()
    raw_one_before = cache.raw_value[page_one].clone()

    runtime.apply_events(
        [event(
            request_id="req-1",
            descriptors=(descriptor(
                page_index=2,
                logical_block_id=2,
                role=TMHPageRole.WARM_INT8_INT4,
            ),),
            total_pages=3,
            recent_start_page=3,
            hot_pages=0,
            sequence=2,
        )],
        {"req-1": 0},
    )

    torch.testing.assert_close(cache.request_slot_by_row_page[0, :2], slots_before[:2])
    assert cache.request_role_by_row_page[0, :2].tolist() == [1, 1]
    torch.testing.assert_close(cache.raw_key[page_zero], raw_zero_before)
    torch.testing.assert_close(cache.raw_value[page_one], raw_one_before)


def test_tmh_role_transition_materializes_payload_before_publication():
    cache = make_physical_cache()
    runtime = TMHPhysicalRuntime()
    runtime.register_cache("model.layers.0.self_attn", cache)
    raw_descriptor = descriptor(
        page_index=1,
        logical_block_id=1,
        role=TMHPageRole.HOT_RAW,
    )
    runtime.apply_events(
        [event(
            request_id="req-1",
            descriptors=(raw_descriptor,),
            total_pages=2,
            recent_start_page=1,
            hot_pages=1,
        )],
        {"req-1": 0},
    )
    raw_slot = cache.request_slot_by_row_page[0, 1].item()
    key = torch.linspace(-2, 2, 16 * 2 * 16).reshape(16, 2, 16)
    value = torch.sin(torch.arange(16 * 2 * 16).reshape(16, 2, 16) / 19)
    runtime._write_page(
        cache,
        runtime._request_bindings[("req-1", "model.layers.0.self_attn", 1)].allocation,
        key,
        value,
        16,
    )

    warm_descriptor = descriptor(
        page_index=1,
        logical_block_id=1,
        role=TMHPageRole.WARM_INT8_INT4,
    )
    runtime.apply_events(
        [event(
            request_id="req-1",
            descriptors=(warm_descriptor,),
            total_pages=2,
            recent_start_page=2,
            hot_pages=0,
            sequence=2,
        )],
        {"req-1": 0},
    )
    assert cache.request_role_by_row_page[0, 1].item() == int(
        TMHPageRole.WARM_INT8_INT4
    )
    warm_binding = runtime._request_bindings[
        ("req-1", "model.layers.0.self_attn", 1)
    ]
    reconstructed_k, reconstructed_v = runtime._read_page(
        cache, warm_binding.allocation, 16
    )
    torch.testing.assert_close(reconstructed_k, key, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(reconstructed_v, value, atol=8e-2, rtol=8e-2)

    runtime.apply_events(
        [event(
            request_id="req-1",
            descriptors=(raw_descriptor,),
            total_pages=2,
            recent_start_page=1,
            hot_pages=1,
            sequence=3,
        )],
        {"req-1": 0},
    )
    rebound = runtime._request_bindings[("req-1", "model.layers.0.self_attn", 1)]
    roundtrip_k, roundtrip_v = runtime._read_page(cache, rebound.allocation, 16)
    assert rebound.allocation.slot == raw_slot
    torch.testing.assert_close(roundtrip_k, key, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(roundtrip_v, value, atol=8e-2, rtol=8e-2)


def test_tmh_overlay_copies_canonical_payload_before_visibility():
    cache = make_physical_cache()
    runtime = TMHPhysicalRuntime()
    runtime.register_cache("model.layers.0.self_attn", cache)
    canonical = descriptor(
        page_index=1,
        logical_block_id=1,
        role=TMHPageRole.HOT_RAW,
        prefix_cached=True,
    )
    runtime.apply_events(
        [event(
            request_id="req-1",
            descriptors=(canonical,),
            total_pages=2,
            recent_start_page=1,
            hot_pages=1,
        )],
        {"req-1": 0},
    )
    canonical_binding = runtime._request_bindings[
        ("req-1", "model.layers.0.self_attn", 1)
    ]
    cache.raw_key[canonical_binding.allocation.slot].fill_(6.0)
    cache.raw_value[canonical_binding.allocation.slot].fill_(-7.0)
    overlay = descriptor(
        page_index=1,
        logical_block_id=1,
        role=TMHPageRole.HOT_RAW,
        storage=TMHStorageKind.REQUEST_OVERLAY,
        prefix_cached=True,
    )

    runtime.apply_events(
        [event(
            request_id="req-1",
            descriptors=(overlay,),
            total_pages=2,
            recent_start_page=1,
            hot_pages=1,
            sequence=2,
        )],
        {"req-1": 0},
    )
    overlay_binding = runtime._request_bindings[
        ("req-1", "model.layers.0.self_attn", 1)
    ]
    assert overlay_binding.allocation.slot != canonical_binding.allocation.slot
    torch.testing.assert_close(
        cache.raw_key[overlay_binding.allocation.slot],
        cache.raw_key[canonical_binding.allocation.slot],
    )
    torch.testing.assert_close(
        cache.raw_value[overlay_binding.allocation.slot],
        cache.raw_value[canonical_binding.allocation.slot],
    )


def test_tmh_event_replay_is_idempotent_and_reordering_fails_closed():
    cache = make_physical_cache()
    runtime = TMHPhysicalRuntime()
    runtime.register_cache("model.layers.0.self_attn", cache)
    first = event(
        request_id="req-1",
        descriptors=(descriptor(
            page_index=0,
            logical_block_id=0,
            role=TMHPageRole.PINNED_RAW,
        ),),
        total_pages=1,
        recent_start_page=0,
        hot_pages=1,
    )
    runtime.apply_events([first], {"req-1": 0})
    slot_before = cache.request_slot_by_row_page[0, 0].item()
    free_before = runtime._raw_free_slots["model.layers.0.self_attn"].copy()

    runtime.apply_events([first], {"req-1": 0})
    assert cache.request_slot_by_row_page[0, 0].item() == slot_before
    assert runtime._raw_free_slots["model.layers.0.self_attn"] == free_before
    with pytest.raises(RuntimeError, match="out-of-order"):
        runtime.apply_events(
            [event(
                request_id="req-1",
                descriptors=(),
                total_pages=1,
                recent_start_page=0,
                hot_pages=1,
                sequence=3,
            )],
            {"req-1": 0},
        )
    assert cache.request_slot_by_row_page[0, 0].item() == slot_before


def test_tmh_truncate_removes_trailing_pages_and_validity():
    cache = make_physical_cache()
    runtime = TMHPhysicalRuntime()
    runtime.register_cache("model.layers.0.self_attn", cache)
    runtime.apply_events(
        [event(
            request_id="req-1",
            descriptors=tuple(
                descriptor(
                    page_index=page,
                    logical_block_id=page,
                    role=TMHPageRole.HOT_RAW,
                )
                for page in range(3)
            ),
            total_pages=3,
            recent_start_page=0,
            hot_pages=3,
        )],
        {"req-1": 0},
    )
    runtime.apply_events(
        [event(
            request_id="req-1",
            descriptors=(),
            total_pages=1,
            recent_start_page=0,
            hot_pages=1,
            sequence=2,
        )],
        {"req-1": 0},
    )
    assert cache.request_slot_by_row_page[0].tolist()[1:] == [-1] * 7
    assert cache.native_block_table_by_seq[0].tolist()[1:] == [-1] * 7
    assert not cache.request_materialized_by_row_page[0, 1:].any()
    assert {
        key[2]
        for key in runtime._request_bindings
        if key[0] == "req-1"
    } == {0}


def test_tmh_reused_overlay_slot_is_cleared_and_generation_advances():
    cache = make_physical_cache()
    runtime = TMHPhysicalRuntime()
    runtime.register_cache("model.layers.0.self_attn", cache)
    first = descriptor(
        page_index=0,
        logical_block_id=0,
        role=TMHPageRole.HOT_RAW,
        storage=TMHStorageKind.REQUEST_OVERLAY,
        valid_tokens=16,
    )
    runtime.apply_events(
        [event(
            request_id="req-1",
            descriptors=(first,),
            total_pages=1,
            recent_start_page=0,
            hot_pages=1,
        )],
        {"req-1": 0},
    )
    first_binding = runtime._request_bindings[
        ("req-1", "model.layers.0.self_attn", 0)
    ]
    cache.raw_key[first_binding.allocation.slot].fill_(123)
    cache.raw_value[first_binding.allocation.slot].fill_(-45)
    runtime.apply_events(
        [event(
            request_id="req-1",
            descriptors=(),
            total_pages=0,
            recent_start_page=0,
            hot_pages=0,
            released_request_ids=("req-1",),
            sequence=2,
        )],
        {},
    )
    second = descriptor(
        request_id="req-2",
        page_index=0,
        logical_block_id=1,
        role=TMHPageRole.HOT_RAW,
        storage=TMHStorageKind.REQUEST_OVERLAY,
        valid_tokens=4,
    )
    runtime.apply_events(
        [event(
            request_id="req-2",
            descriptors=(second,),
            total_pages=1,
            recent_start_page=0,
            hot_pages=1,
        )],
        {"req-2": 1},
    )
    second_binding = runtime._request_bindings[
        ("req-2", "model.layers.0.self_attn", 0)
    ]
    assert second_binding.allocation.slot == first_binding.allocation.slot
    assert second_binding.allocation.slot_generation > (
        first_binding.allocation.slot_generation
    )
    assert cache.raw_key[second_binding.allocation.slot].count_nonzero().item() == 0
    assert cache.raw_value[second_binding.allocation.slot].count_nonzero().item() == 0


@pytest.mark.parametrize(
    "stage", ["validated", "reserved", "materialized", "published", "released"]
)
def test_tmh_transition_failure_rolls_back_atomically(stage: str):
    cache = make_physical_cache()
    runtime = TMHPhysicalRuntime()
    runtime.register_cache("model.layers.0.self_attn", cache)
    runtime.apply_events(
        [event(
            request_id="req-1",
            descriptors=(descriptor(
                page_index=0,
                logical_block_id=0,
                role=TMHPageRole.HOT_RAW,
            ),),
            total_pages=1,
            recent_start_page=0,
            hot_pages=1,
        )],
        {"req-1": 0},
    )
    slot_before = cache.request_slot_by_row_page.clone()
    role_before = cache.request_role_by_row_page.clone()
    binding_before = runtime._request_bindings.copy()

    def inject(candidate: str) -> None:
        if candidate == stage:
            raise RuntimeError(f"injected at {stage}")

    runtime._failure_injector = inject
    with pytest.raises(RuntimeError, match="injected"):
        runtime.apply_events(
            [event(
                request_id="req-1",
                descriptors=(descriptor(
                    page_index=0,
                    logical_block_id=0,
                    role=TMHPageRole.WARM_INT8_INT4,
                ),),
                total_pages=1,
                recent_start_page=1,
                hot_pages=0,
                sequence=2,
            )],
            {"req-1": 0},
        )
    torch.testing.assert_close(cache.request_slot_by_row_page, slot_before)
    torch.testing.assert_close(cache.request_role_by_row_page, role_before)
    assert runtime._request_bindings == binding_before


@pytest.mark.parametrize(
    "values",
    [
        torch.linspace(0.1, 4.0, 16),
        torch.linspace(-4.0, -0.1, 16),
        torch.full((16,), 2.5),
        torch.full((16,), -2.5),
        torch.zeros(16),
        torch.tensor([100.0] + [0.01] * 15),
    ],
)
def test_tmh_int4_contract_contains_zero_and_handles_sign_definite(values):
    packed, scale, zero_point = quantize_tmh_int4(values)
    reconstructed = dequantize_tmh_int4(packed, scale, zero_point)
    assert 0 <= int(zero_point) <= 15
    assert torch.isfinite(reconstructed).all()
    assert (reconstructed - values).abs().max() <= float(scale) + 1e-5


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_tmh_int4_contract_rejects_nonfinite_values(bad: float):
    values = torch.zeros(16)
    values[3] = bad
    with pytest.raises(ValueError, match="NaN or Inf"):
        quantize_tmh_int4(values)


def test_tmh_early_int4_rejects_odd_value_dimensions_at_initialization():
    spec = TMHFullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=15,
        head_size_v=15,
        dtype=torch.float16,
        tmh_late_layer=False,
        tmh_max_num_seqs=1,
        tmh_max_model_pages=2,
    )
    backing = torch.empty(spec.physical_allocation_bytes(4), dtype=torch.uint8)
    with pytest.raises(ValueError, match="even head_size_v"):
        reshape_tmh_physical_kv_cache(backing, spec, num_logical_blocks=4)


def test_tmh_raw_cache_matches_rocm_paged_attention_split_views():
    cache = make_physical_cache()

    key_cache, value_cache = PagedAttention.split_kv_cache(
        cache.raw_kv_cache,
        cache.spec.num_kv_heads,
        cache.spec.head_size,
    )

    assert key_cache.shape == cache.raw_key.shape
    assert value_cache.shape == cache.raw_value.shape
    assert key_cache.stride() == cache.raw_key.stride()
    assert value_cache.stride() == cache.raw_value.stride()
    assert key_cache.data_ptr() == cache.raw_key.data_ptr()
    assert value_cache.data_ptr() == cache.raw_value.data_ptr()
