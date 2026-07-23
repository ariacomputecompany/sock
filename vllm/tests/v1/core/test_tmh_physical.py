# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.attention.ops.paged_attn import PagedAttention
from vllm.v1.core.tmh_policy import (
    TMHKVRuntimePolicy,
    TMHLayerShape,
    TMHPageRole,
    TMHPhysicalEvent,
    TMHPhysicalPageDescriptor,
    TMHStorageKind,
)
from vllm.v1.kv_cache_interface import TMHFullAttentionSpec
from vllm.v1.tmh_physical import (
    TMHPhysicalRuntime,
    reshape_tmh_physical_kv_cache,
    tmh_rocm_native_raw_attention_args,
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


def descriptor(
    *,
    request_id: str = "req-1",
    page_index: int,
    logical_block_id: int,
    role: TMHPageRole,
    storage: TMHStorageKind,
) -> TMHPhysicalPageDescriptor:
    return TMHPhysicalPageDescriptor(
        request_id=request_id,
        layer_name="model.layers.0.self_attn",
        logical_block_id=logical_block_id,
        page_index=page_index,
        role=role,
        storage=storage,
        prefix_cached=storage == TMHStorageKind.REQUEST_OVERLAY,
        k_quant_mode="raw" if role == TMHPageRole.HOT_RAW else "int8_per_token_head",
        v_quant_mode="raw" if role == TMHPageRole.HOT_RAW else "int8_per_token_head",
    )


def test_tmh_physical_runtime_maps_request_pages_to_canonical_and_overlay_slots():
    cache = make_physical_cache()
    runtime = TMHPhysicalRuntime()
    runtime.register_cache("model.layers.0.self_attn", cache)

    runtime.apply_events(
        [
            TMHPhysicalEvent(
                request_id="req-1",
                descriptors=(
                    descriptor(
                        page_index=1,
                        logical_block_id=1,
                        role=TMHPageRole.WARM_INT8_INT8,
                        storage=TMHStorageKind.CANONICAL,
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
    assert cache.canonical_role_by_logical_block[1].item() == int(
        TMHPageRole.WARM_INT8_INT8
    )
    assert cache.request_slot_by_row_page[0, 3].item() == 0

    runtime.apply_events(
        [
            TMHPhysicalEvent(
                request_id="req-1",
                descriptors=(),
                total_pages=0,
                recent_start_page=0,
                hot_pages=0,
                released_request_ids=("req-1",),
            )
        ],
        {},
    )

    runtime.apply_events(
        [
            TMHPhysicalEvent(
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
    assert cache.request_slot_by_row_page[1, 3].item() == 0


def test_tmh_policy_emits_release_descriptors_for_forgotten_canonical_pages():
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
    assert {
        descriptor.logical_block_id
        for descriptor in events[0].released_descriptors
    } == {0, 1, 2, 3}


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
            storage=TMHStorageKind.CANONICAL,
        )
        for page in range(raw_capacity)
    )
    runtime.apply_events(
        [
            TMHPhysicalEvent(
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
            TMHPhysicalEvent(
                request_id="req-1",
                descriptors=(),
                total_pages=0,
                recent_start_page=0,
                hot_pages=0,
                released_request_ids=("req-1",),
                released_descriptors=first_descriptors,
            )
        ],
        {},
    )

    runtime.apply_events(
        [
            TMHPhysicalEvent(
                request_id="req-2",
                descriptors=tuple(
                    descriptor(
                        request_id="req-2",
                        page_index=page,
                        logical_block_id=page + raw_capacity,
                        role=TMHPageRole.HOT_RAW,
                        storage=TMHStorageKind.CANONICAL,
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


def test_tmh_rocm_native_raw_attention_args_only_for_batches_without_warm_pages():
    cache = make_physical_cache()
    runtime = TMHPhysicalRuntime()
    runtime.register_cache("model.layers.0.self_attn", cache)
    runtime.apply_events(
        [
            TMHPhysicalEvent(
                request_id="req-1",
                descriptors=(
                    descriptor(
                        page_index=0,
                        logical_block_id=0,
                        role=TMHPageRole.PINNED_RAW,
                        storage=TMHStorageKind.CANONICAL,
                    ),
                    descriptor(
                        page_index=1,
                        logical_block_id=1,
                        role=TMHPageRole.HOT_RAW,
                        storage=TMHStorageKind.CANONICAL,
                    ),
                ),
                total_pages=2,
                recent_start_page=1,
                hot_pages=1,
            )
        ],
        {"req-1": 0},
    )

    class Meta:
        max_seq_len = 32
        seq_lens = [32]
        block_table = torch.empty((1, 2), dtype=torch.int32)

    native = tmh_rocm_native_raw_attention_args(cache, Meta())
    assert native is not None
    raw_kv_cache, block_table = native
    assert raw_kv_cache.data_ptr() == cache.raw_kv_cache.data_ptr()
    assert block_table.tolist() == [[0, 1]]

    Meta.max_seq_len = 33
    assert tmh_rocm_native_raw_attention_args(cache, Meta()) is None


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
