# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.core.tmh_policy import (
    TMHPageRole,
    TMHPhysicalEvent,
    TMHPhysicalPageDescriptor,
    TMHStorageKind,
)
from vllm.v1.kv_cache_interface import TMHFullAttentionSpec
from vllm.v1.tmh_physical import (
    TMHPhysicalRuntime,
    reshape_tmh_physical_kv_cache,
)


def make_physical_cache():
    spec = TMHFullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=4,
        head_size_v=4,
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
        head_size=4,
        head_size_v=4,
        dtype=torch.float16,
        tmh_hot_budget_pct=25.0,
        tmh_max_num_seqs=1024,
        tmh_max_model_pages=16,
    )

    raw_pages, warm_pages = spec.physical_pool_page_counts(29693)

    assert raw_pages > 1024
    assert warm_pages > 0
    assert raw_pages + warm_pages == 29693


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

    assert cache.request_block_by_row_page[0, 1].item() == 1
    assert cache.request_role_by_row_page[0, 1].item() == int(TMHPageRole.WARM_INT8_INT8)
    assert cache.request_storage_by_row_page[0, 1].item() == int(TMHStorageKind.CANONICAL)
    assert cache.canonical_role_by_logical_block[1].item() == int(
        TMHPageRole.WARM_INT8_INT8
    )
    assert cache.request_block_by_row_page[0, 3].item() == 3
    assert cache.request_role_by_row_page[0, 3].item() == int(TMHPageRole.HOT_RAW)
    assert cache.request_storage_by_row_page[0, 3].item() == int(
        TMHStorageKind.REQUEST_OVERLAY
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
    assert cache.request_block_by_row_page[1, 3].item() == 3
