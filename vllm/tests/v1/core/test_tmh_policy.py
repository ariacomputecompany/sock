# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config.cache import CacheConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock, get_request_block_hasher
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.tmh_policy import TMHKVRuntimePolicy, TMHStorageKind
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    TMHFullAttentionSpec,
)
from vllm.v1.request import Request


def make_config(policy: str = "accounting") -> KVCacheConfig:
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=4,
        head_size_v=4,
        dtype=torch.float16,
    )
    return KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[f"model.layers.{i}.self_attn" for i in range(6)],
                kv_cache_spec=spec,
            )
        ],
        tmh_kv_policy=policy,
        tmh_hot_budget_pct=25.0,
    )


def test_tmh_accounting_uses_live_allocator_blocks() -> None:
    policy = TMHKVRuntimePolicy.from_kv_cache_config(make_config(), 16)
    blocks = tuple([[KVCacheBlock(i) for i in range(1, 5)]])

    pressure = policy.record_allocation(
        request_id="req-1",
        total_tokens=64,
        prompt_tokens=16,
        blocks_by_group=blocks,
    )

    assert pressure is not None
    assert pressure.kv_layout == "tmh_fidelity_paged_kv"
    assert pressure.policy == "accounting"
    assert pressure.physical is False
    assert pressure.total_pages == 4
    assert pressure.hot_pages == 1
    assert pressure.recent_start_page == 3
    assert pressure.old_tokens == 32
    assert pressure.regular_live_bytes == 12288
    assert pressure.tmh_effective_bytes == 11776
    assert pressure.same_hot_uniform_int8_bytes == 12288
    assert round(pressure.warm_reduction_vs_uniform_int8_pct, 3) == 8.333
    assert round(pressure.total_reduction_vs_same_hot_uniform_int8_pct, 3) == 4.167


def test_tmh_policy_is_disabled_by_default() -> None:
    policy = TMHKVRuntimePolicy.from_kv_cache_config(make_config("off"), 16)

    assert policy.record_allocation(
        request_id="req-1",
        total_tokens=64,
        prompt_tokens=16,
        blocks_by_group=tuple([[KVCacheBlock(1)]]),
    ) is None


def test_cache_config_tmh_layout_selects_physical_policy() -> None:
    config = CacheConfig(kv_layout="tmh")

    assert config.tmh_kv_policy == "physical"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cache_dtype": "fp8"}, "cache_dtype='auto'"),
        ({"kv_offloading_size": 1.0}, "cannot be combined with KV offloading"),
    ],
)
def test_tmh_rejects_unsupported_cache_feature_combinations(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CacheConfig(kv_layout="tmh", **kwargs)


def test_tmh_physical_descriptors_are_prefix_cache_aware() -> None:
    policy = TMHKVRuntimePolicy.from_kv_cache_config(make_config("physical"), 16)
    blocks = [KVCacheBlock(i) for i in range(1, 5)]
    blocks[2].ref_cnt = 2
    blocks[3].ref_cnt = 2

    pressure = policy.record_allocation(
        request_id="req-physical",
        total_tokens=64,
        prompt_tokens=16,
        blocks_by_group=(blocks,),
    )

    assert pressure is not None
    assert pressure.physical is True
    events = policy.take_physical_events()
    assert len(events) == 1
    descriptors = {
        (descriptor.layer_name, descriptor.page_index): descriptor
        for descriptor in events[0].descriptors
    }
    layer = "model.layers.0.self_attn"
    assert descriptors[(layer, 0)].storage == TMHStorageKind.CANONICAL
    assert descriptors[(layer, 0)].prefix_cached is False
    assert descriptors[(layer, 2)].storage == TMHStorageKind.CANONICAL
    assert descriptors[(layer, 2)].prefix_cached is True
    assert descriptors[(layer, 3)].storage == TMHStorageKind.REQUEST_OVERLAY
    assert descriptors[(layer, 3)].prefix_cached is True


def test_tmh_physical_descriptors_can_be_recorded_from_warmup_block_ids() -> None:
    policy = TMHKVRuntimePolicy.from_kv_cache_config(make_config("physical"), 16)

    policy.record_physical_descriptors_from_block_ids(
        request_id="req-warmup",
        total_tokens=64,
        logical_block_ids=(1, 2, 3, 4),
        prefix_cached_page_indices=frozenset({3}),
    )

    events = policy.take_physical_events()
    assert len(events) == 1
    descriptors = {
        (descriptor.layer_name, descriptor.page_index): descriptor
        for descriptor in events[0].descriptors
    }
    layer = "model.layers.0.self_attn"
    assert descriptors[(layer, 1)].storage == TMHStorageKind.CANONICAL
    assert descriptors[(layer, 3)].storage == TMHStorageKind.REQUEST_OVERLAY


def test_tmh_physical_long_requests_expand_the_hot_raw_window_only_under_four_way_contention() -> None:
    policy = TMHKVRuntimePolicy.from_kv_cache_config(make_config("physical"), 16)

    policy.set_active_physical_request_ids({"req-medium"})

    medium = policy.record_allocation(
        request_id="req-medium",
        total_tokens=640,
        prompt_tokens=640,
        blocks_by_group=([KVCacheBlock(i) for i in range(40)],),
    )

    assert medium is not None
    assert medium.total_pages == 40
    assert medium.hot_pages == 20
    assert medium.recent_start_page == 20

    medium_events = policy.take_physical_events()
    assert len(medium_events) == 1
    medium_descriptors = {
        (descriptor.layer_name, descriptor.page_index): descriptor
        for descriptor in medium_events[0].descriptors
    }
    layer = "model.layers.0.self_attn"
    assert medium_descriptors[(layer, 19)].role.name == "WARM_INT8_INT4"
    assert medium_descriptors[(layer, 20)].role.name == "HOT_RAW"

    policy.set_active_physical_request_ids({"req-long-1", "req-long-2", "req-long-3"})

    long1 = policy.record_allocation(
        request_id="req-long-1",
        total_tokens=1024,
        prompt_tokens=1024,
        blocks_by_group=([KVCacheBlock(i) for i in range(64)],),
    )
    long2 = policy.record_allocation(
        request_id="req-long-2",
        total_tokens=1024,
        prompt_tokens=1024,
        blocks_by_group=([KVCacheBlock(i) for i in range(64, 128)],),
    )
    long3 = policy.record_allocation(
        request_id="req-long-3",
        total_tokens=1024,
        prompt_tokens=1024,
        blocks_by_group=([KVCacheBlock(i) for i in range(128, 192)],),
    )

    assert long1 is not None and long2 is not None and long3 is not None
    assert long1.hot_pages == 32
    assert long2.hot_pages == 32
    assert long3.hot_pages == 32

    policy.set_active_physical_request_ids({"req-long-1", "req-long-2", "req-long-3", "req-long-4"})

    long4 = policy.record_allocation(
        request_id="req-long-4",
        total_tokens=1024,
        prompt_tokens=1024,
        blocks_by_group=([KVCacheBlock(i) for i in range(192, 256)],),
    )

    assert long4 is not None
    assert long4.hot_pages == 64
    assert long4.recent_start_page == 0
    assert policy._pending_long_rebalance_request_ids == set()

    policy.set_active_physical_request_ids({"req-long-1", "req-long-2", "req-long-3", "req-long-4"})

    long1_updated = policy.record_allocation(
        request_id="req-long-1",
        total_tokens=1024,
        prompt_tokens=1024,
        blocks_by_group=([KVCacheBlock(i) for i in range(64)],),
    )

    assert long1_updated is not None
    assert long1_updated.hot_pages == 64
    assert long1_updated.recent_start_page == 0
    assert policy._pending_long_rebalance_request_ids == set()

    long_events = policy.take_physical_events()
    assert len(long_events) == 5
    long1_event = next(event for event in reversed(long_events) if event.request_id == "req-long-1")
    long1_descriptors = {
        (descriptor.layer_name, descriptor.page_index): descriptor
        for descriptor in long1_event.descriptors
    }
    long4_event = next(event for event in reversed(long_events) if event.request_id == "req-long-4")
    long4_descriptors = {
        (descriptor.layer_name, descriptor.page_index): descriptor
        for descriptor in long4_event.descriptors
    }
    assert long1_descriptors[(layer, 15)].role.name == "HOT_RAW"
    assert long1_descriptors[(layer, 16)].role.name == "HOT_RAW"
    assert long1_descriptors[(layer, 31)].role.name == "HOT_RAW"
    assert long4_descriptors[(layer, 16)].role.name == "HOT_RAW"


def test_tmh_monotonic_same_page_growth_is_fused_until_new_page() -> None:
    policy = TMHKVRuntimePolicy.from_kv_cache_config(make_config('physical'), 16)

    policy.record_physical_descriptors_from_block_ids(
        request_id='req-progress',
        total_tokens=1,
        logical_block_ids=(1,),
    )
    assert len(policy.take_physical_events()) == 1

    policy.record_physical_descriptors_from_block_ids(
        request_id='req-progress',
        total_tokens=2,
        logical_block_ids=(1,),
    )
    assert policy.take_physical_events() == []

    policy.record_physical_descriptors_from_block_ids(
        request_id='req-progress',
        total_tokens=16,
        logical_block_ids=(1,),
    )
    assert policy.take_physical_events() == []

    policy.record_physical_descriptors_from_block_ids(
        request_id='req-progress',
        total_tokens=17,
        logical_block_ids=(1, 2),
    )
    next_page_events = policy.take_physical_events()
    assert len(next_page_events) == 1
    descriptors = {
        (descriptor.layer_name, descriptor.page_index): descriptor
        for descriptor in next_page_events[0].descriptors
    }
    assert all(
        descriptor.valid_tokens == 16
        for descriptor in descriptors.values()
        if descriptor.page_index == 0
    )
    assert all(
        descriptor.valid_tokens == 1
        for descriptor in descriptors.values()
        if descriptor.page_index == 1
    )


def test_tmh_pending_same_request_delta_events_are_coalesced() -> None:
    policy = TMHKVRuntimePolicy.from_kv_cache_config(make_config('physical'), 16)

    policy.record_physical_descriptors_from_block_ids(
        request_id='req-coalesce',
        total_tokens=1,
        logical_block_ids=(1,),
    )
    policy.record_physical_descriptors_from_block_ids(
        request_id='req-coalesce',
        total_tokens=17,
        logical_block_ids=(1, 2),
    )

    events = policy.take_physical_events()
    assert len(events) == 1
    event = events[0]
    assert event.request_id == 'req-coalesce'
    assert event.sequence == 1
    assert event.expected_base_version == 0
    assert event.target_version == 1
    descriptors = {
        (descriptor.layer_name, descriptor.page_index): descriptor
        for descriptor in event.descriptors
    }
    assert all(
        descriptor.valid_tokens == 16
        for descriptor in descriptors.values()
        if descriptor.page_index == 0
    )
    assert all(
        descriptor.valid_tokens == 1
        for descriptor in descriptors.values()
        if descriptor.page_index == 1
    )


def test_tmh_active_requests_keep_same_request_delta_ordering() -> None:
    policy = TMHKVRuntimePolicy.from_kv_cache_config(make_config("physical"), 16)
    policy.set_active_physical_request_ids({"req-active"})

    policy.record_physical_descriptors_from_block_ids(
        request_id="req-active",
        total_tokens=1,
        logical_block_ids=(1,),
    )
    policy.record_physical_descriptors_from_block_ids(
        request_id="req-active",
        total_tokens=17,
        logical_block_ids=(1, 2),
    )

    events = policy.take_physical_events()
    assert len(events) == 2
    assert [event.sequence for event in events] == [1, 2]
    assert [event.expected_base_version for event in events] == [0, 1]
    assert [event.target_version for event in events] == [1, 2]

def test_tmh_physical_forget_request_releases_request_overlays() -> None:
    policy = TMHKVRuntimePolicy.from_kv_cache_config(make_config("physical"), 16)
    blocks = [KVCacheBlock(i) for i in range(1, 5)]
    blocks[3].ref_cnt = 2

    policy.record_allocation(
        request_id="req-release",
        total_tokens=64,
        prompt_tokens=16,
        blocks_by_group=(blocks,),
    )
    policy.take_physical_events()

    policy.forget_request("req-release")

    events = policy.take_physical_events()
    assert len(events) == 1
    assert events[0].released_request_ids == ("req-release",)


def test_kv_cache_manager_records_tmh_pressure_from_allocate_slots() -> None:
    manager = KVCacheManager(
        kv_cache_config=make_config(),
        max_model_len=128,
        scheduler_block_size=16,
        hash_block_size=16,
        enable_caching=True,
    )
    sampling_params = SamplingParams(max_tokens=17)
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    init_none_hash(sha256)
    request = Request(
        request_id="req-manager",
        prompt_token_ids=list(range(64)),
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(16, sha256),
    )

    allocated = manager.allocate_slots(request, num_new_tokens=64)

    assert allocated is not None
    pressure = manager.tmh_policy.latest_by_request["req-manager"]
    assert pressure.regular_live_bytes == 12288
    assert pressure.tmh_effective_bytes == 11776
    assert round(pressure.warm_reduction_vs_uniform_int8_pct, 3) == 8.333

    manager.free(request)

    assert "req-manager" not in manager.tmh_policy.latest_by_request
    assert "req-manager" not in manager.tmh_policy._regular_live_bytes_cache


def test_tmh_descriptors_use_their_own_cache_group_blocks() -> None:
    early = TMHFullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=4,
        head_size_v=4,
        dtype=torch.float16,
        tmh_late_layer=False,
    )
    late = TMHFullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=4,
        head_size_v=4,
        dtype=torch.float16,
        tmh_late_layer=True,
    )
    config = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["model.layers.0.self_attn"],
                kv_cache_spec=early,
            ),
            KVCacheGroupSpec(
                layer_names=["model.layers.5.self_attn"],
                kv_cache_spec=late,
            ),
        ],
        tmh_kv_policy="physical",
        tmh_hot_budget_pct=25.0,
    )
    policy = TMHKVRuntimePolicy.from_kv_cache_config(config, 16)
    group_zero = [KVCacheBlock(i, allocation_generation=3) for i in range(4)]
    group_one = [KVCacheBlock(i + 20, allocation_generation=7) for i in range(4)]

    policy.record_allocation(
        request_id="multi-group",
        total_tokens=64,
        prompt_tokens=64,
        blocks_by_group=(group_zero, group_one),
    )
    descriptors = policy.take_physical_events()[0].descriptors
    by_layer_page = {
        (item.layer_name, item.page_index): item for item in descriptors
    }
    early_page = by_layer_page[("model.layers.0.self_attn", 1)]
    late_page = by_layer_page[("model.layers.5.self_attn", 1)]
    assert (early_page.cache_group_id, early_page.logical_block_id) == (0, 1)
    assert early_page.allocation_generation == 3
    assert early_page.role.name == "WARM_INT8_INT4"
    assert (late_page.cache_group_id, late_page.logical_block_id) == (1, 21)
    assert late_page.allocation_generation == 7
    assert late_page.role.name == "WARM_INT8_INT8"


def test_tmh_policy_uses_global_late_boundary_after_group_serialization() -> None:
    serialized_group_spec = TMHFullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        head_size_v=64,
        dtype=torch.float16,
        # A merged/group-level flag is not authoritative for an individual
        # layer. The global boundary is stable across scheduler serialization.
        tmh_late_layer=False,
        tmh_late_layer_start=16,
    )
    config = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["model.layers.16.self_attn.attn"],
                kv_cache_spec=serialized_group_spec,
            )
        ],
        tmh_kv_policy="physical",
        tmh_hot_budget_pct=25.0,
    )
    policy = TMHKVRuntimePolicy.from_kv_cache_config(config, 16)
    policy.record_allocation(
        request_id="late-boundary",
        total_tokens=64,
        prompt_tokens=64,
        blocks_by_group=([KVCacheBlock(i) for i in range(4)],),
    )
    descriptor = next(
        item
        for item in policy.take_physical_events()[0].descriptors
        if item.page_index == 1
    )
    assert descriptor.role.name == "WARM_INT8_INT8"


def test_tmh_logical_block_generation_releases_stale_physical_identity() -> None:
    policy = TMHKVRuntimePolicy.from_kv_cache_config(make_config("physical"), 16)
    first = KVCacheBlock(5, allocation_generation=1)
    policy.record_allocation(
        request_id="first",
        total_tokens=16,
        prompt_tokens=16,
        blocks_by_group=([first],),
    )
    policy.take_physical_events()
    policy.forget_request("first")
    policy.take_physical_events()

    reused = KVCacheBlock(5, allocation_generation=2)
    policy.record_allocation(
        request_id="second",
        total_tokens=16,
        prompt_tokens=16,
        blocks_by_group=([reused],),
    )
    event = policy.take_physical_events()[0]
    assert {item.allocation_generation for item in event.released_descriptors} == {1}
    assert {item.allocation_generation for item in event.descriptors} == {2}
