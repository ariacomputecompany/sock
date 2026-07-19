# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
    assert pressure.tmh_effective_bytes == 8704
    assert pressure.same_hot_uniform_int8_bytes == 9216
    assert round(pressure.warm_reduction_vs_uniform_int8_pct, 3) == 16.667
    assert round(pressure.total_reduction_vs_same_hot_uniform_int8_pct, 3) == 5.556


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
    assert pressure.tmh_effective_bytes == 8704
    assert round(pressure.warm_reduction_vs_uniform_int8_pct, 3) == 16.667

    manager.free(request)

    assert "req-manager" not in manager.tmh_policy.latest_by_request
    assert "req-manager" not in manager.tmh_policy._regular_live_bytes_cache
