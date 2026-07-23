# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.ops.tmh_triton_ops import (
    tmh_reshape_and_cache,
    tmh_unified_attention,
)
from vllm.v1.core.tmh_policy import TMHKVRuntimePolicy
from vllm.v1.tmh_physical import TMHPhysicalKVCache

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


@torch.inference_mode()
def warmup_tmh_physical_kernels(model_runner: "GPUModelRunner") -> None:
    """Compile physical TMH kernels without forcing model-layer warmup.

    Physical TMH warmup owns descriptor application plus the TMH cache-writer and
    layout-aware attention kernels. Dense, quantized linear, and MoE kernels are
    materialized by the normal runtime warmup/request path instead of being
    pulled into this KV-specific warmup through a synthetic full-model decode.
    """

    if model_runner.is_pooling_model:
        return

    kv_cache_config = model_runner.kv_cache_config
    if kv_cache_config.tmh_kv_policy != "physical":
        raise RuntimeError(
            "warmup_tmh_physical_kernels requires physical TMH KV policy."
        )
    runtime = getattr(model_runner, "tmh_physical_runtime", None)
    if runtime is None:
        raise RuntimeError("physical TMH KV policy requires TMHPhysicalRuntime.")
    if not kv_cache_config.kv_cache_groups:
        return
    caches = _representative_physical_caches(runtime)
    if not caches:
        return

    block_size = kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
    decode_query_len = model_runner.uniform_decode_query_len
    max_descriptor_pages = min(
        cache.request_slot_by_row_page.shape[1] for cache in caches
    )
    prompt_pages = min(max_descriptor_pages, max(1, min(4, max_descriptor_pages)))
    prompt_len = max(block_size, prompt_pages * block_size, decode_query_len + 1)
    max_num_reqs = min(
        model_runner.scheduler_config.max_num_seqs,
        max(1, model_runner.max_num_tokens // max(prompt_len, decode_query_len)),
    )
    group_block_sizes = [
        group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups
    ]
    block_counts = [cdiv(prompt_len, size) for size in group_block_sizes]
    blocks_per_req = sum(block_counts)
    max_num_reqs = min(
        max_num_reqs,
        max(1, (kv_cache_config.num_blocks - 1) // max(1, blocks_per_req)),
    )
    if max_num_reqs <= 0:
        logger.warning(
            "Skipping physical TMH warmup because no KV blocks are available."
        )
        return

    warmup_num_reqs = [1]
    if max_num_reqs > 1:
        warmup_num_reqs.append(max_num_reqs)

    for shape_index, num_reqs in enumerate(warmup_num_reqs):
        logger.info(
            "Warming physical TMH kernels directly with num_reqs=%d "
            "prompt_len=%d decode_query_len=%d block_size=%d.",
            num_reqs,
            prompt_len,
            decode_query_len,
            block_size,
        )
        tmh_policy = TMHKVRuntimePolicy.from_kv_cache_config(
            kv_cache_config, block_size
        )
        req_ids = [f"_tmh_physical_warmup_{shape_index}_{i}_" for i in range(num_reqs)]
        next_block_id = 1

        def alloc_blocks(num_blocks: int) -> list[int]:
            nonlocal next_block_id
            block_ids = list(range(next_block_id, next_block_id + num_blocks))
            next_block_id += num_blocks
            return block_ids

        blocks_by_req: dict[str, tuple[list[int], ...]] = {}
        for req_id in req_ids:
            blocks_by_req[req_id] = tuple(alloc_blocks(n) for n in block_counts)

        _record_tmh_descriptors(
            tmh_policy=tmh_policy,
            total_tokens_by_req={req_id: prompt_len for req_id in req_ids},
            block_ids_by_req=blocks_by_req,
        )
        runtime.apply_events(
            tmh_policy.take_physical_events(),
            {req_id: index for index, req_id in enumerate(req_ids)},
        )

        try:
            for cache in caches:
                _warmup_physical_cache(
                    model_runner=model_runner,
                    cache=cache,
                    num_reqs=num_reqs,
                    prompt_len=prompt_len,
                    block_size=block_size,
                )
        finally:
            for req_id in req_ids:
                tmh_policy.forget_request(req_id)
            runtime.apply_events(tmh_policy.take_physical_events(), {})
            torch.accelerator.synchronize()


def _record_tmh_descriptors(
    *,
    tmh_policy: TMHKVRuntimePolicy,
    total_tokens_by_req: dict[str, int],
    block_ids_by_req: dict[str, tuple[list[int], ...]],
) -> None:
    for req_id, total_tokens in total_tokens_by_req.items():
        tmh_policy.record_physical_descriptors_from_block_ids(
            request_id=req_id,
            total_tokens=total_tokens,
            logical_block_ids=tuple(block_ids_by_req[req_id][0]),
        )


def _representative_physical_caches(runtime) -> list[TMHPhysicalKVCache]:
    caches_by_shape: dict[tuple[object, ...], TMHPhysicalKVCache] = {}
    for cache in runtime._caches.values():
        if not isinstance(cache, TMHPhysicalKVCache):
            continue
        shape = (
            cache.spec.block_size,
            cache.spec.num_kv_heads,
            cache.spec.head_size,
            cache.spec.head_size_v,
            cache.spec.dtype,
            cache.warm_value.shape[-1],
        )
        caches_by_shape.setdefault(shape, cache)
    return list(caches_by_shape.values())


def _warmup_physical_cache(
    *,
    model_runner: "GPUModelRunner",
    cache: TMHPhysicalKVCache,
    num_reqs: int,
    prompt_len: int,
    block_size: int,
) -> None:
    device = cache.raw_key.device
    num_kv_heads = cache.spec.num_kv_heads
    head_size = cache.spec.head_size
    head_size_v = cache.spec.head_size_v
    num_query_heads = model_runner.num_query_heads
    if num_query_heads % num_kv_heads != 0:
        raise RuntimeError(
            "TMH physical warmup requires query heads to be divisible by KV heads."
        )

    total_tokens = num_reqs * prompt_len
    query_start_loc = torch.arange(
        0,
        total_tokens + 1,
        prompt_len,
        dtype=torch.int32,
        device=device,
    )
    seq_lens = torch.full((num_reqs,), prompt_len, dtype=torch.int32, device=device)
    slot_mapping = torch.arange(total_tokens, dtype=torch.int64, device=device)
    metadata = SimpleNamespace(
        num_actual_tokens=total_tokens,
        max_query_len=prompt_len,
        query_start_loc=query_start_loc,
        max_seq_len=prompt_len,
        seq_lens=seq_lens,
        block_table=torch.empty(
            (num_reqs, cdiv(prompt_len, block_size)),
            dtype=torch.int32,
            device=device,
        ),
        causal=True,
    )

    key = torch.empty(
        (total_tokens, num_kv_heads, head_size),
        device=device,
        dtype=cache.spec.dtype,
    )
    value = torch.empty(
        (total_tokens, num_kv_heads, head_size_v),
        device=device,
        dtype=cache.spec.dtype,
    )
    key.normal_(mean=0.0, std=0.01)
    value.normal_(mean=0.0, std=0.01)
    tmh_reshape_and_cache(key, value, cache, slot_mapping, metadata, None)

    query = torch.empty(
        (total_tokens, num_query_heads, head_size),
        device=device,
        dtype=cache.spec.dtype,
    )
    output = torch.empty_like(query)
    query.normal_(mean=0.0, std=0.01)
    tmh_unified_attention(
        q=query,
        cache=cache,
        out=output,
        attn_metadata=metadata,
        seq_to_request_row=None,
        softmax_scale=1.0 / math.sqrt(head_size),
        causal=True,
        window_size=(-1, -1),
        softcap=0.0,
    )

    decode_metadata = SimpleNamespace(
        num_actual_tokens=num_reqs,
        max_query_len=1,
        query_start_loc=torch.arange(0, num_reqs + 1, dtype=torch.int32, device=device),
        max_seq_len=prompt_len,
        seq_lens=seq_lens,
        block_table=metadata.block_table,
        causal=True,
    )
    decode_query = torch.empty(
        (num_reqs, num_query_heads, head_size),
        device=device,
        dtype=cache.spec.dtype,
    )
    decode_output = torch.empty_like(decode_query)
    decode_query.normal_(mean=0.0, std=0.01)
    tmh_unified_attention(
        q=decode_query,
        cache=cache,
        out=decode_output,
        attn_metadata=decode_metadata,
        seq_to_request_row=None,
        softmax_scale=1.0 / math.sqrt(head_size),
        causal=True,
        window_size=(-1, -1),
        softcap=0.0,
    )
