# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
from triton.runtime.jit import compute_cache_key

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.ops.tmh_triton_ops import (
    _get_tmh_attention_launch_kwargs,
    _get_tmh_mixed_tile_size,
    _get_tmh_query_block_shape,
    _tmh_mixed_attention_kernel,
    tmh_reshape_and_cache,
    tmh_physical_attention,
)
from vllm.v1.core.tmh_policy import (
    TMHKVRuntimePolicy,
    TMHPhysicalPageDescriptor,
    TMHStorageKind,
)
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
    decode_query_len = getattr(
        model_runner,
        "uniform_decode_query_len",
        getattr(model_runner, "decode_query_len", 1),
    )
    max_descriptor_pages = min(
        cache.request_slot_by_row_page.shape[1] for cache in caches
    )
    prompt_pages = min(max_descriptor_pages, max(1, min(4, max_descriptor_pages)))
    main_prompt_len = max(block_size, prompt_pages * block_size, decode_query_len + 1)
    prompt_lens = list(range(1, block_size + 1))
    if main_prompt_len not in prompt_lens:
        prompt_lens.append(main_prompt_len)
    long_decode_prompt_len = min(
        max_descriptor_pages * block_size,
        max(main_prompt_len, block_size * 32),
    )
    if long_decode_prompt_len not in prompt_lens:
        prompt_lens.append(long_decode_prompt_len)
    group_block_sizes = [
        group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups
    ]

    shape_index = 0
    warmed_any = False
    for prompt_len in prompt_lens:
        max_num_reqs = min(
            model_runner.scheduler_config.max_num_seqs,
            max(1, model_runner.max_num_tokens // max(prompt_len, decode_query_len)),
        )
        block_counts = [cdiv(prompt_len, size) for size in group_block_sizes]
        blocks_per_req = sum(block_counts)
        max_num_reqs = min(
            max_num_reqs,
            max(1, (kv_cache_config.num_blocks - 1) // max(1, blocks_per_req)),
        )
        if max_num_reqs <= 0:
            continue

        warmup_num_reqs = sorted(
            {
                1,
                *range(2, min(max_num_reqs, 4) + 1),
                max_num_reqs,
            }
        )

        for num_reqs in warmup_num_reqs:
            warmed_any = True
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
            req_ids = [
                f"_tmh_physical_warmup_{shape_index}_{i}_"
                for i in range(num_reqs)
            ]
            shape_index += 1
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
            physical_events = tmh_policy.take_physical_events()
            published_descriptors = tuple(
                descriptor
                for physical_event in physical_events
                for descriptor in physical_event.descriptors
            )
            runtime.apply_events(
                physical_events,
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
                _release_warmup_canonical_pages(runtime, published_descriptors)
                torch.accelerator.synchronize()

    if not warmed_any:
        logger.warning(
            "Skipping physical TMH warmup because no KV blocks are available."
        )
        return
    _run_tmh_correctness_smoke(
        model_runner=model_runner,
        runtime=runtime,
        kv_cache_config=kv_cache_config,
        caches=caches,
        block_size=block_size,
        group_block_sizes=group_block_sizes,
    )


def _record_tmh_descriptors(
    *,
    tmh_policy: TMHKVRuntimePolicy,
    total_tokens_by_req: dict[str, int],
    block_ids_by_req: dict[str, tuple[list[int], ...]],
) -> None:
    for req_id, total_tokens in total_tokens_by_req.items():
        tmh_policy.record_physical_descriptors_from_group_block_ids(
            request_id=req_id,
            total_tokens=total_tokens,
            logical_block_ids_by_group={
                group_id: tuple(block_ids)
                for group_id, block_ids in enumerate(block_ids_by_req[req_id])
            },
        )


def _release_warmup_canonical_pages(
    runtime,
    descriptors: tuple[TMHPhysicalPageDescriptor, ...],
) -> None:
    unique: dict[tuple[object, ...], TMHPhysicalPageDescriptor] = {}
    for descriptor in descriptors:
        if descriptor.storage != TMHStorageKind.CANONICAL:
            continue
        key = (
            descriptor.layer_name,
            descriptor.cache_group_id,
            descriptor.logical_block_id,
            descriptor.allocation_generation,
        )
        unique[key] = descriptor
    for descriptor in unique.values():
        runtime.release_descriptor(descriptor)


def _run_tmh_correctness_smoke(
    *,
    model_runner: "GPUModelRunner",
    runtime,
    kv_cache_config,
    caches: list[TMHPhysicalKVCache],
    block_size: int,
    group_block_sizes: list[int],
) -> None:
    max_pages = min(cache.request_slot_by_row_page.shape[1] for cache in caches)
    prompt_pages = min(3, max_pages)
    if prompt_pages < 3:
        logger.warning(
            "Skipping mixed TMH numerical smoke because fewer than three "
            "physical descriptor pages are configured."
        )
        return
    prompt_len = prompt_pages * block_size
    block_counts = [cdiv(prompt_len, size) for size in group_block_sizes]
    if sum(block_counts) >= kv_cache_config.num_blocks:
        logger.warning(
            "Skipping mixed TMH numerical smoke because too few logical "
            "blocks are available."
        )
        return
    req_id = "_tmh_physical_correctness_smoke_"
    next_block_id = 1
    blocks_by_group: list[list[int]] = []
    for count in block_counts:
        blocks_by_group.append(list(range(next_block_id, next_block_id + count)))
        next_block_id += count
    policy = TMHKVRuntimePolicy.from_kv_cache_config(kv_cache_config, block_size)
    _record_tmh_descriptors(
        tmh_policy=policy,
        total_tokens_by_req={req_id: prompt_len},
        block_ids_by_req={req_id: tuple(blocks_by_group)},
    )
    physical_events = policy.take_physical_events()
    published = tuple(
        descriptor
        for physical_event in physical_events
        for descriptor in physical_event.descriptors
    )
    runtime.apply_events(physical_events, {req_id: 0})
    try:
        for cache in caches:
            _warmup_physical_cache(
                model_runner=model_runner,
                cache=cache,
                num_reqs=1,
                prompt_len=prompt_len,
                block_size=block_size,
                validate_output=True,
            )
        torch.accelerator.synchronize()
        logger.info("Physical TMH deterministic numerical smoke passed.")
    finally:
        policy.forget_request(req_id)
        runtime.apply_events(policy.take_physical_events(), {})
        _release_warmup_canonical_pages(runtime, published)
        torch.accelerator.synchronize()


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
            cache.spec.tmh_late_layer,
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
    validate_output: bool = False,
) -> None:
    device = cache.raw_key.device
    num_kv_heads = cache.spec.num_kv_heads
    head_size = cache.spec.head_size
    head_size_v = cache.spec.head_size_v
    num_query_heads = getattr(model_runner, "num_query_heads", None)
    if num_query_heads is None:
        num_query_heads = model_runner.model_config.get_num_attention_heads(
            model_runner.parallel_config
        )
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

    # Attention implementations commonly pass Q/K/V as non-contiguous views
    # of one fused projection. Preserve that production stride class during
    # warmup; Triton may specialize attention launches on pointer/stride
    # properties even when logical shapes match contiguous synthetic tensors.
    fused_heads = num_query_heads + 2 * num_kv_heads
    fused_head_size = max(head_size, head_size_v)
    qkv_storage = torch.empty(
        (total_tokens, fused_heads, fused_head_size),
        device=device,
        dtype=cache.spec.dtype,
    )
    query = qkv_storage[:, :num_query_heads, :head_size]
    key = qkv_storage[
        :, num_query_heads : num_query_heads + num_kv_heads, :head_size
    ]
    value = qkv_storage[:, num_query_heads + num_kv_heads :, :head_size_v]
    if validate_output:
        _fill_deterministic(key, modulus=97, denominator=32.0)
        _fill_deterministic(value, modulus=89, denominator=29.0)
    else:
        key.normal_(mean=0.0, std=0.01)
        value.normal_(mean=0.0, std=0.01)
    tmh_reshape_and_cache(key, value, cache, slot_mapping, metadata, None)

    output = torch.empty(
        query.shape,
        device=device,
        dtype=cache.spec.dtype,
    )
    if validate_output:
        _fill_deterministic(query, modulus=83, denominator=31.0)
    else:
        query.normal_(mean=0.0, std=0.01)
    tmh_physical_attention(
        q=query,
        cache=cache,
        out=output,
        attn_metadata=metadata,
        seq_to_request_row=None,
        softmax_scale=1.0 / math.sqrt(head_size),
        causal=True,
        window_size=(-1, -1),
        softcap=0,
    )
    if validate_output:
        reference = _causal_reference(
            query=query,
            key=key,
            value=value,
            num_reqs=num_reqs,
            seq_len=prompt_len,
        )
        torch.testing.assert_close(
            output.float(),
            reference,
            atol=0.30,
            rtol=0.30,
            msg=lambda message: (
                "TMH prefill correctness smoke disagrees with the FP32 "
                f"reference: {message}"
            ),
        )

    decode_metadata = SimpleNamespace(
        num_actual_tokens=num_reqs,
        max_query_len=1,
        query_start_loc=torch.arange(0, num_reqs + 1, dtype=torch.int32, device=device),
        max_seq_len=max(prompt_len, block_size * 33) if num_reqs >= 2 else prompt_len,
        seq_lens=seq_lens,
        block_table=metadata.block_table,
        causal=True,
    )
    head_size_padded = 1 << (head_size - 1).bit_length()
    decode_metadata.seq_threshold_3D = num_reqs
    decode_metadata.num_par_softmax_segments = 16
    decode_metadata.softmax_segm_output = torch.empty(
        (
            num_reqs,
            num_query_heads,
            decode_metadata.num_par_softmax_segments,
            head_size_padded,
        ),
        dtype=torch.float32,
        device=device,
    )
    decode_metadata.softmax_segm_max = torch.empty(
        (num_reqs, num_query_heads, decode_metadata.num_par_softmax_segments),
        dtype=torch.float32,
        device=device,
    )
    decode_metadata.softmax_segm_expsum = torch.empty_like(
        decode_metadata.softmax_segm_max
    )
    decode_qkv_storage = torch.empty(
        (num_reqs, fused_heads, fused_head_size),
        device=device,
        dtype=cache.spec.dtype,
    )
    decode_query = decode_qkv_storage[:, :num_query_heads, :head_size]
    decode_output = torch.empty(
        decode_query.shape,
        device=device,
        dtype=cache.spec.dtype,
    )
    if validate_output:
        _fill_deterministic(decode_query, modulus=79, denominator=37.0)
    else:
        decode_query.normal_(mean=0.0, std=0.01)
    kv_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    tmh_physical_attention(
        q=decode_query,
        cache=cache,
        out=decode_output,
        attn_metadata=decode_metadata,
        seq_to_request_row=None,
        softmax_scale=1.0 / math.sqrt(head_size),
        causal=True,
        window_size=(-1, -1),
        softcap=0,
        kv_cache_dtype="auto",
        k_scale=kv_scale,
        v_scale=kv_scale,
    )
    if prompt_len >= block_size * 3:
        _force_compile_mixed_decode_2d_kernel(
            cache=cache,
            decode_query=decode_query,
            decode_output=decode_output,
            attn_metadata=decode_metadata,
            num_reqs=num_reqs,
        )
    if validate_output:
        reference_decode = _decode_reference(
            query=decode_query,
            key=key,
            value=value,
            num_reqs=num_reqs,
            seq_len=prompt_len,
        )
        torch.testing.assert_close(
            decode_output.float(),
            reference_decode,
            atol=0.30,
            rtol=0.30,
            msg=lambda message: (
                "TMH decode correctness smoke disagrees with the FP32 "
                f"reference: {message}"
            ),
        )


def _force_compile_mixed_decode_2d_kernel(
    *,
    cache: TMHPhysicalKVCache,
    decode_query: torch.Tensor,
    decode_output: torch.Tensor,
    attn_metadata,
    num_reqs: int,
) -> None:
    num_query_heads = decode_query.shape[1]
    num_kv_heads = cache.raw_key.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = decode_query.shape[2]
    block_size = cache.spec.block_size
    block_m, block_q = _get_tmh_query_block_shape(
        num_queries_per_kv=num_queries_per_kv,
        head_size=head_size,
        num_query_tokens=decode_query.shape[0],
        num_seqs=num_reqs,
    )
    tile_size = _get_tmh_mixed_tile_size(block_size=block_size)
    head_size_padded = 1 << (head_size - 1).bit_length()
    total_num_q_blocks = decode_query.shape[0] // block_q + num_reqs
    launch_kwargs = _get_tmh_attention_launch_kwargs(
        block_m=block_m,
        head_size=head_size,
    )
    packed_v = not cache.spec.tmh_late_layer
    query_storage = torch.empty(
        (decode_query.shape[0] + 1, decode_query.shape[1], decode_query.shape[2]),
        device=decode_query.device,
        dtype=decode_query.dtype,
    )
    output_storage = torch.empty(
        (decode_output.shape[0] + 1, decode_output.shape[1], decode_output.shape[2]),
        device=decode_output.device,
        dtype=decode_output.dtype,
    )
    seq_rows = torch.arange(num_reqs, dtype=torch.int32, device=decode_query.device)
    for offset in (0, 1):
        query_view = query_storage[offset : offset + decode_query.shape[0]]
        output_view = output_storage[offset : offset + decode_output.shape[0]]
        kwargs = dict(
            output_ptr=output_view,
            segm_output_ptr=output_view,
            segm_max_ptr=output_view,
            segm_expsum_ptr=output_view,
            query_ptr=query_view,
            raw_key_ptr=cache.raw_key,
            raw_value_ptr=cache.raw_value,
            warm_key_ptr=cache.warm_key,
            warm_value_ptr=cache.warm_value,
            warm_k_scale_ptr=cache.warm_k_scale,
            warm_v_scale_ptr=cache.warm_v_scale,
            request_slot_ptr=cache.request_slot_by_row_page,
            request_role_ptr=cache.request_role_by_row_page,
            seq_to_request_row_ptr=seq_rows,
            seq_lens_ptr=attn_metadata.seq_lens,
            query_start_len_ptr=attn_metadata.query_start_loc,
            sink_ptr=None,
            alibi_slopes_ptr=None,
            qq_bias_ptr=None,
            mm_prefix_range_ptr=None,
            scale=1.0 / math.sqrt(head_size),
            out_scale=1.0,
            softcap=0,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            request_stride=cache.request_slot_by_row_page.stride(0),
            query_stride_0=query_view.stride(0),
            query_stride_1=query_view.stride(1),
            output_stride_0=output_view.stride(0),
            output_stride_1=output_view.stride(1),
            qq_bias_stride_0=0,
            stride_raw_k_slot=cache.raw_key.stride(0),
            stride_raw_k_head=cache.raw_key.stride(1),
            stride_raw_k_dim_block=cache.raw_key.stride(2),
            stride_raw_k_tok=cache.raw_key.stride(3),
            stride_raw_k_x=cache.raw_key.stride(4),
            stride_raw_v_slot=cache.raw_value.stride(0),
            stride_raw_v_head=cache.raw_value.stride(1),
            stride_raw_v_dim=cache.raw_value.stride(2),
            stride_raw_v_tok=cache.raw_value.stride(3),
            stride_warm_k_slot=cache.warm_key.stride(0),
            stride_warm_k_tok=cache.warm_key.stride(1),
            stride_warm_k_head=cache.warm_key.stride(2),
            stride_warm_v_slot=cache.warm_value.stride(0),
            stride_warm_v_tok=cache.warm_value.stride(1),
            stride_warm_v_head=cache.warm_value.stride(2),
            stride_warm_ks_slot=cache.warm_k_scale.stride(0),
            stride_warm_ks_tok=cache.warm_k_scale.stride(1),
            stride_warm_ks_head=cache.warm_k_scale.stride(2),
            stride_warm_vs_slot=cache.warm_v_scale.stride(0),
            stride_warm_vs_tok=cache.warm_v_scale.stride(1),
            stride_warm_vs_head=cache.warm_v_scale.stride(2),
            BLOCK_SIZE=block_size,
            TILE_SIZE=tile_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=head_size_padded,
            BLOCK_Q=block_q,
            BLOCK_M=block_m,
            NUM_SEGMENTS_PER_SEQ=1,
            num_seqs=num_reqs,
            USE_ALIBI_SLOPES=False,
            USE_ALIBI_SQRT=False,
            USE_QQ_BIAS=False,
            USE_SOFTCAP=False,
            USE_SINKS=False,
            SLIDING_WINDOW=0,
            USE_MM_PREFIX=False,
            MAX_MM_RANGES=0,
            USE_FP8=False,
            WARM_VALUE_PACKED=packed_v,
            RAW_X=16 // query_view.element_size(),
            IS_3D=False,
            CHUNK_LOOKBACK=-1,
            CHUNK_SIZE=-1,
            **launch_kwargs,
        )
        kernel_cache, kernel_key_cache, _target, _backend, binder = (
            _tmh_mixed_attention_kernel.device_caches[query_view.device.index]
        )
        _bound_args, specialization, options = binder(**kwargs)
        key = compute_cache_key(kernel_key_cache, specialization, options)
        logger.info(
            "TMH explicit mixed warmup key offset=%d packed=%s key=%r",
            offset,
            packed_v,
            key,
        )
        _tmh_mixed_attention_kernel.run(
            grid=(total_num_q_blocks, num_kv_heads),
            warmup=True,
            output_ptr=output_view,
            segm_output_ptr=output_view,
            segm_max_ptr=output_view,
            segm_expsum_ptr=output_view,
            query_ptr=query_view,
            raw_key_ptr=cache.raw_key,
            raw_value_ptr=cache.raw_value,
            warm_key_ptr=cache.warm_key,
            warm_value_ptr=cache.warm_value,
            warm_k_scale_ptr=cache.warm_k_scale,
            warm_v_scale_ptr=cache.warm_v_scale,
            request_slot_ptr=cache.request_slot_by_row_page,
            request_role_ptr=cache.request_role_by_row_page,
            seq_to_request_row_ptr=seq_rows,
            seq_lens_ptr=attn_metadata.seq_lens,
            query_start_len_ptr=attn_metadata.query_start_loc,
            sink_ptr=None,
            alibi_slopes_ptr=None,
            qq_bias_ptr=None,
            mm_prefix_range_ptr=None,
            scale=1.0 / math.sqrt(head_size),
            out_scale=1.0,
            softcap=0,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            request_stride=cache.request_slot_by_row_page.stride(0),
            query_stride_0=query_view.stride(0),
            query_stride_1=query_view.stride(1),
            output_stride_0=output_view.stride(0),
            output_stride_1=output_view.stride(1),
            qq_bias_stride_0=0,
            stride_raw_k_slot=cache.raw_key.stride(0),
            stride_raw_k_head=cache.raw_key.stride(1),
            stride_raw_k_dim_block=cache.raw_key.stride(2),
            stride_raw_k_tok=cache.raw_key.stride(3),
            stride_raw_k_x=cache.raw_key.stride(4),
            stride_raw_v_slot=cache.raw_value.stride(0),
            stride_raw_v_head=cache.raw_value.stride(1),
            stride_raw_v_dim=cache.raw_value.stride(2),
            stride_raw_v_tok=cache.raw_value.stride(3),
            stride_warm_k_slot=cache.warm_key.stride(0),
            stride_warm_k_tok=cache.warm_key.stride(1),
            stride_warm_k_head=cache.warm_key.stride(2),
            stride_warm_v_slot=cache.warm_value.stride(0),
            stride_warm_v_tok=cache.warm_value.stride(1),
            stride_warm_v_head=cache.warm_value.stride(2),
            stride_warm_ks_slot=cache.warm_k_scale.stride(0),
            stride_warm_ks_tok=cache.warm_k_scale.stride(1),
            stride_warm_ks_head=cache.warm_k_scale.stride(2),
            stride_warm_vs_slot=cache.warm_v_scale.stride(0),
            stride_warm_vs_tok=cache.warm_v_scale.stride(1),
            stride_warm_vs_head=cache.warm_v_scale.stride(2),
            BLOCK_SIZE=block_size,
            TILE_SIZE=tile_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=head_size_padded,
            BLOCK_Q=block_q,
            BLOCK_M=block_m,
            NUM_SEGMENTS_PER_SEQ=1,
            num_seqs=num_reqs,
            USE_ALIBI_SLOPES=False,
            USE_ALIBI_SQRT=False,
            USE_QQ_BIAS=False,
            USE_SOFTCAP=False,
            USE_SINKS=False,
            SLIDING_WINDOW=0,
            USE_MM_PREFIX=False,
            MAX_MM_RANGES=0,
            USE_FP8=False,
            WARM_VALUE_PACKED=packed_v,
            RAW_X=16 // query_view.element_size(),
            IS_3D=False,
            CHUNK_LOOKBACK=-1,
            CHUNK_SIZE=-1,
            **launch_kwargs,
        )


def _fill_deterministic(
    tensor: torch.Tensor,
    *,
    modulus: int,
    denominator: float,
) -> None:
    values = torch.arange(tensor.numel(), device=tensor.device, dtype=torch.float32)
    values = ((values % modulus) - modulus // 2) / denominator
    tensor.copy_(values.reshape(tensor.shape).to(tensor.dtype))


def _causal_reference(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_reqs: int,
    seq_len: int,
) -> torch.Tensor:
    output = torch.empty_like(query, dtype=torch.float32)
    queries_per_kv = query.shape[1] // key.shape[1]
    scale = 1.0 / math.sqrt(query.shape[2])
    causal_mask = torch.triu(
        torch.ones((seq_len, seq_len), device=query.device, dtype=torch.bool),
        diagonal=1,
    )
    for request_index in range(num_reqs):
        start = request_index * seq_len
        stop = start + seq_len
        for query_head in range(query.shape[1]):
            kv_head = query_head // queries_per_kv
            scores = (
                query[start:stop, query_head].float()
                @ key[start:stop, kv_head].float().T
            ) * scale
            scores.masked_fill_(causal_mask, float("-inf"))
            output[start:stop, query_head] = torch.softmax(scores, dim=-1) @ value[
                start:stop, kv_head
            ].float()
    return output


def _decode_reference(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_reqs: int,
    seq_len: int,
) -> torch.Tensor:
    output = torch.empty_like(query, dtype=torch.float32)
    queries_per_kv = query.shape[1] // key.shape[1]
    scale = 1.0 / math.sqrt(query.shape[2])
    for request_index in range(num_reqs):
        start = request_index * seq_len
        stop = start + seq_len
        for query_head in range(query.shape[1]):
            kv_head = query_head // queries_per_kv
            scores = (
                query[request_index, query_head].float()
                @ key[start:stop, kv_head].float().T
            ) * scale
            output[request_index, query_head] = torch.softmax(scores, dim=-1) @ value[
                start:stop, kv_head
            ].float()
    return output
