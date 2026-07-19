# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for physical TMH fidelity-paged KV."""

from __future__ import annotations

from typing import Any

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.int4_per_token_head import (
    pack_int4_nibbles,
    unpack_int4_nibbles,
)
from vllm.v1.attention.ops.triton_attention_helpers import (
    apply_alibi_to_score,
    apply_softcap,
    compute_kv_seq_mask,
    compute_tile_loop_bounds,
    find_seq_idx,
    init_softmax_M,
    load_qq_bias_tile,
    resolve_seq_and_query_len,
    softmax_step,
)

TMH_ROLE_PINNED_RAW = 0
TMH_ROLE_HOT_RAW = 1
TMH_ROLE_WARM_INT8_INT4 = 2
TMH_ROLE_WARM_INT8_INT8 = 3


@triton.jit
def _pack_scale_zp(scale, zp):
    scale_bits = scale.to(tl.int32, bitcast=True)
    zp_bits = zp.to(tl.int32) & 0xF
    return ((scale_bits & -16) | zp_bits).to(tl.float32, bitcast=True)


@triton.jit
def _unpack_scale_zp(packed_scale):
    bits = packed_scale.to(tl.int32, bitcast=True)
    zp = (bits & 0xF).to(tl.float32)
    scale = (bits & -16).to(tl.float32, bitcast=True)
    return scale, zp


@triton.jit
def _tmh_reshape_and_cache_kernel(
    key_ptr,
    value_ptr,
    raw_key_ptr,
    raw_value_ptr,
    warm_key_ptr,
    warm_value_ptr,
    warm_k_scale_ptr,
    warm_v_scale_ptr,
    request_role_ptr,
    request_slot_ptr,
    seq_to_request_row_ptr,
    query_start_len_ptr,
    seq_lens_ptr,
    slot_mapping_ptr,
    stride_key_tok: tl.int64,
    stride_key_head: tl.int64,
    stride_val_tok: tl.int64,
    stride_val_head: tl.int64,
    stride_raw_k_slot: tl.int64,
    stride_raw_k_tok: tl.int64,
    stride_raw_k_head: tl.int64,
    stride_raw_v_slot: tl.int64,
    stride_raw_v_tok: tl.int64,
    stride_raw_v_head: tl.int64,
    stride_warm_k_slot: tl.int64,
    stride_warm_k_tok: tl.int64,
    stride_warm_k_head: tl.int64,
    stride_warm_v_slot: tl.int64,
    stride_warm_v_tok: tl.int64,
    stride_warm_v_head: tl.int64,
    stride_warm_ks_slot: tl.int64,
    stride_warm_ks_tok: tl.int64,
    stride_warm_ks_head: tl.int64,
    stride_warm_vs_slot: tl.int64,
    stride_warm_vs_tok: tl.int64,
    stride_warm_vs_head: tl.int64,
    request_stride: tl.int64,
    num_seqs: tl.int32,
    block_size: tl.constexpr,
    head_size: tl.constexpr,
    head_size_v: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    WARM_VALUE_PACKED: tl.constexpr,
):
    tok = tl.program_id(0)
    head = tl.program_id(1)

    logical_slot = tl.load(slot_mapping_ptr + tok).to(tl.int64)
    if logical_slot < 0:
        return

    seq_idx = find_seq_idx(query_start_len_ptr, tok, num_seqs, 1, False)
    req_row = tl.load(seq_to_request_row_ptr + seq_idx).to(tl.int64)
    cur_start = tl.load(query_start_len_ptr + seq_idx)
    cur_stop = tl.load(query_start_len_ptr + seq_idx + 1)
    query_len = cur_stop - cur_start
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    abs_pos = (seq_len - query_len) + (tok - cur_start)
    page_index = abs_pos // block_size
    offset_in_page = abs_pos % block_size

    desc_idx = req_row * request_stride + page_index
    role = tl.load(request_role_ptr + desc_idx).to(tl.int32)
    physical_slot = tl.load(request_slot_ptr + desc_idx).to(tl.int64)
    if physical_slot < 0:
        return

    offs = tl.arange(0, HEAD_SIZE_PADDED)
    k_mask = offs < head_size
    v_mask = offs < head_size_v
    key_base = key_ptr + tok * stride_key_tok + head * stride_key_head
    val_base = value_ptr + tok * stride_val_tok + head * stride_val_head
    key_vals = tl.load(key_base + offs, mask=k_mask, other=0.0).to(tl.float32)
    val_vals = tl.load(val_base + offs, mask=v_mask, other=0.0).to(tl.float32)

    is_raw = (role == 0) | (role == 1)
    if is_raw:
        raw_k_offset = (
            physical_slot * stride_raw_k_slot
            + offset_in_page * stride_raw_k_tok
            + head * stride_raw_k_head
            + offs
        )
        raw_v_offset = (
            physical_slot * stride_raw_v_slot
            + offset_in_page * stride_raw_v_tok
            + head * stride_raw_v_head
            + offs
        )
        tl.store(raw_key_ptr + raw_k_offset, key_vals, mask=k_mask)
        tl.store(raw_value_ptr + raw_v_offset, val_vals, mask=v_mask)
    else:
        k_absmax = tl.max(tl.abs(tl.where(k_mask, key_vals, 0.0)))
        k_scale = tl.maximum(k_absmax / 127.0, 1e-6)
        k_q = key_vals * (1.0 / k_scale)
        k_q = tl.where(k_q >= 0, k_q + 0.5, k_q - 0.5)
        k_q = tl.clamp(k_q, -128.0, 127.0)
        warm_k_offset = (
            physical_slot * stride_warm_k_slot
            + offset_in_page * stride_warm_k_tok
            + head * stride_warm_k_head
            + offs
        )
        tl.store(warm_key_ptr + warm_k_offset, k_q, mask=k_mask)
        scale_offset = (
            physical_slot * stride_warm_ks_slot
            + offset_in_page * stride_warm_ks_tok
            + head * stride_warm_ks_head
        )
        tl.store(warm_k_scale_ptr + scale_offset, k_scale)

        warm_v_offset = (
            physical_slot * stride_warm_v_slot
            + offset_in_page * stride_warm_v_tok
            + head * stride_warm_v_head
            + offs
        )
        v_scale_offset = (
            physical_slot * stride_warm_vs_slot
            + offset_in_page * stride_warm_vs_tok
            + head * stride_warm_vs_head
        )
        if role == 3:
            v_absmax = tl.max(tl.abs(tl.where(v_mask, val_vals, 0.0)))
            v_scale = tl.maximum(v_absmax / 127.0, 1e-6)
            v_q = val_vals * (1.0 / v_scale)
            v_q = tl.where(v_q >= 0, v_q + 0.5, v_q - 0.5)
            v_q = tl.clamp(v_q, -128.0, 127.0)
            tl.store(warm_value_ptr + warm_v_offset, v_q, mask=v_mask)
            tl.store(warm_v_scale_ptr + v_scale_offset, v_scale)
        elif WARM_VALUE_PACKED:
            packed_offs = tl.arange(0, HEAD_SIZE_PADDED // 2)
            even_offs = packed_offs * 2
            odd_offs = even_offs + 1
            even_mask = even_offs < head_size_v
            odd_mask = odd_offs < head_size_v
            v_even = tl.load(val_base + even_offs, mask=even_mask, other=0.0).to(
                tl.float32
            )
            v_odd = tl.load(val_base + odd_offs, mask=odd_mask, other=0.0).to(
                tl.float32
            )
            v_min = tl.minimum(
                tl.min(tl.where(even_mask, v_even, float("inf"))),
                tl.min(tl.where(odd_mask, v_odd, float("inf"))),
            )
            v_max = tl.maximum(
                tl.max(tl.where(even_mask, v_even, float("-inf"))),
                tl.max(tl.where(odd_mask, v_odd, float("-inf"))),
            )
            v4_scale = tl.maximum((v_max - v_min) / 15.0, 1e-6)
            v_zp = tl.clamp(
                tl.where(
                    -v_min / v4_scale >= 0,
                    (-v_min / v4_scale + 0.5).to(tl.int32),
                    (-v_min / v4_scale - 0.5).to(tl.int32),
                ).to(tl.float32),
                0.0,
                15.0,
            )
            inv_v4 = 1.0 / v4_scale
            v_even_q = tl.clamp(
                tl.where(
                    v_even * inv_v4 + v_zp >= 0,
                    (v_even * inv_v4 + v_zp + 0.5).to(tl.int32),
                    (v_even * inv_v4 + v_zp - 0.5).to(tl.int32),
                ).to(tl.float32),
                0.0,
                15.0,
            )
            v_odd_q = tl.clamp(
                tl.where(
                    v_odd * inv_v4 + v_zp >= 0,
                    (v_odd * inv_v4 + v_zp + 0.5).to(tl.int32),
                    (v_odd * inv_v4 + v_zp - 0.5).to(tl.int32),
                ).to(tl.float32),
                0.0,
                15.0,
            )
            packed = pack_int4_nibbles(v_even_q.to(tl.uint8), v_odd_q.to(tl.uint8))
            packed_offset = (
                physical_slot * stride_warm_v_slot
                + offset_in_page * stride_warm_v_tok
                + head * stride_warm_v_head
                + packed_offs
            )
            tl.store(
                warm_value_ptr + packed_offset,
                packed,
                mask=packed_offs < (head_size_v // 2),
            )
            tl.store(
                warm_v_scale_ptr + v_scale_offset,
                _pack_scale_zp(v4_scale, v_zp),
            )


@triton.jit
def _tmh_unified_attention_kernel(
    output_ptr,
    query_ptr,
    raw_key_ptr,
    raw_value_ptr,
    warm_key_ptr,
    warm_value_ptr,
    warm_k_scale_ptr,
    warm_v_scale_ptr,
    request_role_ptr,
    request_slot_ptr,
    seq_to_request_row_ptr,
    seq_lens_ptr,
    query_start_len_ptr,
    sink_ptr,
    alibi_slopes_ptr,
    qq_bias_ptr,
    mm_prefix_range_ptr,
    scale,
    out_scale,
    softcap,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    request_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    qq_bias_stride_0: tl.int64,
    stride_raw_k_slot: tl.int64,
    stride_raw_k_tok: tl.int64,
    stride_raw_k_head: tl.int64,
    stride_raw_v_slot: tl.int64,
    stride_raw_v_tok: tl.int64,
    stride_raw_v_head: tl.int64,
    stride_warm_k_slot: tl.int64,
    stride_warm_k_tok: tl.int64,
    stride_warm_k_head: tl.int64,
    stride_warm_v_slot: tl.int64,
    stride_warm_v_tok: tl.int64,
    stride_warm_v_head: tl.int64,
    stride_warm_ks_slot: tl.int64,
    stride_warm_ks_tok: tl.int64,
    stride_warm_ks_head: tl.int64,
    stride_warm_vs_slot: tl.int64,
    stride_warm_vs_tok: tl.int64,
    stride_warm_vs_head: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    num_seqs: tl.int32,
    USE_ALIBI_SLOPES: tl.constexpr,
    USE_ALIBI_SQRT: tl.constexpr,
    USE_QQ_BIAS: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    MAX_MM_RANGES: tl.constexpr,
    USE_FP8: tl.constexpr,
    WARM_VALUE_PACKED: tl.constexpr,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    (
        seq_idx,
        q_block_local_idx,
        cur_start,
        query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )
    if q_block_local_idx * BLOCK_Q >= query_len:
        return

    req_row = tl.load(seq_to_request_row_ptr + seq_idx).to(tl.int64)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv
    query_offset_0 = cur_start + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_mask_0 = tl.where(query_pos < query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)
    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)

    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    M = init_softmax_M(
        sink_ptr,
        query_offset_1,
        query_mask_1,
        0,
        BLOCK_M,
        USE_SINKS,
        False,
    )
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)
    context_len = seq_len - query_len

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1,
            mask=query_mask_1,
            other=0.0,
        )
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0

    loop_lo, loop_hi, max_seq_prefix_len = compute_tile_loop_bounds(
        context_len,
        seq_len,
        query_len,
        q_block_local_idx,
        0,
        0,
        TILE_SIZE,
        BLOCK_M,
        BLOCK_Q,
        num_queries_per_kv,
        SLIDING_WINDOW,
        USE_MM_PREFIX,
        False,
    )

    for j in range(loop_lo, loop_hi):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len
        page_index = (j * TILE_SIZE) // BLOCK_SIZE
        offset_in_page = offs_t
        desc_idx = req_row * request_stride + page_index
        role = tl.load(
            request_role_ptr + desc_idx,
            mask=(j * TILE_SIZE) < max_seq_prefix_len,
            other=-1,
        ).to(tl.int32)
        physical_slot = tl.load(
            request_slot_ptr + desc_idx,
            mask=(j * TILE_SIZE) < max_seq_prefix_len,
            other=-1,
        ).to(tl.int64)
        is_raw = (role == 0) | (role == 1)
        is_warm = (role == 2) | (role == 3)

        if is_raw:
            raw_k_offset = (
                physical_slot * stride_raw_k_slot
                + offset_in_page[None, :] * stride_raw_k_tok
                + kv_head_idx * stride_raw_k_head
                + offs_d[:, None]
            )
            raw_v_offset = (
                physical_slot * stride_raw_v_slot
                + offset_in_page[:, None] * stride_raw_v_tok
                + kv_head_idx * stride_raw_v_head
                + offs_d[None, :]
            )
            K = tl.load(
                raw_key_ptr + raw_k_offset,
                mask=dim_mask[:, None] & tile_mask[None, :],
                other=0.0,
            ).to(Q.dtype)
            V = tl.load(
                raw_value_ptr + raw_v_offset,
                mask=dim_mask[None, :] & tile_mask[:, None],
                other=0.0,
            ).to(Q.dtype)
        elif is_warm:
            warm_k_offset = (
                physical_slot * stride_warm_k_slot
                + offset_in_page[None, :] * stride_warm_k_tok
                + kv_head_idx * stride_warm_k_head
                + offs_d[:, None]
            )
            warm_ks_idx = (
                physical_slot * stride_warm_ks_slot
                + offset_in_page * stride_warm_ks_tok
                + kv_head_idx * stride_warm_ks_head
            )
            warm_k_scale = tl.load(
                warm_k_scale_ptr + warm_ks_idx,
                mask=tile_mask,
                other=1.0,
            )
            K = (
                tl.load(
                    warm_key_ptr + warm_k_offset,
                    mask=dim_mask[:, None] & tile_mask[None, :],
                    other=0,
                ).to(tl.float32)
                * warm_k_scale[None, :]
            ).to(Q.dtype)

            warm_vs_idx = (
                physical_slot * stride_warm_vs_slot
                + offset_in_page * stride_warm_vs_tok
                + kv_head_idx * stride_warm_vs_head
            )
            warm_v_scale = tl.load(
                warm_v_scale_ptr + warm_vs_idx,
                mask=tile_mask,
                other=1.0,
            )
            if WARM_VALUE_PACKED:
                packed_offs = offs_d // 2
                packed_offset = (
                    physical_slot * stride_warm_v_slot
                    + offset_in_page[:, None] * stride_warm_v_tok
                    + kv_head_idx * stride_warm_v_head
                    + packed_offs[None, :]
                )
                packed = tl.load(
                    warm_value_ptr + packed_offset,
                    mask=dim_mask[None, :] & tile_mask[:, None],
                    other=0,
                )
                lo, hi = unpack_int4_nibbles(packed)
                nibble = tl.where((offs_d[None, :] % 2) == 0, lo, hi).to(tl.float32)
                v4_scale, v4_zp = _unpack_scale_zp(warm_v_scale)
                V = ((nibble - v4_zp[:, None]) * v4_scale[:, None]).to(Q.dtype)
            else:
                warm_v_offset = (
                    physical_slot * stride_warm_v_slot
                    + offset_in_page[:, None] * stride_warm_v_tok
                    + kv_head_idx * stride_warm_v_head
                    + offs_d[None, :]
                )
                V = (
                    tl.load(
                        warm_value_ptr + warm_v_offset,
                        mask=dim_mask[None, :] & tile_mask[:, None],
                        other=0,
                    ).to(tl.float32)
                    * warm_v_scale[:, None]
                ).to(Q.dtype)
        else:
            K = tl.zeros([HEAD_SIZE_PADDED, TILE_SIZE], dtype=tl.float32).to(Q.dtype)
            V = tl.zeros([TILE_SIZE, HEAD_SIZE_PADDED], dtype=tl.float32).to(Q.dtype)

        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = compute_kv_seq_mask(
            query_abs_pos,
            seq_offset,
            seq_idx,
            seq_len,
            mm_prefix_range_ptr,
            SLIDING_WINDOW,
            USE_MM_PREFIX,
            MAX_MM_RANGES,
        )
        S = tl.dot(Q, K) * scale
        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)
        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
            S,
            float("-inf"),
        )
        if USE_ALIBI_SLOPES:
            S = apply_alibi_to_score(
                S, alibi_slope, seq_offset, context_len, query_pos, USE_ALIBI_SQRT
            )
        if USE_QQ_BIAS:
            S += load_qq_bias_tile(
                qq_bias_row_ptrs, seq_offset, context_len, qq_bias_stride_0
            )

        M, L, P, alpha = softmax_step(S, M, L)
        acc = acc * alpha[:, None]
        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            dist = context_len + qpos_lo - seq_offset[:, None]
            V = tl.where(dist < SLIDING_WINDOW, V, 0.0)
        acc += tl.dot(P.to(V.dtype), V)

    acc = acc / L[:, None]
    if USE_FP8:
        acc *= tl.load(out_scale)
    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )
    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


def _seq_to_request_row(
    provided: torch.Tensor | None,
    *,
    num_seqs: int,
    device: torch.device,
) -> torch.Tensor:
    if provided is not None:
        return provided
    return torch.arange(num_seqs, device=device, dtype=torch.int32)


def tmh_reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    cache,
    slot_mapping: torch.Tensor,
    attn_metadata,
    seq_to_request_row: torch.Tensor | None,
) -> None:
    num_tokens, num_kv_heads, head_size = key.shape
    head_size_v = value.shape[2]
    head_size_padded = triton.next_power_of_2(max(head_size, head_size_v))
    block_size = cache.spec.block_size
    num_seqs = len(attn_metadata.seq_lens)
    seq_rows = _seq_to_request_row(
        seq_to_request_row,
        num_seqs=num_seqs,
        device=key.device,
    )
    packed_v = cache.warm_value.shape[-1] * 2 == head_size_v
    num_warps = (
        4
        if current_platform.is_rocm()
        else min(16, max(1, head_size_padded // 32))
    )
    _tmh_reshape_and_cache_kernel[(num_tokens, num_kv_heads)](
        key_ptr=key,
        value_ptr=value,
        raw_key_ptr=cache.raw_key,
        raw_value_ptr=cache.raw_value,
        warm_key_ptr=cache.warm_key,
        warm_value_ptr=cache.warm_value,
        warm_k_scale_ptr=cache.warm_k_scale,
        warm_v_scale_ptr=cache.warm_v_scale,
        request_role_ptr=cache.request_role_by_row_page,
        request_slot_ptr=cache.request_slot_by_row_page,
        seq_to_request_row_ptr=seq_rows,
        query_start_len_ptr=attn_metadata.query_start_loc,
        seq_lens_ptr=attn_metadata.seq_lens,
        slot_mapping_ptr=slot_mapping,
        stride_key_tok=key.stride(0),
        stride_key_head=key.stride(1),
        stride_val_tok=value.stride(0),
        stride_val_head=value.stride(1),
        stride_raw_k_slot=cache.raw_key.stride(0),
        stride_raw_k_tok=cache.raw_key.stride(1),
        stride_raw_k_head=cache.raw_key.stride(2),
        stride_raw_v_slot=cache.raw_value.stride(0),
        stride_raw_v_tok=cache.raw_value.stride(1),
        stride_raw_v_head=cache.raw_value.stride(2),
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
        request_stride=cache.request_role_by_row_page.stride(0),
        num_seqs=num_seqs,
        block_size=block_size,
        head_size=head_size,
        head_size_v=head_size_v,
        HEAD_SIZE_PADDED=head_size_padded,
        WARM_VALUE_PACKED=packed_v,
        num_warps=num_warps,
    )


def tmh_unified_attention(
    *,
    q: torch.Tensor,
    cache,
    out: torch.Tensor,
    attn_metadata,
    seq_to_request_row: torch.Tensor | None,
    softmax_scale: float,
    causal,
    window_size: tuple[int, int],
    softcap: float,
    alibi_slopes=None,
    use_alibi_sqrt: bool = False,
    output_scale=None,
    qq_bias=None,
    sinks=None,
    mm_prefix_range=None,
) -> None:
    if not bool(causal):
        raise ValueError("TMH physical attention currently requires causal attention")
    if window_size[1] not in (-1, 0):
        raise ValueError("TMH physical attention requires decoder-style windowing")
    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"
    use_mm_prefix = False
    max_mm_ranges = 0
    if mm_prefix_range is not None:
        if mm_prefix_range.ndim == 3:
            use_mm_prefix = True
            max_mm_ranges = mm_prefix_range.shape[1]
        else:
            raise ValueError(
                f"Unsupported mm_prefix_range shape: {mm_prefix_range.shape}"
            )

    num_seqs = len(attn_metadata.seq_lens)
    num_query_heads = q.shape[1]
    num_kv_heads = cache.raw_key.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]
    if cache.raw_value.shape[3] != head_size:
        raise ValueError(
            "TMH physical attention currently requires head_size_v == head_size"
        )
    block_size = cache.spec.block_size
    block_m = (
        16
        if num_queries_per_kv <= 16
        else triton.next_power_of_2(num_queries_per_kv)
    )
    block_q = block_m // num_queries_per_kv
    total_num_q_blocks = q.shape[0] // block_q + num_seqs
    sliding_window = 1 + window_size[0] if window_size[0] >= 0 else 0
    tile_size = _get_tmh_tile_size(
        head_size,
        sliding_window,
        q.element_size(),
        block_size,
    )
    head_size_padded = triton.next_power_of_2(head_size)
    seq_rows = _seq_to_request_row(
        seq_to_request_row,
        num_seqs=num_seqs,
        device=q.device,
    )
    packed_v = cache.warm_value.shape[-1] * 2 == head_size
    launch_kwargs: dict[str, Any] = {}
    if current_platform.is_rocm():
        launch_kwargs["num_warps"] = 4
    _tmh_unified_attention_kernel[(total_num_q_blocks, num_kv_heads)](
        output_ptr=out,
        query_ptr=q,
        raw_key_ptr=cache.raw_key,
        raw_value_ptr=cache.raw_value,
        warm_key_ptr=cache.warm_key,
        warm_value_ptr=cache.warm_value,
        warm_k_scale_ptr=cache.warm_k_scale,
        warm_v_scale_ptr=cache.warm_v_scale,
        request_role_ptr=cache.request_role_by_row_page,
        request_slot_ptr=cache.request_slot_by_row_page,
        seq_to_request_row_ptr=seq_rows,
        seq_lens_ptr=attn_metadata.seq_lens,
        query_start_len_ptr=attn_metadata.query_start_loc,
        sink_ptr=sinks,
        alibi_slopes_ptr=alibi_slopes,
        qq_bias_ptr=qq_bias,
        mm_prefix_range_ptr=mm_prefix_range,
        scale=softmax_scale,
        out_scale=1 / output_scale if output_scale is not None else 1.0,
        softcap=softcap,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        request_stride=cache.request_role_by_row_page.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        qq_bias_stride_0=qq_bias.stride(0) if qq_bias is not None else 0,
        stride_raw_k_slot=cache.raw_key.stride(0),
        stride_raw_k_tok=cache.raw_key.stride(1),
        stride_raw_k_head=cache.raw_key.stride(2),
        stride_raw_v_slot=cache.raw_value.stride(0),
        stride_raw_v_tok=cache.raw_value.stride(1),
        stride_raw_v_head=cache.raw_value.stride(2),
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
        num_seqs=num_seqs,
        USE_ALIBI_SLOPES=alibi_slopes is not None,
        USE_ALIBI_SQRT=use_alibi_sqrt,
        USE_QQ_BIAS=qq_bias is not None,
        USE_SOFTCAP=softcap > 0,
        USE_SINKS=sinks is not None,
        SLIDING_WINDOW=sliding_window,
        USE_MM_PREFIX=use_mm_prefix,
        MAX_MM_RANGES=max_mm_ranges,
        USE_FP8=output_scale is not None,
        WARM_VALUE_PACKED=packed_v,
        **launch_kwargs,
    )


def _get_tmh_tile_size(
    head_size: int,
    sliding_window: int,
    element_size: int,
    block_size: int,
) -> int:
    del element_size
    del head_size
    del sliding_window
    return block_size
