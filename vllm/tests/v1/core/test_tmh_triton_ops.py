# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.tmh_triton_ops import (
    _get_tmh_mixed_tile_size,
    _get_tmh_raw_tile_size,
    tmh_reshape_and_cache,
    tmh_physical_attention,
)
from vllm.v1.kv_cache_interface import TMHFullAttentionSpec
from vllm.v1.tmh_physical import reshape_tmh_physical_kv_cache


def test_tmh_raw_tile_size_stays_physical_page_sized():
    assert _get_tmh_mixed_tile_size(block_size=16) == 16
    assert (
        _get_tmh_raw_tile_size(
            head_size=32,
            sliding_window=0,
            element_size=2,
            block_size=16,
            is_prefill=True,
        )
        == 16
    )
    assert (
        _get_tmh_raw_tile_size(
            head_size=32,
            sliding_window=0,
            element_size=2,
            block_size=16,
            is_prefill=False,
        )
        == 16
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_tmh_triton_attention_reads_raw_and_warm_int4_pages():
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16
    spec = TMHFullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=32,
        head_size_v=32,
        dtype=dtype,
        tmh_hot_budget_pct=25.0,
        tmh_late_layer=False,
        tmh_max_num_seqs=4,
        tmh_max_model_pages=3,
    )
    num_logical_blocks = 8
    backing = torch.empty(
        spec.physical_allocation_bytes(num_logical_blocks),
        dtype=torch.uint8,
        device=device,
    )
    cache = reshape_tmh_physical_kv_cache(
        backing,
        spec,
        num_logical_blocks=num_logical_blocks,
    )
    cache.request_slot_by_row_page[0, 0] = 0
    cache.request_slot_by_row_page[0, 1] = 0
    cache.request_slot_by_row_page[0, 2] = 1
    cache.native_block_table_by_seq[0, 0] = 0
    cache.native_block_table_by_seq[0, 1] = 0
    cache.native_block_table_by_seq[0, 2] = 1

    tokens = 48
    heads = 1
    q = torch.randn(tokens, heads, 32, device=device, dtype=dtype)
    k = torch.randn(tokens, heads, 32, device=device, dtype=dtype)
    v = torch.randn(tokens, heads, 32, device=device, dtype=dtype)
    out = torch.empty_like(q)
    slot_mapping = torch.arange(tokens, device=device, dtype=torch.int64)
    query_start = torch.tensor([0, tokens], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([tokens], device=device, dtype=torch.int32)
    seq_rows = torch.tensor([0], device=device, dtype=torch.int32)
    meta = SimpleNamespace(
        num_actual_tokens=tokens,
        query_start_loc=query_start,
        seq_lens=seq_lens,
        max_query_len=tokens,
        max_seq_len=tokens,
        causal=True,
    )

    tmh_reshape_and_cache(k, v, cache, slot_mapping, meta, seq_rows)
    tmh_physical_attention(
        q=q,
        key=k,
        value=v,
        cache=cache,
        out=out,
        attn_metadata=meta,
        seq_to_request_row=seq_rows,
        softmax_scale=1.0 / math.sqrt(32),
        causal=True,
        window_size=(-1, 0),
        softcap=0.0,
        kv_cache_dtype="auto",
        k_scale=None,
        v_scale=None,
    )

    ref_scores = (q[:, 0].float() @ k[:, 0].float().T) / math.sqrt(32)
    causal_mask = torch.triu(
        torch.ones(tokens, tokens, device=device),
        diagonal=1,
    ).bool()
    ref_scores = ref_scores.masked_fill(causal_mask, float("-inf"))
    ref = torch.softmax(ref_scores, dim=-1) @ v[:, 0].float()

    torch.testing.assert_close(out[:16, 0].float(), ref[:16], atol=5e-2, rtol=5e-2)
    warm_error = float((out[16:, 0].float() - ref[16:]).abs().max())
    assert warm_error < 0.35


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_tmh_triton_attention_uses_raw_fast_path_for_all_raw_batches():
    torch.manual_seed(1)
    device = torch.device("cuda")
    dtype = torch.float16
    spec = TMHFullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=32,
        head_size_v=32,
        dtype=dtype,
        tmh_hot_budget_pct=25.0,
        tmh_late_layer=False,
        tmh_max_num_seqs=1,
        tmh_max_model_pages=3,
    )
    num_logical_blocks = 8
    backing = torch.empty(
        spec.physical_allocation_bytes(num_logical_blocks),
        dtype=torch.uint8,
        device=device,
    )
    cache = reshape_tmh_physical_kv_cache(
        backing,
        spec,
        num_logical_blocks=num_logical_blocks,
    )
    cache.request_slot_by_row_page[0, 0] = 0
    cache.request_slot_by_row_page[0, 1] = 1
    cache.request_slot_by_row_page[0, 2] = 2
    cache.native_block_table_by_seq[0, 0] = 0
    cache.native_block_table_by_seq[0, 1] = 1
    cache.native_block_table_by_seq[0, 2] = 2

    tokens = 48
    heads = 1
    q = torch.randn(tokens, heads, 32, device=device, dtype=dtype)
    k = torch.randn(tokens, heads, 32, device=device, dtype=dtype)
    v = torch.randn(tokens, heads, 32, device=device, dtype=dtype)
    out = torch.empty_like(q)
    slot_mapping = torch.arange(tokens, device=device, dtype=torch.int64)
    query_start = torch.tensor([0, tokens], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([tokens], device=device, dtype=torch.int32)
    seq_rows = torch.tensor([0], device=device, dtype=torch.int32)
    meta = SimpleNamespace(
        num_actual_tokens=tokens,
        query_start_loc=query_start,
        seq_lens=seq_lens,
        max_query_len=tokens,
        max_seq_len=tokens,
        causal=True,
    )

    cache.warm_key.zero_()
    cache.warm_value.zero_()
    cache.warm_k_scale.zero_()
    cache.warm_v_scale.zero_()
    tmh_reshape_and_cache(k, v, cache, slot_mapping, meta, seq_rows)
    assert cache.warm_key.abs().sum().item() == 0
    assert cache.warm_value.abs().sum().item() == 0
    assert cache.warm_k_scale.abs().sum().item() == 0
    assert cache.warm_v_scale.abs().sum().item() == 0
    tmh_physical_attention(
        q=q,
        key=k,
        value=v,
        cache=cache,
        out=out,
        attn_metadata=meta,
        seq_to_request_row=seq_rows,
        softmax_scale=1.0 / math.sqrt(32),
        causal=True,
        window_size=(-1, 0),
        softcap=0.0,
        kv_cache_dtype="auto",
        k_scale=None,
        v_scale=None,
    )

    ref_scores = (q[:, 0].float() @ k[:, 0].float().T) / math.sqrt(32)
    causal_mask = torch.triu(
        torch.ones(tokens, tokens, device=device),
        diagonal=1,
    ).bool()
    ref_scores = ref_scores.masked_fill(causal_mask, float("-inf"))
    ref = torch.softmax(ref_scores, dim=-1) @ v[:, 0].float()

    torch.testing.assert_close(out[:, 0].float(), ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform.is_rocm(),
    reason="requires ROCm GPU",
)
def test_tmh_native_raw_decode_reads_physical_raw_pages():
    torch.manual_seed(2)
    device = torch.device("cuda")
    dtype = torch.float16
    spec = TMHFullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=32,
        head_size_v=32,
        dtype=dtype,
        tmh_hot_budget_pct=25.0,
        tmh_late_layer=False,
        tmh_max_num_seqs=1,
        tmh_max_model_pages=4,
    )
    num_logical_blocks = 16
    backing = torch.empty(
        spec.physical_allocation_bytes(num_logical_blocks),
        dtype=torch.uint8,
        device=device,
    )
    cache = reshape_tmh_physical_kv_cache(
        backing,
        spec,
        num_logical_blocks=num_logical_blocks,
    )
    for page in range(4):
        cache.request_slot_by_row_page[0, page] = page
        cache.native_block_table_by_seq[0, page] = page

    tokens = 49
    heads = 1
    q = torch.randn(tokens, heads, 32, device=device, dtype=dtype)
    k = torch.randn(tokens, heads, 32, device=device, dtype=dtype)
    v = torch.randn(tokens, heads, 32, device=device, dtype=dtype)
    slot_mapping = torch.arange(tokens, device=device, dtype=torch.int64)
    seq_rows = torch.tensor([0], device=device, dtype=torch.int32)
    prefill_meta = SimpleNamespace(
        num_actual_tokens=tokens,
        query_start_loc=torch.tensor([0, tokens], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([tokens], device=device, dtype=torch.int32),
        max_query_len=tokens,
        max_seq_len=tokens,
        causal=True,
    )
    tmh_reshape_and_cache(k, v, cache, slot_mapping, prefill_meta, seq_rows)

    decode_q = q[-1:].contiguous()
    out_with_scale = torch.empty_like(decode_q)
    out_identity_scale = torch.empty_like(decode_q)
    decode_meta = SimpleNamespace(
        num_actual_tokens=1,
        query_start_loc=torch.tensor([0, 1], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([tokens], device=device, dtype=torch.int32),
        max_query_len=1,
        max_seq_len=tokens,
        causal=True,
    )
    scale = torch.tensor(1.0, device=device, dtype=torch.float32)
    tmh_physical_attention(
        q=decode_q,
        cache=cache,
        out=out_with_scale,
        attn_metadata=decode_meta,
        seq_to_request_row=seq_rows,
        softmax_scale=1.0 / math.sqrt(32),
        causal=True,
        window_size=(-1, 0),
        softcap=0.0,
        kv_cache_dtype="auto",
        k_scale=scale,
        v_scale=scale,
    )
    tmh_physical_attention(
        q=decode_q,
        cache=cache,
        out=out_identity_scale,
        attn_metadata=decode_meta,
        seq_to_request_row=seq_rows,
        softmax_scale=1.0 / math.sqrt(32),
        causal=True,
        window_size=(-1, 0),
        softcap=0.0,
        kv_cache_dtype="auto",
        k_scale=None,
        v_scale=None,
    )

    ref_scores = (decode_q[:, 0].float() @ k[:, 0].float().T) / math.sqrt(32)
    ref = torch.softmax(ref_scores, dim=-1) @ v[:, 0].float()
    torch.testing.assert_close(out_with_scale[:, 0].float(), ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(
        out_identity_scale[:, 0].float(),
        out_with_scale[:, 0].float(),
        atol=5e-2,
        rtol=5e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_tmh_triton_segmented_decode_reads_raw_and_warm_int4_pages():
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16
    spec = TMHFullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=32,
        head_size_v=32,
        dtype=dtype,
        tmh_hot_budget_pct=25.0,
        tmh_late_layer=False,
        tmh_max_num_seqs=4,
        tmh_max_model_pages=3,
    )
    num_logical_blocks = 8
    backing = torch.empty(
        spec.physical_allocation_bytes(num_logical_blocks),
        dtype=torch.uint8,
        device=device,
    )
    cache = reshape_tmh_physical_kv_cache(
        backing,
        spec,
        num_logical_blocks=num_logical_blocks,
    )
    cache.request_slot_by_row_page[0, 0] = 0
    cache.request_slot_by_row_page[0, 1] = 0
    cache.request_slot_by_row_page[0, 2] = 1
    cache.native_block_table_by_seq[0, 0] = 0
    cache.native_block_table_by_seq[0, 1] = 0
    cache.native_block_table_by_seq[0, 2] = 1

    tokens = 48
    heads = 1
    q = torch.randn(1, heads, 32, device=device, dtype=dtype)
    k = torch.randn(tokens, heads, 32, device=device, dtype=dtype)
    v = torch.randn(tokens, heads, 32, device=device, dtype=dtype)
    out = torch.empty_like(q)
    slot_mapping = torch.arange(tokens, device=device, dtype=torch.int64)
    seq_rows = torch.tensor([0], device=device, dtype=torch.int32)
    prefill_meta = SimpleNamespace(
        num_actual_tokens=tokens,
        query_start_loc=torch.tensor([0, tokens], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([tokens], device=device, dtype=torch.int32),
        max_query_len=tokens,
        max_seq_len=tokens,
        causal=True,
    )
    decode_meta = SimpleNamespace(
        num_actual_tokens=1,
        query_start_loc=torch.tensor([0, 1], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([tokens], device=device, dtype=torch.int32),
        max_query_len=1,
        max_seq_len=4096,
        causal=True,
        seq_threshold_3D=16,
        num_par_softmax_segments=4,
        softmax_segm_output=torch.empty(
            (16, heads, 4, 32), device=device, dtype=torch.float32
        ),
        softmax_segm_max=torch.empty((16, heads, 4), device=device, dtype=torch.float32),
        softmax_segm_expsum=torch.empty(
            (16, heads, 4), device=device, dtype=torch.float32
        ),
    )

    tmh_reshape_and_cache(k, v, cache, slot_mapping, prefill_meta, seq_rows)
    tmh_physical_attention(
        q=q,
        cache=cache,
        out=out,
        attn_metadata=decode_meta,
        seq_to_request_row=seq_rows,
        softmax_scale=1.0 / math.sqrt(32),
        causal=True,
        window_size=(-1, 0),
        softcap=0.0,
    )

    ref_scores = (q[:, 0].float() @ k[:, 0].float().T) / math.sqrt(32)
    ref = torch.softmax(ref_scores, dim=-1) @ v[:, 0].float()
    torch.testing.assert_close(out[:, 0].float(), ref, atol=0.35, rtol=0.35)
