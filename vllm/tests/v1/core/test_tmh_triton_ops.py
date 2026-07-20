# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.ops.tmh_triton_ops import (
    tmh_reshape_and_cache,
    tmh_unified_attention,
)
from vllm.v1.kv_cache_interface import TMHFullAttentionSpec
from vllm.v1.tmh_physical import reshape_tmh_physical_kv_cache


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
        tmh_max_num_seqs=1,
        tmh_max_model_pages=2,
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
    cache.request_role_by_row_page[0, 0] = 0
    cache.request_slot_by_row_page[0, 0] = 0
    cache.request_role_by_row_page[0, 1] = 2
    cache.request_slot_by_row_page[0, 1] = 0

    tokens = 20
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
    tmh_unified_attention(
        q=q,
        cache=cache,
        out=out,
        attn_metadata=meta,
        seq_to_request_row=seq_rows,
        softmax_scale=1.0 / math.sqrt(32),
        causal=True,
        window_size=(-1, 0),
        softcap=0.0,
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
        tmh_max_num_seqs=1,
        tmh_max_model_pages=2,
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
    cache.request_role_by_row_page[0, 0] = 0
    cache.request_slot_by_row_page[0, 0] = 0
    cache.request_role_by_row_page[0, 1] = 2
    cache.request_slot_by_row_page[0, 1] = 0

    tokens = 20
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
        max_seq_len=tokens,
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
    tmh_unified_attention(
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
