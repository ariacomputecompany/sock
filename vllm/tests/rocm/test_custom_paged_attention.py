# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import rocm


def test_rocm_custom_paged_attention_rejects_gfx1x_head_size_64(monkeypatch):
    monkeypatch.setattr(rocm, "_ON_GFX9", False)
    monkeypatch.setattr(rocm, "_ON_GFX1X", True)
    rocm.rocm_custom_paged_attention_rejection_reasons.cache_clear()
    rocm.use_rocm_custom_paged_attention.cache_clear()

    reasons = rocm.rocm_custom_paged_attention_rejection_reasons(
        qtype=torch.bfloat16,
        head_size=64,
        block_size=16,
        gqa_ratio=7,
        max_seq_len=32768,
        sliding_window=0,
        kv_cache_dtype="auto",
    )

    assert reasons == ("head_size=64 requires Triton on gfx1x",)
    assert not rocm.use_rocm_custom_paged_attention(
        qtype=torch.bfloat16,
        head_size=64,
        block_size=16,
        gqa_ratio=7,
        max_seq_len=32768,
        sliding_window=0,
        kv_cache_dtype="auto",
    )


def test_rocm_custom_paged_attention_accepts_gfx1x_qwen_15b_shape(monkeypatch):
    monkeypatch.setattr(rocm, "_ON_GFX9", False)
    monkeypatch.setattr(rocm, "_ON_GFX1X", True)
    rocm.rocm_custom_paged_attention_rejection_reasons.cache_clear()
    rocm.use_rocm_custom_paged_attention.cache_clear()

    reasons = rocm.rocm_custom_paged_attention_rejection_reasons(
        qtype=torch.bfloat16,
        head_size=128,
        block_size=16,
        gqa_ratio=6,
        max_seq_len=32768,
        sliding_window=0,
        kv_cache_dtype="auto",
    )

    assert reasons == ()
    assert rocm.use_rocm_custom_paged_attention(
        qtype=torch.bfloat16,
        head_size=128,
        block_size=16,
        gqa_ratio=6,
        max_seq_len=32768,
        sliding_window=0,
        kv_cache_dtype="auto",
    )
