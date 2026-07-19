# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton attention backend entry point for physical TMH KV layout."""

from __future__ import annotations

import torch

from vllm.forward_context import get_forward_context
from vllm.v1.attention.ops.tmh_triton_ops import (
    tmh_reshape_and_cache,
    tmh_unified_attention,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
)
from vllm.v1.attention.backend import AttentionLayer, AttentionMetadata
from vllm.v1.tmh_physical import TMHPhysicalKVCache


class TMHTritonAttentionBackend(TritonAttentionBackend):
    """Dedicated backend name for physical TMH layout."""

    @staticmethod
    def get_name() -> str:
        return "TMH_TRITON_ATTN"

    @staticmethod
    def get_impl_cls() -> type["TMHTritonAttentionImpl"]:
        return TMHTritonAttentionImpl


class TMHTritonAttentionImpl(TritonAttentionImpl):
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: TMHPhysicalKVCache,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return output.fill_(0)
        if not isinstance(kv_cache, TMHPhysicalKVCache):
            if isinstance(kv_cache, torch.Tensor) and kv_cache.numel() == 0:
                return output.fill_(0)
            raise RuntimeError(
                "TMH_TRITON_ATTN requires TMHPhysicalKVCache; refusing to "
                "consume a standard KV tensor as physical TMH."
            )
        forward_context = get_forward_context()
        seq_to_request_row = forward_context.additional_kwargs.get(
            "tmh_seq_to_request_row"
        )
        mm_prefix_range_tensor = getattr(
            attn_metadata,
            "mm_prefix_range_tensor",
            None,
        )
        tmh_unified_attention(
            q=query[: attn_metadata.num_actual_tokens],
            cache=kv_cache,
            out=output[: attn_metadata.num_actual_tokens],
            attn_metadata=attn_metadata,
            seq_to_request_row=seq_to_request_row,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            window_size=self.sliding_window,
            softcap=self.logits_soft_cap,
            alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
            output_scale=output_scale,
            sinks=self.sinks,
            mm_prefix_range=mm_prefix_range_tensor,
        )
        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: TMHPhysicalKVCache,
        slot_mapping: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ):
        if attn_metadata is None:
            return
        if not isinstance(kv_cache, TMHPhysicalKVCache):
            if isinstance(kv_cache, torch.Tensor) and kv_cache.numel() == 0:
                return
            raise RuntimeError(
                "TMH_TRITON_ATTN requires TMHPhysicalKVCache; refusing to "
                "write standard KV as physical TMH."
            )
        del layer
        forward_context = get_forward_context()
        seq_to_request_row = forward_context.additional_kwargs.get(
            "tmh_seq_to_request_row"
        )
        tmh_reshape_and_cache(
            key,
            value,
            kv_cache,
            slot_mapping,
            attn_metadata,
            seq_to_request_row,
        )
