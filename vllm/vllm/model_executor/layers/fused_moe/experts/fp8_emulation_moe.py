# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 MoE emulation on top of the standard Triton experts.

This backend is used when a checkpoint stores MoE weights in ordinary FP8
compressed-tensors format but the current accelerator does not expose a native
FP8 MoE kernel for that layout. It keeps the checkpoint format intact and
dequantizes expert weights to the request compute dtype before dispatching to
the regular Triton MoE path.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
)

logger = init_logger(__name__)


class Fp8QuantizationEmulationTritonExperts(TritonExperts):
    """Dequantize static-channel FP8 MoE weights and run BF16/FP16 Triton MoE."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        w1_scale = quant_config.w1_scale
        w2_scale = quant_config.w2_scale
        delegate_quant_config = FusedMoEQuantConfig.make(
            quant_dtype=None,
            weight_dtype=None,
            gemm1_alpha=quant_config.gemm1_alpha,
            gemm1_beta=quant_config.gemm1_beta,
            gemm1_clamp_limit=quant_config.gemm1_clamp_limit,
        )

        super().__init__(moe_config, delegate_quant_config)
        logger.warning_once(
            "Using Fp8QuantizationEmulationTritonExperts MoE backend. "
            "Weights are dequantized to the compute dtype before Triton MoE "
            "execution because this device does not expose a native FP8 MoE "
            "kernel for the checkpoint quantization layout."
        )

        self.w1_scale_val = w1_scale
        self.w2_scale_val = w2_scale
        self.quantization_emulation = True

    @property
    def quant_dtype(self) -> torch.dtype | str | None:
        return self.quant_config.quant_dtype

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (
            kFp8StaticChannelSym,
            kFp8DynamicTokenSym,
        )

    def _dequantize_weights(
        self,
        weights: torch.Tensor,
        scales: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if weights.element_size() >= 2:
            return weights.to(dtype)

        if weights.ndim != 3:
            raise ValueError(
                "FP8 MoE emulation expects expert weights in "
                f"[expert, out, in] layout; received shape {tuple(weights.shape)}."
            )

        if scales.ndim == 1:
            scales = scales.view(scales.shape[0], 1, 1)
        elif scales.ndim == 2:
            scales = scales.unsqueeze(-1)
        elif scales.ndim != 3:
            raise ValueError(
                "FP8 MoE emulation expects scales to be per-expert, "
                f"per-output-channel, or broadcastable 3D; received "
                f"shape {tuple(scales.shape)}."
            )

        try:
            torch.broadcast_shapes(weights.shape, scales.shape)
        except RuntimeError as exc:
            raise ValueError(
                "FP8 MoE emulation received weight scales that are not "
                "broadcastable to expert weights: "
                f"weights={tuple(weights.shape)} scales={tuple(scales.shape)}."
            ) from exc

        return (weights.to(torch.float32) * scales.to(torch.float32)).to(dtype)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert self.w1_scale_val is not None
        assert self.w2_scale_val is not None
        w1_dequant = self._dequantize_weights(
            w1, self.w1_scale_val, hidden_states.dtype
        )
        w2_dequant = self._dequantize_weights(
            w2, self.w2_scale_val, hidden_states.dtype
        )

        super().apply(
            output=output,
            hidden_states=hidden_states,
            w1=w1_dequant,
            w2=w2_dequant,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            a1q_scale=None,
            a2_scale=None,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=expert_tokens_meta,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
