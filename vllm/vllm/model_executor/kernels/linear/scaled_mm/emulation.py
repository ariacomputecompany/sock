# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 scaled-mm emulation for devices without native FP8 GEMM support."""

import math

import torch

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)


def _num_tokens(output_shape: list) -> int:
    return math.prod(output_shape[:-1])


class EmulationFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    """Dequantize FP8 activations/weights and run standard matmul."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if c.activation_quant_key.scale.group_shape.is_per_group():
            return False, "block-scaled activations should use block kernels."
        if c.weight_quant_key.scale.group_shape.is_per_group():
            return False, "block-scaled weights should use block kernels."
        return True, None

    @staticmethod
    def _broadcast_weight_scale(B: torch.Tensor, Bs: torch.Tensor) -> torch.Tensor:
        if Bs.dim() == 0 or Bs.numel() == 1:
            return Bs.reshape(1, 1)
        if Bs.dim() == 1:
            return Bs.reshape(1, -1)
        if Bs.shape == (B.shape[1], 1):
            return Bs.t()
        return Bs

    @staticmethod
    def _broadcast_activation_scale(A: torch.Tensor, As: torch.Tensor) -> torch.Tensor:
        if As.dim() == 0 or As.numel() == 1:
            return As.reshape(1, 1)
        if As.dim() == 1:
            return As.reshape(-1, 1)
        return As

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        compute_dtype = torch.float32 if out_dtype == torch.float32 else torch.bfloat16
        A_dq = A.to(compute_dtype) * self._broadcast_activation_scale(A, As).to(
            compute_dtype
        )
        B_dq = B.to(compute_dtype) * self._broadcast_weight_scale(B, Bs).to(
            compute_dtype
        )
        output = torch.matmul(A_dq, B_dq)
        if bias is not None:
            output = output + bias.to(output.dtype)

        num_tokens = _num_tokens(output_shape)
        return torch.narrow(output, 0, 0, num_tokens).to(out_dtype).view(*output_shape)
