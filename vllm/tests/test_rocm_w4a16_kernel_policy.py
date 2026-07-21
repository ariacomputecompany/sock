# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.kernels.linear import choose_mp_linear_kernel
from vllm.model_executor.kernels.linear.mixed_precision import MPLinearLayerConfig
from vllm.model_executor.kernels.linear.mixed_precision.conch import ConchLinearKernel
from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
    TritonW4A16LinearKernel,
)
from vllm.platforms import PlatformEnum
from vllm.scalar_type import scalar_types


def _gptq_w4a16_config() -> MPLinearLayerConfig:
    return MPLinearLayerConfig(
        full_weight_shape=(5120, 5120),
        partition_weight_shape=(5120, 5120),
        weight_type=scalar_types.uint4b8,
        act_type=torch.bfloat16,
        group_size=128,
        zero_points=False,
        has_g_idx=False,
        zero_point_offset=1,
    )


def test_rocm_non_validated_rdna_non_cdna_falls_back_to_conch(monkeypatch):
    import vllm.model_executor.kernels.linear as linear_kernels
    import vllm.model_executor.kernels.linear.mixed_precision.rdna3_w4a16 as rdna_w4a16
    import vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 as triton_w4a16

    monkeypatch.setattr(linear_kernels.current_platform, "_enum", PlatformEnum.ROCM)
    monkeypatch.setattr(linear_kernels.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(linear_kernels.current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(linear_kernels.current_platform, "get_device_capability", lambda: None)
    monkeypatch.setattr(rdna_w4a16.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(triton_w4a16.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(triton_w4a16.current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1100", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_mi3xx", lambda: False)

    assert choose_mp_linear_kernel(_gptq_w4a16_config()) is ConchLinearKernel


def test_triton_w4a16_rejects_non_cdna_rocm(monkeypatch):
    import vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 as triton_w4a16

    monkeypatch.setattr(triton_w4a16.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(triton_w4a16.current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_mi3xx", lambda: False)

    can_implement, reason = TritonW4A16LinearKernel.can_implement(
        _gptq_w4a16_config()
    )

    assert not can_implement
    assert reason is not None
    assert "MI3xx" in reason


def test_triton_w4a16_allows_cdna_rocm(monkeypatch):
    import vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 as triton_w4a16

    monkeypatch.setattr(triton_w4a16.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(triton_w4a16.current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_mi3xx", lambda: True)

    can_implement, reason = TritonW4A16LinearKernel.can_implement(
        _gptq_w4a16_config()
    )

    assert can_implement
    assert reason is None
