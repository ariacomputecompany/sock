#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ROCm Triton W4A16 GEMM kernel.

Run `pytest tests/kernels/quantization/test_triton_w4a16.py`.
"""

import importlib

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

# This test module is ROCm/Triton specific. Avoid import-time failures on
# non-ROCm or environments without Triton by skipping early.
if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

pytest.importorskip("triton")

device = "cuda"

triton_w4a16_module = importlib.import_module(
    "vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16"
)
triton_w4a16_gemm = triton_w4a16_module.triton_w4a16_gemm
TritonW4A16LinearKernel = triton_w4a16_module.TritonW4A16LinearKernel


def _pack_int_along_n(w_kn: torch.Tensor, bit_width: int) -> torch.Tensor:
    """Pack low-bit values along N: [K, N] -> [K, N//pack] int32."""
    assert w_kn.dtype == torch.int32
    K, N = w_kn.shape
    pack_factor = 32 // bit_width
    assert N % pack_factor == 0
    mask = (1 << bit_width) - 1
    shifts = (
        torch.arange(pack_factor, device=w_kn.device, dtype=torch.int32) * bit_width
    )
    return torch.sum(
        (w_kn.view(K, N // pack_factor, pack_factor) & mask) << shifts,
        dim=2,
        dtype=torch.int32,
    ).contiguous()


def _unpack_int_along_n(w_packed: torch.Tensor, bit_width: int) -> torch.Tensor:
    """Unpack low-bit values along N: [K, N//pack] -> [K, N] int32."""
    assert w_packed.dtype == torch.int32
    K, N_packed = w_packed.shape
    pack_factor = 32 // bit_width
    mask = (1 << bit_width) - 1
    shifts = (
        torch.arange(pack_factor, device=w_packed.device, dtype=torch.int32)
        * bit_width
    )
    values = (w_packed.unsqueeze(-1) >> shifts) & mask
    return values.reshape(K, N_packed * pack_factor)


def _pack_int_along_k_to_ckpt(w_kn: torch.Tensor, bit_width: int) -> torch.Tensor:
    """Pack low-bit values along K into CT layout: [K,N] -> [N, K//pack]."""
    assert w_kn.dtype == torch.int32
    K, N = w_kn.shape
    pack_factor = 32 // bit_width
    mask = (1 << bit_width) - 1
    assert K % pack_factor == 0
    out = torch.zeros((N, K // pack_factor), dtype=torch.int32, device=w_kn.device)
    for i in range(pack_factor):
        out |= (w_kn[i::pack_factor, :].t() & mask) << (i * bit_width)
    return out.contiguous()


def _w4a16_reference(
    a_mk: torch.Tensor,
    b_packed: torch.Tensor,
    scales_gn: torch.Tensor,
    *,
    group_size: int,
    bit_width: int,
    qzeros_gn: torch.Tensor | None,
    zp_bias: int,
    zero_point_offset: int = 0,
) -> torch.Tensor:
    """Reference implementation for W4A16.

    a_mk: [M,K] fp16/bf16
    b_packed: [K, N//pack] int32, N-packed weights
    scales_gn: [K//G, N] fp16/bf16
    qzeros_gn: [K//G, N//pack] int32, N-packed zeros, or None
    """
    assert a_mk.dtype in (torch.float16, torch.bfloat16)
    assert b_packed.dtype == torch.int32
    assert scales_gn.dtype == a_mk.dtype

    M, K = a_mk.shape
    pack_factor = 32 // bit_width
    N = b_packed.shape[1] * pack_factor
    assert b_packed.shape[0] == K

    assert group_size > 0 and K % group_size == 0
    G = group_size
    num_groups = K // G
    assert scales_gn.shape == (num_groups, N)

    w_int = _unpack_int_along_n(b_packed, bit_width)  # [K,N]
    if qzeros_gn is None:
        z_full = torch.full((K, N), zp_bias, dtype=torch.int32, device=a_mk.device)
    else:
        assert qzeros_gn.shape == (num_groups, N // pack_factor)
        z_gn = _unpack_int_along_n(qzeros_gn, bit_width)  # [G,N] in groups
        z_gn = z_gn + zero_point_offset
        z_full = z_gn.repeat_interleave(G, dim=0)  # [K,N]

    s_full = scales_gn.repeat_interleave(G, dim=0).to(torch.float32)  # [K,N]
    w_fp = (w_int - z_full).to(torch.float32) * s_full  # [K,N]

    out = a_mk.to(torch.float32) @ w_fp  # [M,N]
    return out.to(a_mk.dtype)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
@pytest.mark.parametrize("bit_width,zp_bias", [(2, 2), (4, 8)])
@pytest.mark.parametrize("zero_point_offset", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "M,K,N,G,has_zp",
    [
        (1, 256, 256, 32, False),
        (17, 256, 512, 32, False),
        (32, 512, 256, 64, False),
        (33, 512, 512, 128, False),
        (64, 1024, 256, 256, False),
        (128, 256, 1024, 32, True),
        (64, 512, 512, 64, True),
    ],
)
def test_triton_w4a16_gemm_matches_reference(
    bit_width, zp_bias, zero_point_offset, dtype, M, K, N, G, has_zp
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")
    pack_factor = 32 // bit_width
    if N % pack_factor != 0 or K % G != 0:
        pytest.skip("Invalid test shape")
    if not has_zp and zero_point_offset != 0:
        pytest.skip("zero_point_offset only applies to explicit qzeros")

    set_random_seed(0)

    a = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    max_value = 1 << bit_width
    w_int = torch.randint(0, max_value, (K, N), device=device, dtype=torch.int32)
    b_packed = _pack_int_along_n(w_int, bit_width)

    scales = (0.05 * torch.rand((K // G, N), device=device, dtype=torch.float32)).to(
        dtype
    )

    qzeros = None
    if has_zp:
        zeros_int = torch.randint(
            0,
            max_value,
            (K // G, N),
            device=device,
            dtype=torch.int32,
        )
        qzeros = _pack_int_along_n(zeros_int, bit_width)

    out = triton_w4a16_gemm(
        a=a,
        b_q=b_packed,
        scales=scales,
        qzeros=qzeros,
        group_size=G,
        bit_width=bit_width,
        zp_bias=zp_bias,
        zero_point_offset=zero_point_offset,
    )
    ref = _w4a16_reference(
        a,
        b_packed,
        scales,
        group_size=G,
        bit_width=bit_width,
        qzeros_gn=qzeros,
        zp_bias=zp_bias,
        zero_point_offset=zero_point_offset,
    )

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
def test_triton_w4a16_gemm_requires_contiguous_inputs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")

    set_random_seed(0)
    M, K, N, G = 32, 256, 256, 32
    a = torch.randn((K, M), device=device, dtype=torch.float16).t()  # non-contiguous
    w_int = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    b_packed = _pack_int_along_n(w_int, 4)
    scales = torch.rand((K // G, N), device=device, dtype=torch.float16)

    with pytest.raises(AssertionError):
        triton_w4a16_gemm(
            a=a,
            b_q=b_packed,
            scales=scales,
            qzeros=None,
            group_size=G,
            bit_width=4,
            zp_bias=8,
        )


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
@pytest.mark.parametrize(
    "bit_width,weight_type",
    [
        (2, "uint2b2"),
        (4, "uint4"),
    ],
)
@pytest.mark.parametrize("loader_layout", ["checkpoint", "auto_gptq"])
def test_triton_w4a16_process_weights_after_loading_repacks_layout(
    bit_width, weight_type, loader_layout
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.model_executor.parameter import (
        GroupQuantScaleParameter,
        PackedColumnParameter,
        PackedvLLMParameter,
    )
    from vllm.scalar_type import scalar_types

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:0",
            local_rank=0,
        )
        ensure_model_parallel_initialized(1, 1)

    set_random_seed(0)

    # Small-but-nontrivial shapes.
    K, N = 256, 256
    G = 32
    pack_factor = 32 // bit_width
    assert K % pack_factor == 0 and N % pack_factor == 0 and K % G == 0

    # Build a canonical low-bit weight grid then pack into loader layouts.
    max_value = 1 << bit_width
    w_int_kn = torch.randint(0, max_value, (K, N), device=device, dtype=torch.int32)
    w_ckpt_nkp = _pack_int_along_k_to_ckpt(w_int_kn, bit_width)  # [N, K//pack]
    w_auto_kpn = w_ckpt_nkp.t().contiguous()  # [K//pack, N]

    # Scales in CT checkpoint layout for WNA16: [N, K//G]
    scales_ckpt_nkg = 0.05 * torch.rand((N, K // G), device=device, dtype=torch.float16)
    scales_auto_gn = scales_ckpt_nkg.t().contiguous()  # [K//G, N]

    # Asymmetric case: zero points in either supported loader layout.
    zeros_int_gn = torch.randint(
        0,
        max_value,
        (K // G, N),
        device=device,
        dtype=torch.int32,
    )
    zeros_packed_gnp = _pack_int_along_n(zeros_int_gn, bit_width)  # [K//G, N//pack]
    zeros_ckpt_npkg = zeros_packed_gnp.t().contiguous()  # [N//pack, K//G]

    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=getattr(scalar_types, weight_type),
        act_type=torch.float16,
        group_size=G,
        zero_points=True,
        has_g_idx=False,
    )
    kernel = TritonW4A16LinearKernel(
        config,
        w_q_param_name="weight_packed",
        w_s_param_name="weight_scale",
        w_zp_param_name="weight_zero_point",
        w_gidx_param_name=None,
    )

    # Build dummy layer with vLLM parameter wrappers.
    weight_loader = lambda *args, **kwargs: None

    class DummyLayer(torch.nn.Module):
        pass

    layer = DummyLayer()
    if loader_layout == "checkpoint":
        layer.register_parameter(
            "weight_packed",
            PackedvLLMParameter(
                data=w_ckpt_nkp,
                weight_loader=weight_loader,
                input_dim=1,
                output_dim=0,
                packed_factor=pack_factor,
                packed_dim=1,
            ),
        )
        layer.register_parameter(
            "weight_scale",
            GroupQuantScaleParameter(
                data=scales_ckpt_nkg,
                weight_loader=weight_loader,
                input_dim=1,
                output_dim=0,
            ),
        )
        layer.register_parameter(
            "weight_zero_point",
            PackedColumnParameter(
                data=zeros_ckpt_npkg,
                weight_loader=weight_loader,
                output_dim=0,
                packed_factor=pack_factor,
                packed_dim=0,
            ),
        )
    else:
        layer.register_parameter(
            "weight_packed",
            PackedvLLMParameter(
                data=w_auto_kpn,
                weight_loader=weight_loader,
                input_dim=0,
                output_dim=1,
                packed_factor=pack_factor,
                packed_dim=0,
            ),
        )
        layer.register_parameter(
            "weight_scale",
            GroupQuantScaleParameter(
                data=scales_auto_gn,
                weight_loader=weight_loader,
                input_dim=0,
                output_dim=1,
            ),
        )
        layer.register_parameter(
            "weight_zero_point",
            PackedvLLMParameter(
                data=zeros_packed_gnp,
                weight_loader=weight_loader,
                input_dim=0,
                output_dim=1,
                packed_factor=pack_factor,
                packed_dim=1,
            ),
        )

    kernel.process_weights_after_loading(layer)

    # Expected transformed layouts.
    expected_w_knp = _pack_int_along_n(w_int_kn, bit_width)  # [K, N//pack]
    expected_scales_gn = scales_ckpt_nkg.t().contiguous()  # [K//G, N]
    expected_zeros_gnp = zeros_packed_gnp

    assert tuple(layer.weight_packed.shape) == (K, N // pack_factor)
    assert tuple(layer.weight_scale.shape) == (K // G, N)
    assert tuple(layer.weight_zero_point.shape) == (K // G, N // pack_factor)

    torch.testing.assert_close(layer.weight_packed, expected_w_knp)
    torch.testing.assert_close(layer.weight_scale, expected_scales_gn)
    torch.testing.assert_close(layer.weight_zero_point, expected_zeros_gnp)
