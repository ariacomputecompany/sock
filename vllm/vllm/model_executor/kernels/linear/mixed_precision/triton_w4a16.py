# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton-based W4A16 GEMM kernel for ROCm MI300.

Implements fused low-bit weight dequantization + fp16/bf16 GEMM in a single kernel,
using GPTQ sequential packing (16 int2 or 8 int4 values per int32).
Plugs into the MPLinearKernel selection system and is preferred over
MarlinLinearKernel/ExllamaLinearKernel on ROCm.

Weight layout expected by this kernel (post-process_weights_after_loading):
  qweight: [K, N//pack]  int32  — rows=K (input), cols=N//pack (N is packed)
  scales:  [K//G, N]  fp16/bf16
  qzeros:  [K//G, N//pack]  int32  (optional; None for symmetric biased types)

Checkpoint layout from compressed_tensors_wNa16 create_weights:
  weight_packed:     [N, K//pack]  int32  (output_dim=0, input_dim=1, packed_dim=1)
  weight_scale:      [N, K//G]  fp16   (output_dim=0, input_dim=1)
  weight_zero_point: [N//pack, K//G]  int32 (output_dim=0, packed_dim=0)
"""

import torch

from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl, triton

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

TRITON_W4A16_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128, 256]
TRITON_W4A16_SUPPORTED_QUANT_TYPES = [
    scalar_types.uint2b2,  # symmetric GPTQ 2-bit (bias=2)
    scalar_types.uint4b8,  # symmetric GPTQ (bias=8)
    scalar_types.uint4,  # asymmetric with explicit zeros
]


@triton.jit
def triton_w4a16_gemm_kernel(
    # Pointers
    a_ptr,  # [M, K]  fp16/bf16 activations
    b_ptr,  # [K, N//pack]  int32 packed weights (N is the packed dim)
    scales_ptr,  # [K//G, N]  fp16/bf16 scales
    zeros_ptr,  # [K//G, N//pack]  int32 packed zeros (unused when HAS_ZP=False)
    c_ptr,  # [M, N]  fp16/bf16 output
    # Dimensions
    M,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,  # stride in b along the packed N dim
    stride_cm,
    stride_cn,
    # Quantization parameters
    group_size,
    BIT_WIDTH: tl.constexpr,
    PACK_FACTOR: tl.constexpr,
    BIT_MASK: tl.constexpr,
    ZERO_POINT_OFFSET: tl.constexpr,
    # Whether explicit zero points are provided
    HAS_ZP: tl.constexpr,
    # Zero bias used when HAS_ZP is False (e.g. 2 for uint2b2, 8 for uint4b8)
    ZP_BIAS: tl.constexpr,
    # Block sizes (tuned for MI300 wavefront=64)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused W4A16 GEMM: C[M,N] = A[M,K] @ dequant(B)[K,N]

    B is stored as [K, N//pack] int32 using GPTQ sequential packing:
      each int32 packs consecutive N-values at BIT_WIDTH-spaced offsets.

    Dequant: w_fp = (w_int - zero) * scale
      HAS_ZP=True:  zero is loaded from zeros_ptr and unpacked
      HAS_ZP=False: zero = ZP_BIAS constant for biased symmetric types
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row/col offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # b/zeros are stored with N packed: N//PACK_FACTOR int32 columns per K row
    offs_bn = pid_n * (BLOCK_N // PACK_FACTOR) + tl.arange(0, BLOCK_N // PACK_FACTOR)

    # GPTQ sequential shifts tiled across BLOCK_N:
    #   [0,bit_width,...,32-bit_width] repeating for every packed int32.
    # Build 1D shifts_1d of length BLOCK_N.
    shifts_row = tl.arange(0, PACK_FACTOR) * BIT_WIDTH
    shifts_1d_2d = tl.broadcast_to(
        shifts_row[None, :],
        (BLOCK_N // PACK_FACTOR, PACK_FACTOR),
    )
    shifts_1d = tl.reshape(shifts_1d_2d, (BLOCK_N,))  # [BLOCK_N]
    # Broadcast to [BLOCK_K, BLOCK_N] for weight unpacking
    shifts = tl.broadcast_to(shifts_1d[None, :], (BLOCK_K, BLOCK_N))

    # Scales column offsets: full N-width (one scale per output neuron)
    offs_sn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # ---- Load activations A: [BLOCK_M, BLOCK_K] ----
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        mask_a = (offs_m[:, None] < M) & mask_k[None, :]
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # ---- Load packed weights B: [BLOCK_K, BLOCK_N//pack] int32 ----
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        mask_b = mask_k[:, None] & (offs_bn[None, :] < N // PACK_FACTOR)
        b_packed = tl.load(b_ptrs, mask=mask_b, other=0)

        # ---- Unpack low-bit weights → [BLOCK_K, BLOCK_N] ----
        # tl.interleave(x, x) doubles the last dim by interleaving.
        # Starting from [BLOCK_K, BLOCK_N//pack], repeated interleaves give
        # [BLOCK_K, BLOCK_N], where each int32 is replicated pack times.
        b = tl.interleave(b_packed, b_packed)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        if PACK_FACTOR == 16:
            b = tl.interleave(b, b)
        # Extract the correct packed value for each output column
        b = (b >> shifts) & BIT_MASK

        # ---- Compute scale/zero group row index ----
        g_idx = (k_start * BLOCK_K) // group_size

        # ---- Load scales: [BLOCK_N] → broadcast to [BLOCK_K, BLOCK_N] ----
        scale_offset = g_idx * N + offs_sn
        scale_mask = offs_sn < N
        scales = tl.load(scales_ptr + scale_offset, mask=scale_mask, other=1.0)
        scales = tl.broadcast_to(scales[None, :], (BLOCK_K, BLOCK_N))

        # ---- Load / compute zeros ----
        if HAS_ZP:
            # Load packed zeros row: [BLOCK_N//pack] int32
            zero_offset = g_idx * (N // PACK_FACTOR) + offs_bn
            zero_mask = offs_bn < N // PACK_FACTOR
            z_packed = tl.load(zeros_ptr + zero_offset, mask=zero_mask, other=0)
            # Unpack to [BLOCK_N] using same interleave+shift pattern
            z = tl.interleave(z_packed, z_packed)
            z = tl.interleave(z, z)
            z = tl.interleave(z, z)
            if PACK_FACTOR == 16:
                z = tl.interleave(z, z)
            z = ((z >> shifts_1d) & BIT_MASK) + ZERO_POINT_OFFSET
            z = tl.broadcast_to(z[None, :], (BLOCK_K, BLOCK_N))
        else:
            z = tl.full((BLOCK_K, BLOCK_N), ZP_BIAS, dtype=tl.int32)

        # ---- Dequantize: (w - zero) * scale ----
        b_fp = (b - z).to(a.dtype) * scales

        # ---- Accumulate ----
        accumulator += tl.dot(a, b_fp, out_dtype=tl.float32)

    # ---- Store output C: [BLOCK_M, BLOCK_N] ----
    c = accumulator.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


def triton_w4a16_gemm(
    a: torch.Tensor,  # [M, K] fp16/bf16
    b_q: torch.Tensor,  # [K, N//pack] int32
    scales: torch.Tensor,  # [K//G, N] fp16/bf16
    qzeros: torch.Tensor | None,  # [K//G, N//pack] int32, or None
    group_size: int,
    bit_width: int = 4,
    zp_bias: int = 8,  # bias for biased symmetric types when qzeros is None
    zero_point_offset: int = 0,
) -> torch.Tensor:
    """
    Fused WNA16 GEMM using GPTQ-packed low-bit weights.

    Args:
        a:          Activation matrix [M, K], float16 or bfloat16.
        b_q:        Packed weight matrix [K, N//pack], int32 (GPTQ sequential).
        scales:     Per-group scales [K//G, N], same dtype as a.
        qzeros:     Per-group packed zero points [K//G, N//pack] int32, or None
                    for symmetric quantization (uses zp_bias instead).
        group_size: Quantization group size (resolved from -1 to K by caller).
        bit_width:  Number of bits per packed weight value.
        zp_bias:    Constant zero used when qzeros is None.
        zero_point_offset: Offset added to unpacked explicit zero points
            before dequantization. GPTQv1 stores zero points minus one.

    Returns:
        Output matrix [M, N], same dtype as a.
    """
    assert a.is_contiguous(), "Activation matrix must be contiguous"
    assert b_q.is_contiguous(), "Weight matrix must be contiguous"
    assert scales.is_contiguous(), "Scales must be contiguous"
    assert bit_width in (2, 4), f"Unsupported bit width: {bit_width}"

    M, K = a.shape
    pack_factor = 32 // bit_width
    bit_mask = (1 << bit_width) - 1
    N = b_q.shape[1] * pack_factor

    assert b_q.shape == (K, N // pack_factor), (
        f"b_q shape mismatch: {b_q.shape} vs ({K}, {N // pack_factor})"
    )
    assert scales.shape == (K // group_size, N), (
        f"scales shape mismatch: {scales.shape} vs ({K // group_size}, {N})"
    )
    if qzeros is not None:
        assert qzeros.shape == (K // group_size, N // pack_factor), (
            f"qzeros shape mismatch: {qzeros.shape}"
        )

    c = torch.empty((M, N), dtype=a.dtype, device=a.device)

    has_zp = qzeros is not None
    # Provide a dummy pointer when HAS_ZP=False (Triton requires a valid ptr)
    zeros_ptr = qzeros if has_zp else b_q

    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx1x

        if on_gfx1x():
            # Tuned for RDNA 3.5 (gfx1151, 40 CUs, 32-wide wavefronts).
            if M <= 32:
                BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 64
            elif M <= 64:
                BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
            else:
                BLOCK_M, BLOCK_N, BLOCK_K = 128, 32, 64
        else:
            # Tuned for MI300 (gfx942, 304 CUs, 64-wide wavefronts).
            if M <= 32:
                BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 32
            elif M <= 64:
                BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
            else:
                BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    else:
        if M <= 32:
            BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 32
        elif M <= 64:
            BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        else:
            BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32

    # The kernel loads scales/zeros for a single group per BLOCK_K tile
    # (one g_idx per iteration). If BLOCK_K > group_size, rows at the tail
    # of the tile dequantize with the wrong group's scales, silently
    # corrupting the output. Clamp BLOCK_K to group_size to keep one
    # scale group per tile.
    if group_size < BLOCK_K:
        BLOCK_K = group_size

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    triton_w4a16_gemm_kernel[grid](
        a,
        b_q,
        scales,
        zeros_ptr,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b_q.stride(0),
        b_q.stride(1),
        c.stride(0),
        c.stride(1),
        group_size=group_size,
        BIT_WIDTH=bit_width,
        PACK_FACTOR=pack_factor,
        BIT_MASK=bit_mask,
        ZERO_POINT_OFFSET=zero_point_offset,
        HAS_ZP=has_zp,
        ZP_BIAS=zp_bias,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return c


class TritonW4A16LinearKernel(MPLinearKernel):
    """
    Triton-based W4A16 GEMM kernel for ROCm (MI300 and newer).

    Supports GPTQ-format 2-bit and 4-bit weights with grouped quantization.
    Weight tensors are normalized from AutoGPTQ and compressed-tensors loader
    layouts to the kernel's [K, N//pack] layout.
    """

    SUPPORTED_QUANT_TYPES = TRITON_W4A16_SUPPORTED_QUANT_TYPES

    @classmethod
    def get_min_capability(cls) -> int:
        # Triton handles capability checks itself
        return 0

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not (current_platform.is_rocm() or current_platform.is_cuda()):
            return False, "TritonW4A16LinearKernel requires CUDA or ROCm"

        if current_platform.is_rocm():
            from vllm.platforms.rocm import on_mi3xx

            if not on_mi3xx():
                return (
                    False,
                    "TritonW4A16LinearKernel is only enabled for ROCm CDNA "
                    "MI3xx targets; RDNA targets must use a native RDNA or "
                    "Conch W4A16 backend",
                )

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type {c.weight_type} not supported; "
                f"supported: {cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "Only float16/bfloat16 activations are supported"

        pack_factor = 32 // c.weight_type.size_bits
        N = c.partition_weight_shape[1]
        if N % pack_factor != 0:
            return (
                False,
                f"Output features ({N}) must be divisible by {pack_factor} "
                f"({pack_factor} {c.weight_type.size_bits}-bit values packed per int32)",
            )

        if c.has_g_idx:
            return (
                False,
                "Activation reordering (g_idx) is not supported by "
                "TritonW4A16LinearKernel",
            )

        gs = c.group_size
        if (
            gs not in TRITON_W4A16_SUPPORTED_GROUP_SIZES
            and gs != c.full_weight_shape[0]
        ):
            return (
                False,
                f"Group size {gs} not supported; "
                f"supported: {TRITON_W4A16_SUPPORTED_GROUP_SIZES} "
                f"or full K ({c.full_weight_shape[0]})",
            )

        K = c.partition_weight_shape[0]
        eff_gs = gs if gs != -1 else K
        if K % eff_gs != 0:
            return (False, f"Input features {K} not divisible by group size {eff_gs}")

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Convert checkpoint/loader layouts to kernel layout.

        Compressed-tensors checkpoint layout:
          weight_packed:     [N, K//pack]  int32   input_dim=1, output_dim=0, packed_dim=1
          weight_scale:      [N, K//G]  fp16    input_dim=1, output_dim=0
          weight_zero_point: [N//pack, K//G] int32  output_dim=0, packed_dim=0

        AutoGPTQ loader layout:
          qweight: [K//pack, N]    int32  input_dim=0, output_dim=1, packed_dim=0
          scales:  [K//G, N]    fp16   input_dim=0, output_dim=1
          qzeros:  [K//G, N//pack] int32  input_dim=0, output_dim=1, packed_dim=1

        Kernel needs:
          qweight: [K, N//pack]  int32
          scales:  [K//G, N]  fp16    (transpose weight_scale)
          qzeros:  [K//G, N//pack] int32
        """
        c = self.config
        K, N = c.partition_weight_shape
        group_size = c.group_size if c.group_size != -1 else K
        bit_width = c.weight_type.size_bits
        pack_factor = 32 // bit_width
        bit_mask = (1 << bit_width) - 1
        expected_weight_shape = (K, N // pack_factor)
        k_packed_weight_shape = (K // pack_factor, N)
        transposed_k_packed_weight_shape = (N, K // pack_factor)
        expected_scale_shape = (K // group_size, N)
        transposed_scale_shape = (N, K // group_size)
        expected_zero_shape = (K // group_size, N // pack_factor)
        transposed_zero_shape = (N // pack_factor, K // group_size)

        # Repack K-packed loader/checkpoint weights into the kernel's N-packed
        # layout. AutoGPTQ loaders use [K//pack, N]; compressed-tensors-style
        # checkpoints can use [N, K//pack]. The kernel always consumes
        # [K, N//pack].
        def repack_w_q(x: BasevLLMParameter) -> BasevLLMParameter:
            w = x.data
            input_dim = getattr(x, "input_dim", None)
            output_dim = getattr(x, "output_dim", None)
            packed_dim = getattr(x, "packed_dim", None)
            shifts = torch.arange(
                pack_factor,
                device=w.device,
                dtype=torch.int32,
            ) * bit_width

            if (
                input_dim == 1
                and output_dim == 0
                and packed_dim == 1
                and tuple(w.shape) == transposed_k_packed_weight_shape
            ):
                w_NK = ((w.unsqueeze(-1) >> shifts) & bit_mask).reshape(N, K)
                w_KN = w_NK.t().contiguous()
            elif (
                input_dim == 0
                and output_dim == 1
                and packed_dim == 0
                and tuple(w.shape) == k_packed_weight_shape
            ):
                w_KN = (
                    ((w[:, None, :] >> shifts[None, :, None]) & bit_mask)
                    .reshape(K, N)
                    .contiguous()
                )
            elif tuple(w.shape) == expected_weight_shape:
                x.data = w.contiguous()
                return x
            elif tuple(w.shape) == k_packed_weight_shape:
                w_KN = (
                    ((w[:, None, :] >> shifts[None, :, None]) & bit_mask)
                    .reshape(K, N)
                    .contiguous()
                )
            elif tuple(w.shape) == transposed_k_packed_weight_shape:
                w_NK = ((w.unsqueeze(-1) >> shifts) & bit_mask).reshape(N, K)
                w_KN = w_NK.t().contiguous()
            else:
                raise ValueError(
                    "Unsupported packed-weight layout for "
                    f"{self.__class__.__name__}: got {tuple(w.shape)}, "
                    f"expected {expected_weight_shape}, {k_packed_weight_shape}, "
                    f"or {transposed_k_packed_weight_shape} for K={K}, N={N}, "
                    f"bit_width={bit_width}, pack_factor={pack_factor}."
                )

            # Repack N into N//pack int32 values → [K, N//pack].
            N_packed = N // pack_factor
            w_repacked = torch.sum(
                (w_KN.view(K, N_packed, pack_factor) & bit_mask) << shifts,
                dim=2,
                dtype=torch.int32,
            )
            x.data = w_repacked.contiguous()
            return x

        def repack_w_s(x: BasevLLMParameter) -> BasevLLMParameter:
            scales = x.data
            input_dim = getattr(x, "input_dim", None)
            output_dim = getattr(x, "output_dim", None)
            if (
                input_dim == 1
                and output_dim == 0
                and tuple(scales.shape) == transposed_scale_shape
            ):
                x.data = scales.t().contiguous()
            elif (
                input_dim == 0
                and output_dim == 1
                and tuple(scales.shape) == expected_scale_shape
            ):
                x.data = scales.contiguous()
            elif tuple(scales.shape) == expected_scale_shape:
                x.data = scales.contiguous()
            elif tuple(scales.shape) == transposed_scale_shape:
                x.data = scales.t().contiguous()
            else:
                raise ValueError(
                    "Unsupported scale layout for "
                    f"{self.__class__.__name__}: got {tuple(scales.shape)}, "
                    f"expected {expected_scale_shape} or {transposed_scale_shape} "
                    f"for K={K}, N={N}, group_size={group_size}."
                )
            return x

        self._transform_param(layer, self.w_q_name, repack_w_q)
        self._transform_param(layer, self.w_s_name, repack_w_s)

        if self.w_zp_name is not None:
            zp = getattr(layer, self.w_zp_name, None)
            if zp is not None:
                zero_points = zp.data
                input_dim = getattr(zp, "input_dim", None)
                output_dim = getattr(zp, "output_dim", None)
                packed_dim = getattr(zp, "packed_dim", None)
                if (
                    output_dim == 0
                    and packed_dim == 0
                    and tuple(zero_points.shape) == transposed_zero_shape
                ):
                    normalized_zero_points = zero_points.t().contiguous()
                elif (
                    input_dim == 0
                    and output_dim == 1
                    and packed_dim == 1
                    and tuple(zero_points.shape) == expected_zero_shape
                ):
                    normalized_zero_points = zero_points.contiguous()
                elif tuple(zero_points.shape) == expected_zero_shape:
                    normalized_zero_points = zero_points.contiguous()
                elif tuple(zero_points.shape) == transposed_zero_shape:
                    normalized_zero_points = zero_points.t().contiguous()
                else:
                    raise ValueError(
                        "Unsupported zero-point layout for "
                        f"{self.__class__.__name__}: got {tuple(zero_points.shape)}, "
                        f"expected {expected_zero_shape} or {transposed_zero_shape} "
                        f"for K={K}, N={N}, group_size={group_size}, "
                        f"pack_factor={pack_factor}."
                    )

                replace_parameter(
                    layer,
                    self.w_zp_name,
                    torch.nn.Parameter(
                        normalized_zero_points,
                        requires_grad=False,
                    ),
                )

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        c = self.config
        w_q, w_s, w_zp, _ = self._get_weight_params(layer)

        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        K = c.partition_weight_shape[0]
        group_size = c.group_size if c.group_size != -1 else K

        # For biased symmetric types, use the scalar bias when no zeros tensor
        # is supplied.
        zp_bias = c.weight_type.bias if c.weight_type.has_bias() else 0

        output = triton_w4a16_gemm(
            a=x_2d,
            b_q=w_q,
            scales=w_s,
            qzeros=w_zp,
            group_size=group_size,
            bit_width=c.weight_type.size_bits,
            zp_bias=zp_bias,
            zero_point_offset=c.zero_point_offset,
        )

        if bias is not None:
            output.add_(bias)

        return output.reshape(out_shape)
