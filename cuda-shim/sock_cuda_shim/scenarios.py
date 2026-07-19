"""Canonical adversarial NVIDIA scenarios for sock CUDA work."""

from __future__ import annotations

from dataclasses import dataclass

from sock_cuda_shim.attention import AttentionRequestShape
from sock_cuda_shim.build import CudaBuildContract
from sock_cuda_shim.capabilities import ComputeCapability, gib
from sock_cuda_shim.cuda_graphs import CudaGraphPlan
from sock_cuda_shim.device import CudaDevice, DeviceClass
from sock_cuda_shim.distributed import DistributedPlan
from sock_cuda_shim.environment import CudaEnvironment
from sock_cuda_shim.kv_cache import KVLayout, KVPageSpec, PagedKVRequest, Precision
from sock_cuda_shim.quantization import QuantBackend, QuantizationPlan


@dataclass(frozen=True)
class CudaScenario:
    name: str
    devices: tuple[CudaDevice, ...]
    env: CudaEnvironment
    build: CudaBuildContract
    kv_spec: KVPageSpec
    request: PagedKVRequest
    attention: AttentionRequestShape
    graph: CudaGraphPlan | None
    distributed: DistributedPlan | None
    quantization: QuantizationPlan | None
    should_pass: bool
    why: str


def _request(tokens: int, block_tokens: int) -> PagedKVRequest:
    pages = max(1, (tokens + block_tokens - 1) // block_tokens)
    return PagedKVRequest(
        request_id=f"req-{tokens}",
        prompt_tokens=tokens,
        generated_tokens=0,
        slot_mapping=tuple(range(tokens)),
        block_table=tuple(range(pages)),
    )


RTX4090 = CudaDevice.rtx_4090()
H100 = CudaDevice.h100()
B200 = CudaDevice.b200()
MIG_H100_10GB = CudaDevice(
    name="NVIDIA H100 MIG 1g.10gb",
    capability=ComputeCapability(9, 0),
    total_memory_bytes=gib(10),
    device_class=DeviceClass.MIG_SLICE,
    ordinal=0,
    uuid="MIG-H100-0",
    mig_parent_uuid="GPU-H100-0",
)

DEFAULT_BUILD = CudaBuildContract(
    cuda_version="12.8",
    torch_cuda_version="12.8",
    python_abi="cp312",
    compiled_arches=(
        ComputeCapability(8, 0),
        ComputeCapability(8, 6),
        ComputeCapability(8, 9),
        ComputeCapability(9, 0),
        ComputeCapability(10, 0),
        ComputeCapability(12, 0),
    ),
)

DEFAULT_KV = KVPageSpec(
    block_tokens=16,
    num_layers=64,
    num_kv_heads=8,
    head_size_k=128,
    head_size_v=128,
    precision=Precision.FP16,
    layout=KVLayout.FLASHINFER_PAGED,
)

CANONICAL_SCENARIOS: tuple[CudaScenario, ...] = (
    CudaScenario(
        name="rtx4090_single_gpu_flashinfer_gptq",
        devices=(RTX4090,),
        env=CudaEnvironment.from_mapping({"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}),
        build=DEFAULT_BUILD,
        kv_spec=DEFAULT_KV,
        request=_request(2048, DEFAULT_KV.block_tokens),
        attention=AttentionRequestShape(64, 8, 128, 4096, KVLayout.FLASHINFER_PAGED, Precision.FP16),
        graph=CudaGraphPlan(batch_size=4, max_tokens=2048),
        distributed=None,
        quantization=QuantizationPlan(QuantBackend.GPTQ, Precision.FP16),
        should_pass=True,
        why="4090 should exercise the first rented-GPU production path.",
    ),
    CudaScenario(
        name="h100_fp8_cutlass_mla",
        devices=(H100,),
        env=CudaEnvironment.from_mapping({"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}),
        build=DEFAULT_BUILD,
        kv_spec=DEFAULT_KV,
        request=_request(4096, DEFAULT_KV.block_tokens),
        attention=AttentionRequestShape(128, 8, 128, 8192, KVLayout.FLASHINFER_PAGED, Precision.FP8, is_mla=True),
        graph=CudaGraphPlan(batch_size=8, max_tokens=4096),
        distributed=None,
        quantization=QuantizationPlan(QuantBackend.FP8, Precision.FP8, activation_fp8=True),
        should_pass=True,
        why="Hopper should allow FP8 and TMA-backed MLA routing.",
    ),
    CudaScenario(
        name="blackwell_nvfp4",
        devices=(B200,),
        env=CudaEnvironment.from_mapping({"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}),
        build=DEFAULT_BUILD,
        kv_spec=KVPageSpec(16, 64, 8, 128, 128, Precision.NVFP4, KVLayout.TMH_FIDELITY_PAGED),
        request=_request(4096, 16),
        attention=AttentionRequestShape(128, 8, 128, 8192, KVLayout.TMH_FIDELITY_PAGED, Precision.NVFP4),
        graph=CudaGraphPlan(batch_size=8, max_tokens=4096),
        distributed=None,
        quantization=QuantizationPlan(QuantBackend.NVFP4, Precision.NVFP4, per_token_scale=True),
        should_pass=True,
        why="Blackwell should unlock NVFP4/FP4 gates.",
    ),
    CudaScenario(
        name="mig_too_small_for_large_kv",
        devices=(MIG_H100_10GB,),
        env=CudaEnvironment.from_mapping({"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}),
        build=DEFAULT_BUILD,
        kv_spec=DEFAULT_KV,
        request=_request(8192, DEFAULT_KV.block_tokens),
        attention=AttentionRequestShape(64, 8, 128, 8192, KVLayout.FLASHINFER_PAGED, Precision.FP8),
        graph=CudaGraphPlan(batch_size=8, max_tokens=8192),
        distributed=None,
        quantization=QuantizationPlan(QuantBackend.FP8, Precision.FP8),
        should_pass=True,
        why="MIG is valid CUDA but should be evaluated with tight memory budgets.",
    ),
    CudaScenario(
        name="invalid_launch_blocking_with_graphs",
        devices=(RTX4090,),
        env=CudaEnvironment.from_mapping(
            {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_LAUNCH_BLOCKING": "0",
                "VLLM_CUDAGRAPH_CAPTURE_SIZES": "1,2,4",
            }
        ),
        build=DEFAULT_BUILD,
        kv_spec=DEFAULT_KV,
        request=_request(2048, DEFAULT_KV.block_tokens),
        attention=AttentionRequestShape(64, 8, 128, 4096, KVLayout.FLASHINFER_PAGED, Precision.FP16),
        graph=CudaGraphPlan(batch_size=4, max_tokens=2048, forbidden_ops_seen=("cudaMalloc",)),
        distributed=None,
        quantization=QuantizationPlan(QuantBackend.GPTQ, Precision.FP16),
        should_pass=False,
        why="Graph capture must fail if runtime allocation leaks into capture.",
    ),
)
