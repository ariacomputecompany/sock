from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sock_cuda_shim.attention import AttentionBackend, select_attention_backend
from sock_cuda_shim.build import CudaBuildContract
from sock_cuda_shim.capabilities import ComputeCapability, InvalidCudaConfiguration, gib
from sock_cuda_shim.cuda_graphs import CudaGraphPlan
from sock_cuda_shim.device import CudaDevice
from sock_cuda_shim.diagnostics import evaluate_readiness
from sock_cuda_shim.distributed import DistributedPlan
from sock_cuda_shim.environment import CudaEnvironment
from sock_cuda_shim.kv_cache import (
    KVLayout,
    KVPageSpec,
    PagedKVRequest,
    Precision,
    TMHPhysicalPolicy,
)
from sock_cuda_shim.memory import CudaMemoryPool
from sock_cuda_shim.quantization import QuantBackend, QuantizationPlan
from sock_cuda_shim.scenarios import CANONICAL_SCENARIOS, DEFAULT_BUILD


def test_4090_selects_flashinfer_for_paged_kv() -> None:
    env = CudaEnvironment.from_mapping({"CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
    shape = CANONICAL_SCENARIOS[0].attention

    backend = select_attention_backend(CudaDevice.rtx_4090(), env, shape)

    assert backend is AttentionBackend.FLASHINFER


def test_invalid_cuda_device_order_fails_closed() -> None:
    with pytest.raises(InvalidCudaConfiguration, match="PCI_BUS_ID"):
        CudaEnvironment.from_mapping({"CUDA_DEVICE_ORDER": "FASTEST_FIRST"})


def test_build_contract_rejects_missing_arch() -> None:
    build = CudaBuildContract(
        cuda_version="12.8",
        torch_cuda_version="12.8",
        python_abi="cp312",
        compiled_arches=(ComputeCapability(8, 0),),
    )

    with pytest.raises(InvalidCudaConfiguration, match="compiled CUDA arch"):
        build.validate_for(
            CudaDevice.rtx_4090(),
            CudaEnvironment.from_mapping({"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}),
        )


def test_cuda_memory_pool_models_alignment_and_oom() -> None:
    device = CudaDevice.rtx_4090()
    pool = CudaMemoryPool(device=device, usable_bytes=gib(1), chunk_bytes=64 * 1024 * 1024)

    allocation = pool.reserve(1)

    assert allocation.size == 64 * 1024 * 1024
    with pytest.raises(MemoryError):
        pool.reserve(gib(2))


def test_paged_kv_validates_slot_mapping_and_block_table() -> None:
    spec = KVPageSpec(16, 32, 8, 128, 128, Precision.FP16, KVLayout.FLASHINFER_PAGED)
    request = PagedKVRequest(
        request_id="bad",
        prompt_tokens=33,
        generated_tokens=0,
        slot_mapping=tuple(range(33)),
        block_table=(0, 1),
    )

    with pytest.raises(InvalidCudaConfiguration, match="block table"):
        request.validate(spec)


def test_tmh_physical_policy_reduces_kv_pressure() -> None:
    spec = KVPageSpec(16, 64, 8, 128, 128, Precision.FP16, KVLayout.TMH_FIDELITY_PAGED)
    pressure = TMHPhysicalPolicy(hot_budget_pct=25).pressure(spec, total_tokens=4096)

    assert pressure["layout"] == "tmh_fidelity_paged_kv"
    assert pressure["tmh_effective_bytes"] < pressure["regular_bytes"]
    assert pressure["reduction_pct"] > 0


def test_cuda_graph_rejects_capture_time_malloc() -> None:
    plan = CudaGraphPlan(batch_size=4, max_tokens=2048, forbidden_ops_seen=("cudaMalloc",))

    with pytest.raises(InvalidCudaConfiguration, match="forbidden ops"):
        plan.validate(CudaDevice.rtx_4090())


def test_distributed_plan_requires_p2p_peers() -> None:
    devices = (CudaDevice.h100(ordinal=0), CudaDevice.h100(ordinal=1))
    plan = DistributedPlan(tensor_parallel_size=2, requires_p2p=True)
    env = CudaEnvironment.from_mapping({"CUDA_DEVICE_ORDER": "PCI_BUS_ID"})

    with pytest.raises(InvalidCudaConfiguration, match="lacks NVLink"):
        plan.validate(devices, env)


def test_quantization_gates_nvfp4_to_blackwell() -> None:
    plan = QuantizationPlan(QuantBackend.NVFP4, Precision.NVFP4)

    with pytest.raises(Exception, match="nvfp4"):
        plan.validate(CudaDevice.rtx_4090())
    plan.validate(CudaDevice.b200())


@pytest.mark.parametrize("scenario", CANONICAL_SCENARIOS, ids=lambda item: item.name)
def test_canonical_cuda_scenarios_match_expected_outcome(scenario) -> None:
    report = evaluate_readiness(
        devices=scenario.devices,
        env=scenario.env,
        build=scenario.build,
        kv_spec=scenario.kv_spec,
        request=scenario.request,
        attention_shape=scenario.attention,
        graph_plan=scenario.graph,
        distributed_plan=scenario.distributed,
        quantization_plan=scenario.quantization,
    )

    assert report.ok is scenario.should_pass, report.failures
    if report.ok:
        assert report.selected_attention_backend is not None


def test_default_build_covers_first_rented_4090_path() -> None:
    DEFAULT_BUILD.validate_for(
        CudaDevice.rtx_4090(),
        CudaEnvironment.from_mapping(
            {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "TORCH_CUDA_ARCH_LIST": "8.9",
            }
        ),
    )
