"""CUDA-shaped inference contract execution.

This module does not generate model tokens. It validates the exact CUDA runtime
surfaces a real inference request depends on before the request can be trusted:
build ABI, device capability, KV cache metadata, attention backend selection,
CUDA graph capture legality, distributed topology, and quantization gates.
"""

from __future__ import annotations

from dataclasses import dataclass

from sock_cuda_shim.diagnostics import CudaReadinessReport, evaluate_readiness
from sock_cuda_shim.kv_cache import TMHPhysicalPolicy
from sock_cuda_shim.scenarios import CudaScenario


@dataclass(frozen=True)
class CudaInferenceContractReport:
    scenario_name: str
    ready: bool
    selected_attention_backend: str | None
    kv_layout: str
    total_tokens: int
    graph_capture_required: bool
    tmh_pressure: dict[str, float | int | str] | None
    readiness: CudaReadinessReport


def run_inference_contract(
    scenario: CudaScenario,
    *,
    tmh_policy: TMHPhysicalPolicy | None = None,
) -> CudaInferenceContractReport:
    readiness = evaluate_readiness(
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
    tmh_pressure = None
    if tmh_policy is not None:
        tmh_pressure = tmh_policy.pressure(
            scenario.kv_spec,
            total_tokens=scenario.request.total_tokens,
        )
    return CudaInferenceContractReport(
        scenario_name=scenario.name,
        ready=readiness.ok,
        selected_attention_backend=readiness.selected_attention_backend,
        kv_layout=scenario.kv_spec.layout.value,
        total_tokens=scenario.request.total_tokens,
        graph_capture_required=scenario.graph is not None,
        tmh_pressure=tmh_pressure,
        readiness=readiness,
    )
