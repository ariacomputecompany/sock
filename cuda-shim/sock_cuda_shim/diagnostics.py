"""Production readiness diagnostics for CUDA-shaped sock runs."""

from __future__ import annotations

from dataclasses import dataclass

from sock_cuda_shim.attention import AttentionRequestShape, select_attention_backend
from sock_cuda_shim.build import CudaBuildContract
from sock_cuda_shim.capabilities import CudaShimError
from sock_cuda_shim.cuda_graphs import CudaGraphPlan
from sock_cuda_shim.device import CudaDevice
from sock_cuda_shim.distributed import DistributedPlan
from sock_cuda_shim.environment import CudaEnvironment
from sock_cuda_shim.kv_cache import KVLayout, KVPageSpec, PagedKVRequest, TMHPhysicalPolicy
from sock_cuda_shim.quantization import QuantizationPlan


@dataclass(frozen=True)
class CudaReadinessReport:
    ok: bool
    checks: tuple[str, ...]
    failures: tuple[str, ...]
    selected_attention_backend: str | None = None
    kv_memory_pressure: dict[str, float | int | str] | None = None


def evaluate_readiness(
    *,
    devices: tuple[CudaDevice, ...],
    env: CudaEnvironment,
    build: CudaBuildContract,
    kv_spec: KVPageSpec,
    request: PagedKVRequest,
    attention_shape: AttentionRequestShape,
    graph_plan: CudaGraphPlan | None = None,
    distributed_plan: DistributedPlan | None = None,
    quantization_plan: QuantizationPlan | None = None,
    tmh_policy: TMHPhysicalPolicy | None = None,
    gpu_memory_utilization: float = 0.90,
    gpu_memory_reserve_bytes: int = 0,
) -> CudaReadinessReport:
    checks: list[str] = []
    failures: list[str] = []
    backend: str | None = None
    kv_memory_pressure: dict[str, float | int | str] | None = None

    def run(name: str, fn) -> None:
        try:
            fn()
            checks.append(name)
        except (CudaShimError, MemoryError, ValueError) as exc:
            failures.append(f"{name}: {exc}")

    if not devices:
        failures.append("devices: no CUDA devices visible")
        return CudaReadinessReport(False, tuple(checks), tuple(failures), backend)

    device = devices[0]
    run("environment", env.validate)
    run("build", lambda: build.validate_for(device, env))
    run("kv_request", lambda: request.validate(kv_spec))
    try:
        kv_memory_pressure = _kv_memory_pressure(
            device=device,
            kv_spec=kv_spec,
            request=request,
            tmh_policy=tmh_policy,
            gpu_memory_utilization=gpu_memory_utilization,
            gpu_memory_reserve_bytes=gpu_memory_reserve_bytes,
        )
        checks.append("kv_memory_budget")
    except (CudaShimError, MemoryError, ValueError) as exc:
        failures.append(f"kv_memory_budget: {exc}")
    run("attention", lambda: None)
    try:
        backend = select_attention_backend(device, env, attention_shape).value
    except (CudaShimError, ValueError) as exc:
        failures.append(f"attention: {exc}")
    if graph_plan is not None:
        run("cuda_graphs", lambda: graph_plan.validate(device))
    if distributed_plan is not None:
        run("distributed", lambda: distributed_plan.validate(devices, env))
    if quantization_plan is not None:
        run("quantization", lambda: quantization_plan.validate(device))
    return CudaReadinessReport(
        ok=not failures,
        checks=tuple(checks),
        failures=tuple(failures),
        selected_attention_backend=backend,
        kv_memory_pressure=kv_memory_pressure,
    )


def _kv_memory_pressure(
    *,
    device: CudaDevice,
    kv_spec: KVPageSpec,
    request: PagedKVRequest,
    tmh_policy: TMHPhysicalPolicy | None,
    gpu_memory_utilization: float,
    gpu_memory_reserve_bytes: int,
) -> dict[str, float | int | str]:
    request.validate(kv_spec)
    total_pages = kv_spec.pages_for_tokens(request.total_tokens)
    regular_bytes = kv_spec.page_bytes * total_pages
    tmh_pressure = (
        tmh_policy.pressure(kv_spec, request.total_tokens)
        if tmh_policy is not None and kv_spec.layout is KVLayout.TMH_FIDELITY_PAGED
        else None
    )
    required_bytes = (
        int(tmh_pressure["tmh_effective_bytes"])
        if tmh_pressure is not None
        else regular_bytes
    )
    budget_bytes = device.memory_budget(gpu_memory_utilization, gpu_memory_reserve_bytes)
    if required_bytes > budget_bytes:
        raise MemoryError(
            "KV cache requires "
            f"{required_bytes} bytes, exceeding CUDA memory budget {budget_bytes} bytes"
        )
    pressure: dict[str, float | int | str] = {
        "layout": kv_spec.layout.value,
        "total_pages": total_pages,
        "regular_bytes": regular_bytes,
        "required_bytes": required_bytes,
        "budget_bytes": budget_bytes,
        "budget_utilization_pct": 100.0 * required_bytes / budget_bytes,
    }
    if tmh_pressure is not None:
        pressure["tmh_effective_bytes"] = int(tmh_pressure["tmh_effective_bytes"])
        pressure["tmh_reduction_pct"] = float(tmh_pressure["reduction_pct"])
    return pressure
