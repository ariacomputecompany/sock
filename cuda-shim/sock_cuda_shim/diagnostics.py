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
from sock_cuda_shim.kv_cache import KVPageSpec, PagedKVRequest
from sock_cuda_shim.quantization import QuantizationPlan


@dataclass(frozen=True)
class CudaReadinessReport:
    ok: bool
    checks: tuple[str, ...]
    failures: tuple[str, ...]
    selected_attention_backend: str | None = None


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
) -> CudaReadinessReport:
    checks: list[str] = []
    failures: list[str] = []
    backend: str | None = None

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
    )
