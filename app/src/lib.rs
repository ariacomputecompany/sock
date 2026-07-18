use std::process::Command;
use std::sync::OnceLock;

use sock_core::{
    AcceleratorVendor, BackendFamily, BackendPolicy, CachePolicy, ConfigEntry, ConfigLayer,
    CoveragePlane, DiagnosticsDocument, EngineSource, ExecutionTopology, FailureMode,
    GuaranteeLevel, GuaranteeTarget, MaterializationExecutionReport, ModelRef, OperatingSystem,
    OptimizationExplainDocument, OptimizationLevel, OptimizationPolicy, RawRequest, ReplayBundle,
    ReplayProofDocument, RequestedEnvironment, RewriteTraceDocument, ShapePoint, ShapePolicy,
    ShapeRange, TargetEngine, WarmupPolicy,
};
use sock_engine::{
    BuildScope, PlanError, Planner, PlannerHostSnapshot, PlanningOutcome, build_soc_plan_document,
    build_vllm_entrypoint_document, build_vllm_integration_document, vllm,
};

#[must_use]
pub fn default_host_snapshot() -> PlannerHostSnapshot {
    static SNAPSHOT: OnceLock<PlannerHostSnapshot> = OnceLock::new();
    SNAPSHOT.get_or_init(detect_host_snapshot).clone()
}

#[must_use]
pub fn default_request() -> RawRequest {
    default_request_for_host(&default_host_snapshot())
}

#[must_use]
pub fn default_request_for_host(host: &PlannerHostSnapshot) -> RawRequest {
    RawRequest {
        engine: TargetEngine::Vllm,
        model: ModelRef {
            repository: "meta-llama/Llama-3.1-8B-Instruct".to_owned(),
            revision: "main".to_owned(),
        },
        engine_source: EngineSource {
            kind: "vendored".to_owned(),
            revision: vllm::revision().to_owned(),
        },
        environment: RequestedEnvironment {
            operating_system: host.operating_system,
            accelerator_vendor: host.accelerator_vendor,
            gpu_arches: host.gpu_arches.clone(),
            cuda_version: host.cuda_version.clone(),
            driver_version: host.driver_version.clone(),
            python_abi: host.python_abi.clone(),
            libc_abi: host.libc_abi.clone(),
        },
        topology: ExecutionTopology {
            tensor_parallelism: default_tensor_parallelism(host.accelerator_vendor),
            pipeline_parallelism: 1,
            replicas: 1,
        },
        backend_policy: BackendPolicy {
            preferred_families: preferred_backend_families(host.accelerator_vendor),
            packaging_strategy: sock_core::PackagingStrategy::PreferPrebuiltThenAot,
            runtime_jit_policy: sock_core::RuntimeJitPolicy {
                disposition: sock_core::RuntimeJitDisposition::Forbidden,
                max_residual_node_count: 0,
            },
            correctness_target: GuaranteeTarget {
                level: GuaranteeLevel::ShapeBoundedAot,
                failure_mode: FailureMode::FailClosed,
            },
            performance_target: GuaranteeTarget {
                level: GuaranteeLevel::WarmupBounded,
                failure_mode: FailureMode::FailClosed,
            },
        },
        shape_policy: ShapePolicy {
            correctness_range: ShapeRange {
                min_batch_size: 1,
                max_batch_size: 8,
                min_sequence_length: 1,
                max_sequence_length: 4096,
            },
            performance_range: ShapeRange {
                min_batch_size: 1,
                max_batch_size: 4,
                min_sequence_length: 1,
                max_sequence_length: 2048,
            },
            hot_shapes: vec![
                ShapePoint {
                    batch_size: 1,
                    sequence_length: 128,
                    plane: CoveragePlane::Performance,
                },
                ShapePoint {
                    batch_size: 4,
                    sequence_length: 2048,
                    plane: CoveragePlane::Performance,
                },
            ],
            cuda_graph_shapes: vec![ShapePoint {
                batch_size: 1,
                sequence_length: 128,
                plane: CoveragePlane::CudaGraph,
            }],
        },
        cache_policy: CachePolicy {
            namespace: "prod".to_owned(),
            allow_cross_machine_reuse: false,
        },
        warmup_policy: WarmupPolicy {
            max_warmup_steps: 6,
            verify_cuda_graph_capture: true,
        },
        optimization_policy: OptimizationPolicy {
            level: OptimizationLevel::O2,
        },
        layered_config: vec![
            ConfigLayer {
                name: "env".to_owned(),
                precedence: 0,
                entries: vec![ConfigEntry {
                    key: "VLLM_USE_V1".to_owned(),
                    value: "1".to_owned(),
                }],
            },
            ConfigLayer {
                name: "project".to_owned(),
                precedence: 1,
                entries: vec![ConfigEntry {
                    key: "tensor_parallel_size".to_owned(),
                    value: default_tensor_parallelism(host.accelerator_vendor).to_string(),
                }],
            },
        ],
    }
}

fn preferred_backend_families(vendor: AcceleratorVendor) -> Vec<BackendFamily> {
    match vendor {
        AcceleratorVendor::Nvidia => vec![
            BackendFamily::FlashInfer,
            BackendFamily::Triton,
            BackendFamily::CudaGraphs,
        ],
        AcceleratorVendor::Amd => vec![BackendFamily::Triton],
        AcceleratorVendor::Unknown => Vec::new(),
    }
}

fn default_tensor_parallelism(vendor: AcceleratorVendor) -> u16 {
    match vendor {
        AcceleratorVendor::Nvidia => 2,
        AcceleratorVendor::Amd | AcceleratorVendor::Unknown => 1,
    }
}

#[must_use]
pub fn default_request_with_optimization(level: OptimizationLevel) -> RawRequest {
    let mut request = default_request();
    request.optimization_policy.level = level;
    request
}

pub fn plan_outcome() -> Result<PlanningOutcome, PlanError> {
    let host = default_host_snapshot();
    Planner::new(host.clone()).resolve(default_request_for_host(&host))
}

pub fn plan_outcome_scoped(scope: &BuildScope) -> Result<PlanningOutcome, PlanError> {
    let host = default_host_snapshot();
    Planner::new(host.clone()).resolve_scoped(default_request_for_host(&host), scope)
}

#[must_use]
pub fn diagnostics_for(outcome: &PlanningOutcome) -> DiagnosticsDocument {
    DiagnosticsDocument::from_outcome(
        &outcome.plan,
        &outcome.verification,
        &outcome.plan.rewrite_trace,
    )
}

#[must_use]
pub fn rewrite_trace_for(outcome: &PlanningOutcome) -> RewriteTraceDocument {
    RewriteTraceDocument::new(&outcome.plan, outcome.plan.rewrite_trace.clone())
}

#[must_use]
pub fn replay_bundle(
    outcome: &PlanningOutcome,
    scope: &BuildScope,
    materialization_report: MaterializationExecutionReport,
) -> ReplayBundle {
    let vllm_integration =
        build_vllm_integration_document(outcome).expect("vllm integration document");
    let soc_plan = build_soc_plan_document(outcome, scope, &vllm_integration);
    let optimization_explain = OptimizationExplainDocument::from_plan(&outcome.plan);
    let backend_decision = sock_core::BackendDecisionDocument::from_plan(&outcome.plan);
    let replay_proof =
        ReplayProofDocument::from_plan_and_materialization(&outcome.plan, &materialization_report)
            .expect("replay proof");
    let vllm_entrypoints = build_vllm_entrypoint_document(outcome, &vllm_integration)
        .expect("vllm entrypoint document");
    ReplayBundle {
        build_plan: outcome.plan.clone(),
        artifact_closure: outcome.closure.clone(),
        verification_report: outcome.verification.clone(),
        diagnostics: diagnostics_for(outcome),
        rewrite_trace: rewrite_trace_for(outcome),
        optimization_explain,
        backend_decision,
        materialization_report,
        replay_proof,
        vllm_integration,
        soc_plan,
        vllm_entrypoints,
    }
}

fn detect_host_snapshot() -> PlannerHostSnapshot {
    if let Some(host) = host_snapshot_from_env() {
        return host;
    }

    let operating_system = OperatingSystem::Linux;
    let python_abi = detect_python_abi();
    let libc_abi = detect_libc_abi();

    if let Some(host) = detect_rocm_snapshot(operating_system, &python_abi, &libc_abi) {
        return host;
    }
    if let Some(host) = detect_nvidia_snapshot(operating_system, &python_abi, &libc_abi) {
        return host;
    }

    if cfg!(target_os = "macos") {
        return PlannerHostSnapshot {
            operating_system,
            accelerator_vendor: AcceleratorVendor::Nvidia,
            gpu_arches: vec!["sm90".to_owned()],
            cuda_version: "12.4".to_owned(),
            driver_version: "550.54".to_owned(),
            python_abi,
            libc_abi,
            flashinfer_prebuilt_available: true,
        };
    }

    PlannerHostSnapshot {
        operating_system,
        accelerator_vendor: AcceleratorVendor::Unknown,
        gpu_arches: Vec::new(),
        cuda_version: String::new(),
        driver_version: String::new(),
        python_abi,
        libc_abi,
        flashinfer_prebuilt_available: false,
    }
}

fn host_snapshot_from_env() -> Option<PlannerHostSnapshot> {
    let profile = std::env::var("SOCK_HOST_PROFILE").ok()?;
    match profile.as_str() {
        "nvidia-sm90" => Some(PlannerHostSnapshot {
            operating_system: OperatingSystem::Linux,
            accelerator_vendor: AcceleratorVendor::Nvidia,
            gpu_arches: vec!["sm90".to_owned()],
            cuda_version: "12.4".to_owned(),
            driver_version: "550.54".to_owned(),
            python_abi: "cp311".to_owned(),
            libc_abi: "glibc-2.35".to_owned(),
            flashinfer_prebuilt_available: true,
        }),
        "amd-gfx1151" => Some(PlannerHostSnapshot {
            operating_system: OperatingSystem::Linux,
            accelerator_vendor: AcceleratorVendor::Amd,
            gpu_arches: vec!["gfx1151".to_owned()],
            cuda_version: "7.14".to_owned(),
            driver_version: "7.14".to_owned(),
            python_abi: "cp312".to_owned(),
            libc_abi: "glibc-2.39".to_owned(),
            flashinfer_prebuilt_available: false,
        }),
        "unknown" => Some(PlannerHostSnapshot {
            operating_system: OperatingSystem::Linux,
            accelerator_vendor: AcceleratorVendor::Unknown,
            gpu_arches: Vec::new(),
            cuda_version: String::new(),
            driver_version: String::new(),
            python_abi: "unknown".to_owned(),
            libc_abi: "unknown".to_owned(),
            flashinfer_prebuilt_available: false,
        }),
        _ => None,
    }
}

fn detect_rocm_snapshot(
    operating_system: OperatingSystem,
    python_abi: &str,
    libc_abi: &str,
) -> Option<PlannerHostSnapshot> {
    let output = run_command("rocm_agent_enumerator", &[])?;
    let mut gpu_arches = output
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("gfx"))
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    gpu_arches.sort();
    gpu_arches.dedup();
    if gpu_arches.is_empty() {
        return None;
    }

    let rocm_version = run_command("dpkg-query", &["-W", "-f=${Version}", "amdrocm"])
        .or_else(|| read_file_trimmed("/opt/rocm/.info/version"))
        .unwrap_or_default();

    Some(PlannerHostSnapshot {
        operating_system,
        accelerator_vendor: AcceleratorVendor::Amd,
        gpu_arches,
        cuda_version: rocm_version.clone(),
        driver_version: rocm_version,
        python_abi: python_abi.to_owned(),
        libc_abi: libc_abi.to_owned(),
        flashinfer_prebuilt_available: false,
    })
}

fn detect_nvidia_snapshot(
    operating_system: OperatingSystem,
    python_abi: &str,
    libc_abi: &str,
) -> Option<PlannerHostSnapshot> {
    let output = run_command(
        "nvidia-smi",
        &["--query-gpu=compute_cap,driver_version", "--format=csv,noheader"],
    )?;
    let first_line = output.lines().map(str::trim).find(|line| !line.is_empty())?;
    let mut parts = first_line.split(',').map(str::trim);
    let compute_cap = parts.next()?;
    let driver_version = parts.next().unwrap_or_default().to_owned();
    let gpu_arches = vec![parse_nvidia_compute_capability(compute_cap)];
    let cuda_version = run_command("nvidia-smi", &[])
        .and_then(|text| parse_cuda_version_from_nvidia_smi(&text))
        .unwrap_or_default();

    Some(PlannerHostSnapshot {
        operating_system,
        accelerator_vendor: AcceleratorVendor::Nvidia,
        gpu_arches,
        cuda_version,
        driver_version,
        python_abi: python_abi.to_owned(),
        libc_abi: libc_abi.to_owned(),
        flashinfer_prebuilt_available: true,
    })
}

fn detect_python_abi() -> String {
    run_command(
        "python3",
        &[
            "-c",
            "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')",
        ],
    )
    .unwrap_or_else(|| "unknown".to_owned())
}

fn detect_libc_abi() -> String {
    run_command("ldd", &["--version"])
        .and_then(|output| output.lines().next().map(parse_glibc_abi))
        .unwrap_or_else(|| "unknown".to_owned())
}

fn parse_glibc_abi(line: &str) -> String {
    let version = line
        .split_whitespace()
        .rev()
        .find(|token| token.chars().next().is_some_and(|ch| ch.is_ascii_digit()))
        .unwrap_or("unknown");
    format!("glibc-{version}")
}

fn parse_nvidia_compute_capability(value: &str) -> String {
    let digits = value
        .chars()
        .filter(|ch| ch.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() {
        return value.trim().to_owned();
    }
    format!("sm{digits}")
}

fn parse_cuda_version_from_nvidia_smi(output: &str) -> Option<String> {
    let marker = "CUDA Version:";
    let idx = output.find(marker)?;
    let rest = &output[idx + marker.len()..];
    let version = rest.split_whitespace().next()?;
    Some(version.to_owned())
}

fn read_file_trimmed(path: &str) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|contents| contents.trim().to_owned())
        .filter(|contents| !contents.is_empty())
}

fn run_command(cmd: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(cmd).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        parse_cuda_version_from_nvidia_smi, parse_glibc_abi, parse_nvidia_compute_capability,
    };

    #[test]
    fn parses_nvidia_compute_capability_into_sm_arch() {
        assert_eq!(parse_nvidia_compute_capability("9.0"), "sm90");
        assert_eq!(parse_nvidia_compute_capability("8.9"), "sm89");
    }

    #[test]
    fn parses_cuda_version_from_nvidia_smi_output() {
        let text = "Driver Version: 550.54       CUDA Version: 12.4";
        assert_eq!(
            parse_cuda_version_from_nvidia_smi(text).as_deref(),
            Some("12.4")
        );
    }

    #[test]
    fn parses_glibc_abi_from_ldd_output() {
        assert_eq!(
            parse_glibc_abi("ldd (Ubuntu GLIBC 2.39-0ubuntu8.7) 2.39"),
            "glibc-2.39"
        );
    }
}
