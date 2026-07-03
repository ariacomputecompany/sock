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
    PlannerHostSnapshot {
        operating_system: OperatingSystem::Linux,
        accelerator_vendor: AcceleratorVendor::Nvidia,
        gpu_arches: vec!["sm90".to_owned()],
        cuda_version: "12.4".to_owned(),
        driver_version: "550.54".to_owned(),
        python_abi: "cp311".to_owned(),
        libc_abi: "glibc-2.35".to_owned(),
        flashinfer_prebuilt_available: true,
    }
}

#[must_use]
pub fn default_request() -> RawRequest {
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
            operating_system: OperatingSystem::Linux,
            accelerator_vendor: AcceleratorVendor::Nvidia,
            gpu_arches: vec!["sm90".to_owned()],
            cuda_version: "12.4".to_owned(),
            driver_version: "550.54".to_owned(),
            python_abi: "cp311".to_owned(),
            libc_abi: "glibc-2.35".to_owned(),
        },
        topology: ExecutionTopology {
            tensor_parallelism: 2,
            pipeline_parallelism: 1,
            replicas: 1,
        },
        backend_policy: BackendPolicy {
            preferred_families: vec![
                BackendFamily::FlashInfer,
                BackendFamily::Triton,
                BackendFamily::CudaGraphs,
            ],
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
                    value: "2".to_owned(),
                }],
            },
        ],
    }
}

#[must_use]
pub fn default_request_with_optimization(level: OptimizationLevel) -> RawRequest {
    let mut request = default_request();
    request.optimization_policy.level = level;
    request
}

pub fn plan_outcome() -> Result<PlanningOutcome, PlanError> {
    Planner::new(default_host_snapshot()).resolve(default_request())
}

pub fn plan_outcome_scoped(scope: &BuildScope) -> Result<PlanningOutcome, PlanError> {
    Planner::new(default_host_snapshot()).resolve_scoped(default_request(), scope)
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
        materialization_report,
        replay_proof,
        vllm_integration,
        soc_plan,
        vllm_entrypoints,
    }
}
