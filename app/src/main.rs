use anyhow::Result;
use clap::{Parser, ValueEnum};
use sock_core::{
    AcceleratorVendor, BackendFamily, BackendPolicy, CachePolicy, ConfigEntry, ConfigLayer,
    CoveragePlane, EngineSource, ExecutionTopology, FailureMode, GuaranteeLevel, GuaranteeTarget,
    ModelRef, OperatingSystem, RawRequest, RequestedEnvironment, ShapePoint, ShapePolicy,
    ShapeRange, TargetEngine, WarmupPolicy, canonical_json,
};
use sock_engine::{Planner, PlannerHostSnapshot, vllm};

#[derive(Debug, Clone, Parser)]
struct Cli {
    #[arg(long, value_enum, default_value_t = OutputMode::Summary)]
    format: OutputMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum OutputMode {
    Summary,
    Json,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let planner = Planner::new(PlannerHostSnapshot {
        operating_system: OperatingSystem::Linux,
        accelerator_vendor: AcceleratorVendor::Nvidia,
        gpu_arches: vec!["sm90".to_owned()],
        cuda_version: "12.4".to_owned(),
        driver_version: "550.54".to_owned(),
        python_abi: "cp311".to_owned(),
        libc_abi: "glibc-2.35".to_owned(),
        flashinfer_prebuilt_available: true,
    });
    let outcome = planner.resolve(RawRequest {
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
            require_prebuilt_artifacts: true,
            allow_runtime_jit: false,
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
    })?;

    match cli.format {
        OutputMode::Summary => {
            println!("engine={}", outcome.plan.normalized_request.engine.as_str());
            println!(
                "plan_identity={}",
                outcome.plan.structural_identity.plan_identity
            );
            println!(
                "primary_backend={}",
                outcome.plan.selected_backends.primary.family.as_str()
            );
            println!(
                "correctness_guarantee={:?}",
                outcome.plan.guarantee_envelope.achieved_correctness
            );
            println!(
                "performance_guarantee={:?}",
                outcome.plan.guarantee_envelope.achieved_performance
            );
            println!("verification={:?}", outcome.verification.status);
            println!("vllm_revision={}", vllm::revision());
        }
        OutputMode::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "plan": serde_json::from_str::<serde_json::Value>(&canonical_json(&outcome.plan)?)?,
                    "closure": serde_json::from_str::<serde_json::Value>(&canonical_json(&outcome.closure)?)?,
                    "verification": serde_json::from_str::<serde_json::Value>(&canonical_json(&outcome.verification)?)?,
                    "vllm_root": vllm::root(),
                    "vllm_revision": vllm::revision(),
                }))?
            );
        }
    }

    Ok(())
}
