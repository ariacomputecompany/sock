use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use sock_core::{
    AcceleratorVendor, BackendFamily, BackendPolicy, CachePolicy, ConfigEntry, ConfigLayer,
    CoveragePlane, DiagnosticsDocument, EngineSource, ExecutionTopology, FailureMode,
    GuaranteeLevel, GuaranteeTarget, ModelRef, OperatingSystem, RawRequest, ReplayBundle,
    ReplayBundleMetadata, RequestedEnvironment, RewriteTraceDocument, ShapePoint, ShapePolicy,
    ShapeRange, TargetEngine, WarmupPolicy, canonical_json, render_diagnostics, render_explain,
    render_plan_summary, render_verification_report,
};
use sock_engine::{Planner, PlannerHostSnapshot, PlanningOutcome, vllm};

#[derive(Debug, Parser)]
#[command(name = "sock")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Plan {
        #[arg(long, value_enum, default_value_t = OutputMode::Summary)]
        format: OutputMode,
    },
    Explain {
        #[arg(long, value_enum, default_value_t = OutputMode::Summary)]
        format: OutputMode,
    },
    Build {
        #[arg(long)]
        out: PathBuf,
        #[arg(long, value_enum, default_value_t = OutputMode::Summary)]
        format: OutputMode,
    },
    Verify {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long, value_enum, default_value_t = OutputMode::Summary)]
        format: OutputMode,
    },
    Replay {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long, value_enum, default_value_t = OutputMode::Summary)]
        format: OutputMode,
    },
    Doctor {
        #[arg(long, value_enum, default_value_t = OutputMode::Summary)]
        format: OutputMode,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum OutputMode {
    Summary,
    Json,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Plan { format } => emit_plan(&plan()?, format)?,
        Command::Explain { format } => {
            let outcome = plan()?;
            let diagnostics = diagnostics_for(&outcome);
            let rewrite_trace = rewrite_trace_for(&outcome);
            emit_explain(&outcome, &diagnostics, &rewrite_trace, format)?;
        }
        Command::Build { out, format } => {
            let outcome = plan()?;
            let bundle = replay_bundle(&outcome);
            let metadata = bundle.write_to(&out)?;
            emit_build(&out, &bundle, &metadata, format)?;
        }
        Command::Verify { bundle, format } => {
            emit_verify(&ReplayBundle::load_from(&bundle)?, format)?;
        }
        Command::Replay { bundle, format } => {
            emit_replay(&ReplayBundle::load_from(&bundle)?, format)?;
        }
        Command::Doctor { format } => emit_doctor(&planner_host(), format)?,
    }

    Ok(())
}

fn planner_host() -> PlannerHostSnapshot {
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

fn production_request() -> RawRequest {
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
    }
}

fn plan() -> Result<PlanningOutcome> {
    Ok(Planner::new(planner_host()).resolve(production_request())?)
}

fn diagnostics_for(outcome: &PlanningOutcome) -> DiagnosticsDocument {
    DiagnosticsDocument::from_outcome(
        &outcome.plan,
        &outcome.verification,
        &outcome.plan.rewrite_trace,
    )
}

fn rewrite_trace_for(outcome: &PlanningOutcome) -> RewriteTraceDocument {
    RewriteTraceDocument::new(&outcome.plan, outcome.plan.rewrite_trace.clone())
}

fn replay_bundle(outcome: &PlanningOutcome) -> ReplayBundle {
    ReplayBundle {
        build_plan: outcome.plan.clone(),
        artifact_closure: outcome.closure.clone(),
        verification_report: outcome.verification.clone(),
        diagnostics: diagnostics_for(outcome),
        rewrite_trace: rewrite_trace_for(outcome),
    }
}

fn emit_plan(outcome: &PlanningOutcome, format: OutputMode) -> Result<()> {
    match format {
        OutputMode::Summary => print!("{}", render_plan_summary(&outcome.plan)),
        OutputMode::Json => println!("{}", canonical_json(&outcome.plan)?),
    }
    Ok(())
}

fn emit_explain(
    outcome: &PlanningOutcome,
    diagnostics: &DiagnosticsDocument,
    rewrite_trace: &RewriteTraceDocument,
    format: OutputMode,
) -> Result<()> {
    match format {
        OutputMode::Summary => {
            print!(
                "{}",
                render_explain(&outcome.plan, diagnostics, rewrite_trace)
            );
        }
        OutputMode::Json => {
            println!(
                "{}",
                canonical_json(&serde_json::json!({
                    "plan": outcome.plan,
                    "diagnostics": diagnostics,
                    "rewrite_trace": rewrite_trace,
                    "verification": outcome.verification,
                }))?
            );
        }
    }
    Ok(())
}

fn emit_build(
    out: &Path,
    bundle: &ReplayBundle,
    metadata: &ReplayBundleMetadata,
    format: OutputMode,
) -> Result<()> {
    match format {
        OutputMode::Summary => {
            println!(
                "bundle={} plan_identity={} replay_entrypoint={}",
                out.display(),
                bundle.build_plan.structural_identity.plan_identity,
                metadata.replay_entrypoint
            );
        }
        OutputMode::Json => {
            println!(
                "{}",
                canonical_json(&serde_json::json!({
                    "bundle_path": out,
                    "plan_identity": bundle.build_plan.structural_identity.plan_identity,
                    "metadata": metadata,
                }))?
            );
        }
    }
    Ok(())
}

fn emit_verify(bundle: &ReplayBundle, format: OutputMode) -> Result<()> {
    match format {
        OutputMode::Summary => print!(
            "{}",
            render_verification_report(&bundle.verification_report)
        ),
        OutputMode::Json => println!("{}", canonical_json(&bundle.verification_report)?),
    }
    Ok(())
}

fn emit_replay(bundle: &ReplayBundle, format: OutputMode) -> Result<()> {
    match format {
        OutputMode::Summary => {
            print!("{}", render_plan_summary(&bundle.build_plan));
            print!(
                "{}",
                render_verification_report(&bundle.verification_report)
            );
            print!("{}", render_diagnostics(&bundle.diagnostics));
        }
        OutputMode::Json => {
            println!(
                "{}",
                canonical_json(&serde_json::json!({
                    "plan": bundle.build_plan,
                    "verification": bundle.verification_report,
                    "diagnostics": bundle.diagnostics,
                }))?
            );
        }
    }
    Ok(())
}

fn emit_doctor(host: &PlannerHostSnapshot, format: OutputMode) -> Result<()> {
    match format {
        OutputMode::Summary => {
            println!(
                "host os={:?} vendor={:?} arches={} cuda={} driver={} python_abi={} libc_abi={} flashinfer_prebuilt={}",
                host.operating_system,
                host.accelerator_vendor,
                host.gpu_arches.join(","),
                host.cuda_version,
                host.driver_version,
                host.python_abi,
                host.libc_abi,
                host.flashinfer_prebuilt_available
            );
        }
        OutputMode::Json => {
            println!(
                "{}",
                canonical_json(&serde_json::json!({
                    "operating_system": format!("{:?}", host.operating_system),
                    "accelerator_vendor": format!("{:?}", host.accelerator_vendor),
                    "gpu_arches": host.gpu_arches,
                    "cuda_version": host.cuda_version,
                    "driver_version": host.driver_version,
                    "python_abi": host.python_abi,
                    "libc_abi": host.libc_abi,
                    "flashinfer_prebuilt_available": host.flashinfer_prebuilt_available,
                }))?
            );
        }
    }
    Ok(())
}
