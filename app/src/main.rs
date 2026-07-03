use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use sock_app::{
    default_host_snapshot, diagnostics_for, plan_outcome, plan_outcome_scoped, replay_bundle,
    rewrite_trace_for,
};
use sock_core::{
    BackendFamily, DiagnosticsDocument, MaterializationExecutionReport, ReplayBundle,
    ReplayBundleMetadata, RewriteTraceDocument, canonical_json, render_diagnostics, render_explain,
    render_plan_summary, render_verification_report,
};
use sock_engine::{
    BuildReadiness, BuildScope, MaterializationExecutor, PlannerHostSnapshot, PlanningOutcome,
    build_vllm_integration_document,
};

#[derive(Debug, Parser)]
#[command(name = "sock")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Plan {
        #[command(flatten)]
        scope: ScopeArgs,
        #[arg(long, value_enum, default_value_t = OutputMode::Summary)]
        format: OutputMode,
    },
    Explain {
        #[command(flatten)]
        scope: ScopeArgs,
        #[arg(long, value_enum, default_value_t = OutputMode::Summary)]
        format: OutputMode,
    },
    Build {
        #[arg(long)]
        out: PathBuf,
        #[command(flatten)]
        scope: ScopeArgs,
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

#[derive(Debug, Clone, Parser, Default)]
struct ScopeArgs {
    #[arg(long = "region")]
    regions: Vec<String>,
    #[arg(long = "artifact-scope")]
    artifact_scopes: Vec<String>,
    #[arg(long = "backend-family", value_enum)]
    backend_families: Vec<BackendFamilyArg>,
    #[arg(long = "cache-namespace")]
    cache_namespaces: Vec<String>,
    #[arg(long = "warmup-scope")]
    warmup_scopes: Vec<String>,
    #[arg(long, value_enum)]
    readiness: Option<ReadinessArg>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ReadinessArg {
    Correctness,
    Performance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BackendFamilyArg {
    Flashinfer,
    Triton,
    AotInductor,
    CudaGraphs,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Plan { scope, format } => emit_plan(&plan(&scope.into_scope())?, format)?,
        Command::Explain { scope, format } => {
            let outcome = plan_with_scope(&scope.into_scope())?;
            let diagnostics = diagnostics_for(&outcome);
            let rewrite_trace = rewrite_trace_for(&outcome);
            emit_explain(&outcome, &diagnostics, &rewrite_trace, format)?;
        }
        Command::Build { out, scope, format } => {
            let outcome = plan_with_scope(&scope.into_scope())?;
            let bundle = replay_bundle(&outcome);
            let materialization = MaterializationExecutor::new().execute(&outcome, &out)?;
            let vllm_integration = build_vllm_integration_document(&outcome)?;
            std::fs::write(
                out.join("vllm_integration.json"),
                canonical_json(&vllm_integration)?.as_bytes(),
            )?;
            let metadata = bundle.write_to(&out)?;
            emit_build(&out, &bundle, &metadata, &materialization, format)?;
        }
        Command::Verify { bundle, format } => {
            emit_verify(&ReplayBundle::load_from(&bundle)?, format)?;
        }
        Command::Replay { bundle, format } => {
            emit_replay(&ReplayBundle::load_from(&bundle)?, format)?;
        }
        Command::Doctor { format } => emit_doctor(&default_host_snapshot(), format)?,
    }

    Ok(())
}

impl ScopeArgs {
    fn into_scope(self) -> BuildScope {
        BuildScope {
            region_names: self.regions.into_iter().collect(),
            artifact_scopes: self.artifact_scopes.into_iter().collect(),
            backend_families: self
                .backend_families
                .into_iter()
                .map(|family| match family {
                    BackendFamilyArg::Flashinfer => BackendFamily::FlashInfer,
                    BackendFamilyArg::Triton => BackendFamily::Triton,
                    BackendFamilyArg::AotInductor => BackendFamily::AotInductor,
                    BackendFamilyArg::CudaGraphs => BackendFamily::CudaGraphs,
                })
                .collect(),
            cache_namespaces: self.cache_namespaces.into_iter().collect(),
            warmup_scopes: self.warmup_scopes.into_iter().collect(),
            readiness: self.readiness.map(|readiness| match readiness {
                ReadinessArg::Correctness => BuildReadiness::Correctness,
                ReadinessArg::Performance => BuildReadiness::Performance,
            }),
        }
    }
}

fn plan(scope: &BuildScope) -> Result<PlanningOutcome> {
    Ok(if scope.is_unscoped() {
        plan_outcome()?
    } else {
        plan_outcome_scoped(scope)?
    })
}

fn plan_with_scope(scope: &BuildScope) -> Result<PlanningOutcome> {
    plan(scope)
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
    materialization: &MaterializationExecutionReport,
    format: OutputMode,
) -> Result<()> {
    match format {
        OutputMode::Summary => {
            println!(
                "bundle={} plan_identity={} replay_entrypoint={} artifacts={} reused={} bytes_written={}",
                out.display(),
                bundle.build_plan.structural_identity.plan_identity,
                metadata.replay_entrypoint,
                materialization.artifact_count,
                materialization.reused_artifact_count,
                materialization.total_bytes_written
            );
        }
        OutputMode::Json => {
            println!(
                "{}",
                canonical_json(&serde_json::json!({
                    "bundle_path": out,
                    "plan_identity": bundle.build_plan.structural_identity.plan_identity,
                    "metadata": metadata,
                    "materialization": materialization,
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
