use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use sock_app::{
    default_host_snapshot, diagnostics_for, plan_outcome, plan_outcome_scoped, replay_bundle,
    rewrite_trace_for,
};
use sock_core::{
    BackendFamily, BuildMeasurementReport, DiagnosticsDocument, MaterializationExecutionReport,
    MeasurementCaseReport, MeasurementComparisonReport, ReplayBundle, ReplayBundleMetadata,
    ResolvedBuildPlan, RewriteTraceDocument, SchemaVersion, canonical_json, render_diagnostics,
    render_explain, render_plan_summary, render_verification_report,
};
use sock_engine::{
    BuildReadiness, BuildScope, BuildTopologyScope, MaterializationExecutor, PlannerHostSnapshot,
    PlanningOutcome, StorageRoots, build_vllm_entrypoint_document, build_vllm_integration_document,
    emit_vllm_entrypoints, validate_scoped_vllm_subset,
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
        #[arg(long)]
        cache_root: Option<PathBuf>,
        #[command(flatten)]
        scope: ScopeArgs,
        #[arg(long, value_enum, default_value_t = OutputMode::Summary)]
        format: OutputMode,
    },
    Prepare {
        #[arg(value_enum)]
        intent: PrepareIntentArg,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        cache_root: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = OutputMode::Summary)]
        format: OutputMode,
    },
    Measure {
        #[arg(value_enum)]
        intent: PrepareIntentArg,
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

#[derive(Debug, Clone, Parser, Default)]
struct ScopeArgs {
    #[arg(long = "region")]
    regions: Vec<String>,
    #[arg(long = "artifact-scope")]
    artifact_scopes: Vec<String>,
    #[arg(long = "backend-family", value_enum)]
    backend_families: Vec<BackendFamilyArg>,
    #[arg(long = "topology-scope", value_enum)]
    topology_scopes: Vec<TopologyScopeArg>,
    #[arg(long = "cache-namespace")]
    cache_namespaces: Vec<String>,
    #[arg(long = "warmup-scope")]
    warmup_scopes: Vec<String>,
    #[arg(long, value_enum)]
    readiness: Option<ReadinessArg>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ReadinessArg {
    EarlyServe,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum TopologyScopeArg {
    Shared,
    RankLocal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum PrepareIntentArg {
    PrefillPath,
    DecodePath,
    DistributedFlashinferStartup,
    ReplaySafeClosure,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Plan { scope, format } => {
            let scope = scope.into_scope();
            emit_plan(
                &scope,
                request_label_for_scope(&scope),
                &plan(&scope)?,
                format,
            )?
        }
        Command::Explain { scope, format } => {
            let scope = scope.into_scope();
            let outcome = plan_with_scope(&scope)?;
            let diagnostics = diagnostics_for(&outcome);
            let rewrite_trace = rewrite_trace_for(&outcome);
            emit_explain(
                &scope,
                request_label_for_scope(&scope),
                &outcome,
                &diagnostics,
                &rewrite_trace,
                format,
            )?;
        }
        Command::Build {
            out,
            cache_root,
            scope,
            format,
        } => {
            let scope = scope.into_scope();
            let outcome = plan_with_scope(&scope)?;
            let bundle = replay_bundle(&outcome);
            let vllm_integration = build_vllm_integration_document(&outcome)?;
            validate_scoped_vllm_subset(&scope, &vllm_integration)?;
            let storage = StorageRoots {
                bundle_root: out.clone(),
                cache_root: cache_root.unwrap_or_else(|| out.join(".sock-cache")),
            };
            let materialization =
                MaterializationExecutor::new().execute(&outcome, &scope, &storage)?;
            std::fs::write(
                out.join("vllm_integration.json"),
                canonical_json(&vllm_integration)?.as_bytes(),
            )?;
            let vllm_entrypoints = build_vllm_entrypoint_document(&outcome, &vllm_integration)?;
            emit_vllm_entrypoints(&out, &vllm_entrypoints)?;
            let metadata = bundle.write_to(&out)?;
            emit_build(
                &scope,
                request_label_for_scope(&scope),
                &out,
                &bundle,
                &metadata,
                &materialization,
                format,
            )?;
        }
        Command::Prepare {
            intent,
            out,
            cache_root,
            format,
        } => {
            let scope = intent_scope(intent);
            let build = materialize_bundle(
                &scope,
                &out,
                &cache_root.unwrap_or_else(|| out.join(".sock-cache")),
            )?;
            emit_build(
                &scope,
                Some(intent_label(intent)),
                &out,
                &build.bundle,
                &build.metadata,
                &build.materialization,
                format,
            )?;
        }
        Command::Measure {
            intent,
            out,
            format,
        } => emit_measure(intent, &out, format)?,
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
            topology_scopes: self
                .topology_scopes
                .into_iter()
                .map(|scope| match scope {
                    TopologyScopeArg::Shared => BuildTopologyScope::Shared,
                    TopologyScopeArg::RankLocal => BuildTopologyScope::RankLocal,
                })
                .collect(),
            cache_namespaces: self.cache_namespaces.into_iter().collect(),
            warmup_scopes: self.warmup_scopes.into_iter().collect(),
            readiness: self.readiness.map(|readiness| match readiness {
                ReadinessArg::EarlyServe => BuildReadiness::EarlyServe,
                ReadinessArg::Correctness => BuildReadiness::Correctness,
                ReadinessArg::Performance => BuildReadiness::Performance,
            }),
        }
    }
}

fn intent_scope(intent: PrepareIntentArg) -> BuildScope {
    match intent {
        PrepareIntentArg::PrefillPath => BuildScope {
            region_names: ["prefill_attention".to_owned()].into_iter().collect(),
            readiness: Some(BuildReadiness::Correctness),
            ..BuildScope::default()
        },
        PrepareIntentArg::DecodePath => BuildScope {
            backend_families: [BackendFamily::CudaGraphs].into_iter().collect(),
            topology_scopes: [BuildTopologyScope::RankLocal].into_iter().collect(),
            readiness: Some(BuildReadiness::Performance),
            ..BuildScope::default()
        },
        PrepareIntentArg::DistributedFlashinferStartup => BuildScope {
            readiness: Some(BuildReadiness::Correctness),
            ..BuildScope::default()
        },
        PrepareIntentArg::ReplaySafeClosure => BuildScope {
            readiness: Some(BuildReadiness::Performance),
            ..BuildScope::default()
        },
    }
}

fn intent_label(intent: PrepareIntentArg) -> &'static str {
    match intent {
        PrepareIntentArg::PrefillPath => "prefill_path",
        PrepareIntentArg::DecodePath => "decode_path",
        PrepareIntentArg::DistributedFlashinferStartup => "distributed_flashinfer_startup",
        PrepareIntentArg::ReplaySafeClosure => "replay_safe_closure",
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

fn emit_plan(
    scope: &BuildScope,
    request_label: Option<&'static str>,
    outcome: &PlanningOutcome,
    format: OutputMode,
) -> Result<()> {
    match format {
        OutputMode::Summary => {
            print!("{}", render_plan_summary(&outcome.plan));
            print!(
                "{}",
                render_request_contract(scope, request_label, &outcome.plan)
            );
        }
        OutputMode::Json => println!("{}", canonical_json(&outcome.plan)?),
    }
    Ok(())
}

fn emit_explain(
    scope: &BuildScope,
    request_label: Option<&'static str>,
    outcome: &PlanningOutcome,
    diagnostics: &DiagnosticsDocument,
    rewrite_trace: &RewriteTraceDocument,
    format: OutputMode,
) -> Result<()> {
    match format {
        OutputMode::Summary => {
            print!("{}", render_plan_summary(&outcome.plan));
            print!(
                "{}",
                render_request_contract(scope, request_label, &outcome.plan)
            );
            print!("{}", render_vllm_native_contract(outcome)?);
            print!(
                "{}",
                render_explain(&outcome.plan, diagnostics, rewrite_trace)
                    .strip_prefix(&render_plan_summary(&outcome.plan))
                    .unwrap_or("")
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
                    "vllm_integration": build_vllm_integration_document(outcome)?,
                }))?
            );
        }
    }
    Ok(())
}

fn emit_build(
    scope: &BuildScope,
    request_label: Option<&'static str>,
    out: &Path,
    bundle: &ReplayBundle,
    metadata: &ReplayBundleMetadata,
    materialization: &MaterializationExecutionReport,
    format: OutputMode,
) -> Result<()> {
    match format {
        OutputMode::Summary => {
            println!(
                "bundle={} plan_identity={} replay_entrypoint={} artifacts={} executed={} reused={} wall_clock_ms={} bytes_written={} rebuild_ms={} readiness={:?}",
                out.display(),
                bundle.build_plan.structural_identity.plan_identity,
                metadata.replay_entrypoint,
                materialization.artifact_count,
                materialization.executed_artifact_count,
                materialization.reused_artifact_count,
                materialization.wall_clock_ms,
                materialization.total_bytes_written,
                materialization.total_rebuild_ms,
                materialization.readiness.achieved_readiness
            );
            print!(
                "{}",
                render_request_contract(scope, request_label, &bundle.build_plan)
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
                    "vllm_integration": bundle.vllm_integration,
                    "vllm_entrypoints": bundle.vllm_entrypoints,
                }))?
            );
        }
    }
    Ok(())
}

fn emit_measure(intent: PrepareIntentArg, out: &Path, format: OutputMode) -> Result<()> {
    std::fs::create_dir_all(out)?;

    let broad_out = out.join("broad-cold");
    let scoped_cold_out = out.join("scoped-cold");
    let scoped_warm_out = out.join("scoped-warm");

    let broad_cache = out.join(".sock-cache-broad");
    let scoped_cache = out.join(".sock-cache-scoped");

    let broad_scope = BuildScope::default();
    let scoped_scope = intent_scope(intent);

    let broad = materialize_bundle(&broad_scope, &broad_out, &broad_cache)?;
    let scoped_cold = materialize_bundle(&scoped_scope, &scoped_cold_out, &scoped_cache)?;
    let scoped_warm = materialize_bundle(&scoped_scope, &scoped_warm_out, &scoped_cache)?;

    let report = BuildMeasurementReport {
        schema_version: SchemaVersion::current(),
        intent: intent_label(intent).to_owned(),
        broad_cold: measurement_case("broad_cold", &broad_scope, &broad.materialization),
        scoped_cold: measurement_case("scoped_cold", &scoped_scope, &scoped_cold.materialization),
        scoped_warm: measurement_case("scoped_warm", &scoped_scope, &scoped_warm.materialization),
        scoped_vs_broad: MeasurementComparisonReport::between(
            "broad_cold",
            &measurement_case("broad_cold", &broad_scope, &broad.materialization),
            "scoped_cold",
            &measurement_case("scoped_cold", &scoped_scope, &scoped_cold.materialization),
        ),
        warm_vs_cold: MeasurementComparisonReport::between(
            "scoped_cold",
            &measurement_case("scoped_cold", &scoped_scope, &scoped_cold.materialization),
            "scoped_warm",
            &measurement_case("scoped_warm", &scoped_scope, &scoped_warm.materialization),
        ),
    };
    std::fs::write(
        out.join("measurement_report.json"),
        canonical_json(&report)?.as_bytes(),
    )?;

    match format {
        OutputMode::Summary => {
            println!(
                "measurement intent={} broad_wall_clock_ms={} scoped_wall_clock_ms={} warm_wall_clock_ms={} scoped_wall_clock_reduction_bps={} warm_wall_clock_reduction_bps={} broad_executed={} scoped_executed={} warm_reused={}",
                report.intent,
                report.broad_cold.wall_clock_ms,
                report.scoped_cold.wall_clock_ms,
                report.scoped_warm.wall_clock_ms,
                report.scoped_vs_broad.wall_clock_reduction_bps,
                report.warm_vs_cold.wall_clock_reduction_bps,
                report.broad_cold.executed_artifact_count,
                report.scoped_cold.executed_artifact_count,
                report.scoped_warm.reused_artifact_count,
            );
        }
        OutputMode::Json => println!("{}", canonical_json(&report)?),
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
            println!(
                "vllm replay roots key={} surfaces={}",
                bundle.vllm_integration.plan_identity,
                bundle.vllm_integration.replay_roots.len()
            );
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
                    "vllm_integration": bundle.vllm_integration,
                    "vllm_entrypoints": bundle.vllm_entrypoints,
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

fn request_label_for_scope(scope: &BuildScope) -> Option<&'static str> {
    if scope.is_unscoped() {
        None
    } else {
        Some("custom_scope")
    }
}

fn render_request_contract(
    scope: &BuildScope,
    request_label: Option<&'static str>,
    plan: &ResolvedBuildPlan,
) -> String {
    let mut out = String::new();
    out.push_str("request contract:\n");
    if let Some(label) = request_label {
        out.push_str(&format!("  - intent={label}\n"));
    }
    out.push_str(&format!(
        "  - requested selectors: regions={} artifact_scopes={} backends={} topology={} caches={} warmups={} readiness={}\n",
        list_or_all(scope.region_names.iter().map(String::as_str)),
        list_or_all(scope.artifact_scopes.iter().map(String::as_str)),
        list_or_all(scope.backend_families.iter().map(|family| family.as_str())),
        list_or_all(scope.topology_scopes.iter().map(|scope| match scope {
            BuildTopologyScope::Shared => "shared",
            BuildTopologyScope::RankLocal => "rank_local",
        })),
        list_or_all(scope.cache_namespaces.iter().map(String::as_str)),
        list_or_all(scope.warmup_scopes.iter().map(String::as_str)),
        match scope.readiness {
            Some(BuildReadiness::EarlyServe) => "early_serve",
            Some(BuildReadiness::Correctness) => "correctness",
            Some(BuildReadiness::Performance) => "performance",
            None => "default",
        }
    ));

    let expanded_regions = plan
        .compile_regions
        .iter()
        .map(|region| region.name.as_str())
        .collect::<Vec<_>>();
    let expanded_artifacts = plan
        .artifact_requirements
        .iter()
        .map(|artifact| format!("{}:{}", artifact.class.as_str(), artifact.scope))
        .collect::<Vec<_>>();
    let expanded_warmups = plan
        .warmup_obligations
        .iter()
        .map(|obligation| obligation.proof.proof_id.clone())
        .collect::<Vec<_>>();
    let expanded_topology = plan
        .artifact_requirements
        .iter()
        .map(|artifact| match artifact.rank_disposition {
            sock_core::RankDisposition::Shared => "shared",
            sock_core::RankDisposition::RankLocal => "rank_local",
        })
        .collect::<BTreeSet<_>>();

    out.push_str(&format!(
        "  - expanded closure: regions={} artifacts={} warmups={} topology={}\n",
        list_or_all(expanded_regions.into_iter()),
        list_or_all(expanded_artifacts.iter().map(String::as_str)),
        list_or_all(expanded_warmups.iter().map(String::as_str)),
        list_or_all(expanded_topology.into_iter()),
    ));

    let compile_ms = plan
        .materialization_graph
        .waves
        .iter()
        .filter_map(|wave| wave.estimate.expected_compile_ms)
        .sum::<u64>();
    let transfer_ms = plan
        .materialization_graph
        .waves
        .iter()
        .filter_map(|wave| wave.estimate.expected_transfer_ms)
        .sum::<u64>();
    let bytes = plan
        .materialization_graph
        .waves
        .iter()
        .filter_map(|wave| wave.estimate.expected_bytes_written)
        .sum::<u64>();
    out.push_str(&format!(
        "  - estimated work: waves={} compile_ms={} transfer_ms={} bytes_written={}\n",
        plan.materialization_graph.waves.len(),
        compile_ms,
        transfer_ms,
        bytes
    ));

    let expansion_reasons = plan
        .artifact_requirements
        .iter()
        .map(|artifact| {
            format!(
                "{}:{} because {}",
                artifact.class.as_str(),
                artifact.scope,
                artifact
                    .admissibility
                    .rationale
                    .first()
                    .map(String::as_str)
                    .unwrap_or("the selected scope requires this artifact")
            )
        })
        .chain(plan.warmup_obligations.iter().map(|obligation| {
            format!(
                "{} because requested readiness requires {} proof coverage",
                obligation.proof.proof_id,
                if obligation.blocking {
                    "blocking"
                } else {
                    "deferred"
                }
            )
        }))
        .collect::<Vec<_>>();
    out.push_str("  - pulled in by closure:\n");
    for reason in expansion_reasons {
        out.push_str(&format!("    {}\n", reason));
    }
    out
}

fn render_vllm_native_contract(outcome: &PlanningOutcome) -> Result<String> {
    let integration = build_vllm_integration_document(outcome)?;
    let preserved_inputs = integration
        .surfaces
        .iter()
        .flat_map(|surface| surface.preserved_inputs.iter().cloned())
        .collect::<BTreeSet<_>>();
    let preserved_abstractions = integration
        .surfaces
        .iter()
        .flat_map(|surface| surface.preserved_abstractions.iter().cloned())
        .collect::<BTreeSet<_>>();
    let mut out = String::new();
    out.push_str("vllm native contract:\n");
    out.push_str(&format!(
        "  - preserved inputs: {}\n",
        list_or_all(preserved_inputs.iter().map(String::as_str))
    ));
    out.push_str(&format!(
        "  - preserved abstractions: {}\n",
        list_or_all(preserved_abstractions.iter().map(String::as_str))
    ));
    out.push_str(&format!(
        "  - replay root key: {}\n",
        integration.plan_identity
    ));
    out.push_str(&format!(
        "  - rooted vllm replay surfaces: {}\n",
        integration
            .replay_roots
            .iter()
            .map(|root| {
                format!(
                    "{}@{}:{}",
                    root.scope_name,
                    root.cache_namespace
                        .as_deref()
                        .unwrap_or("no-cache-namespace"),
                    root.manifest_paths.join("|")
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    ));
    Ok(out)
}

struct BundleBuild {
    bundle: ReplayBundle,
    metadata: ReplayBundleMetadata,
    materialization: MaterializationExecutionReport,
}

fn materialize_bundle(scope: &BuildScope, out: &Path, cache_root: &Path) -> Result<BundleBuild> {
    let outcome = plan_with_scope(scope)?;
    let bundle = replay_bundle(&outcome);
    let vllm_integration = build_vllm_integration_document(&outcome)?;
    validate_scoped_vllm_subset(scope, &vllm_integration)?;
    let storage = StorageRoots {
        bundle_root: out.to_path_buf(),
        cache_root: cache_root.to_path_buf(),
    };
    let materialization = MaterializationExecutor::new().execute(&outcome, scope, &storage)?;
    std::fs::write(
        out.join("vllm_integration.json"),
        canonical_json(&vllm_integration)?.as_bytes(),
    )?;
    let vllm_entrypoints = build_vllm_entrypoint_document(&outcome, &vllm_integration)?;
    emit_vllm_entrypoints(out, &vllm_entrypoints)?;
    let metadata = bundle.write_to(out)?;

    Ok(BundleBuild {
        bundle,
        metadata,
        materialization,
    })
}

fn measurement_case(
    label: &str,
    scope: &BuildScope,
    report: &MaterializationExecutionReport,
) -> MeasurementCaseReport {
    let mut requested_backend_families = scope
        .backend_families
        .iter()
        .map(|family| family.as_str().to_owned())
        .collect::<Vec<_>>();
    requested_backend_families.sort();

    let mut requested_topology_scopes = scope
        .topology_scopes
        .iter()
        .map(|scope| match scope {
            BuildTopologyScope::Shared => "shared".to_owned(),
            BuildTopologyScope::RankLocal => "rank_local".to_owned(),
        })
        .collect::<Vec<_>>();
    requested_topology_scopes.sort();

    let runtime_jit_contradiction_count = report
        .runtime_jit_observations
        .iter()
        .filter(|observation| {
            observation.status == sock_core::RuntimeJitObservationStatus::Contradicted
        })
        .count() as u32;

    MeasurementCaseReport {
        label: label.to_owned(),
        requested_regions: scope.region_names.iter().cloned().collect(),
        requested_artifact_scopes: scope.artifact_scopes.iter().cloned().collect(),
        requested_backend_families,
        requested_topology_scopes,
        requested_cache_namespaces: scope.cache_namespaces.iter().cloned().collect(),
        requested_warmup_scopes: scope.warmup_scopes.iter().cloned().collect(),
        requested_readiness: match scope.readiness {
            Some(BuildReadiness::EarlyServe) => "early_serve".to_owned(),
            Some(BuildReadiness::Correctness) => "correctness".to_owned(),
            Some(BuildReadiness::Performance) => "performance".to_owned(),
            None => "default".to_owned(),
        },
        plan_identity: report.plan_identity.to_string(),
        artifact_count: report.artifact_count,
        executed_artifact_count: report.executed_artifact_count,
        reused_artifact_count: report.reused_artifact_count,
        wall_clock_ms: report.wall_clock_ms,
        total_compile_ms: report.total_compile_ms,
        total_transfer_ms: report.total_transfer_ms,
        total_rebuild_ms: report.total_rebuild_ms,
        total_bytes_written: report.total_bytes_written,
        runtime_jit_contradiction_count,
    }
}

fn list_or_all<'a>(items: impl IntoIterator<Item = &'a str>) -> String {
    let collected = items
        .into_iter()
        .filter(|item| !item.is_empty())
        .collect::<Vec<_>>();
    if collected.is_empty() {
        "all".to_owned()
    } else {
        collected.join(",")
    }
}
