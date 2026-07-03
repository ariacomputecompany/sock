use crate::{
    DiagnosticsDocument, HazardClass, MaterializationExecutionReport, OptimizationExplainDocument,
    ReplayProofDocument, ResolvedBuildPlan, RewriteTraceDocument, SocPlanDocument,
    VerificationReport,
};

#[must_use]
pub fn render_plan_summary(plan: &ResolvedBuildPlan) -> String {
    format!(
        "plan {}\nengine {}\nmodel {}@{}\nbackend {:?}\n",
        plan.structural_identity.plan_identity,
        plan.normalized_request.engine.as_str(),
        plan.normalized_request.model.repository,
        plan.normalized_request.model.revision,
        plan.selected_backends.primary.family
    )
}

#[must_use]
pub fn render_explain(
    plan: &ResolvedBuildPlan,
    diagnostics: &DiagnosticsDocument,
    rewrite_trace: &RewriteTraceDocument,
    optimization_explain: &OptimizationExplainDocument,
) -> String {
    let mut out = String::new();
    out.push_str(&render_plan_summary(plan));
    out.push_str(&format!(
        "guarantee correctness {:?} performance {:?}\n",
        plan.guarantee_envelope.achieved_correctness, plan.guarantee_envelope.achieved_performance
    ));
    out.push_str(&render_optimization_explain(optimization_explain));
    out.push_str("rewrite trace:\n");
    for pass in &rewrite_trace.passes {
        out.push_str(&format!(
            "  - {} {} -> {}\n",
            pass.pass_name(),
            pass.before_identity,
            pass.after_identity
        ));
    }
    out.push_str("diagnostics:\n");
    out.push_str(&render_diagnostics(diagnostics));
    out
}

#[must_use]
pub fn render_optimization_explain(document: &OptimizationExplainDocument) -> String {
    let mut out = String::new();
    out.push_str("optimization envelope:\n");
    out.push_str(&format!(
        "  - level={} profile={} startup_budget_ms={} max_warmup_steps={} artifact_budget={} rank_local_budget={}\n",
        document.level.as_str(),
        document.profile_name,
        document.startup_budget_ms,
        document.max_warmup_steps,
        document.artifact_budget.max_artifact_count,
        document.artifact_budget.max_rank_local_artifacts
    ));
    out.push_str(&format!(
        "  - phase_budgets={}\n",
        document
            .phase_budgets
            .iter()
            .map(|budget| format!("{}:{}", budget.phase_name, budget.startup_budget_ms))
            .collect::<Vec<_>>()
            .join("|")
    ));
    out.push_str("  - actions:\n");
    for action in document
        .compile_actions
        .iter()
        .chain(document.warmup_actions.iter())
        .chain(document.autotune_actions.iter())
        .chain(document.graph_actions.iter())
    {
        out.push_str(&format!(
            "    {} {} because {}\n",
            action.category, action.effect, action.reason
        ));
    }
    out
}

#[must_use]
pub fn render_replay_bundle_explain(
    plan: &ResolvedBuildPlan,
    optimization_explain: &OptimizationExplainDocument,
    verification_report: &VerificationReport,
    diagnostics: &DiagnosticsDocument,
    materialization: &MaterializationExecutionReport,
    replay_proof: &ReplayProofDocument,
) -> String {
    let mut out = String::new();
    out.push_str(&render_plan_summary(plan));
    out.push_str(&render_optimization_explain(optimization_explain));
    out.push_str("replay proof:\n");
    out.push_str(&format!(
        "  - compile_closure={} warmup_closure={} autotune_closure={} cache_compatibility={} topology_reuse={}\n",
        replay_proof.compile_closure_verified,
        replay_proof.warmup_closure_verified,
        replay_proof.autotune_closure_verified,
        replay_proof.cache_artifact_compatibility_verified,
        replay_proof.topology_scoped_reuse_verified
    ));
    out.push_str(&format!(
        "  - realization_mode={} result_artifact_identity={} realization_identity={}\n",
        replay_proof.realization_mode.as_str(),
        replay_proof.result_artifact_identity,
        replay_proof.realization_identity
    ));
    out.push_str(&format!(
        "  - compile_artifacts={} warmup_proofs={} autotune_proofs={}\n",
        replay_proof.compile_closure_artifacts.join("|"),
        replay_proof.warmup_closure_proofs.join("|"),
        if replay_proof.autotune_closure_proofs.is_empty() {
            "none".to_owned()
        } else {
            replay_proof.autotune_closure_proofs.join("|")
        }
    ));
    out.push_str(&format!(
        "  - materialization executed={} reused={} closure_outcome={:?}\n",
        materialization.executed_artifact_count,
        materialization.reused_artifact_count,
        materialization.closure_outcome
    ));
    out.push_str(&format!(
        "  - contradiction_contract={}\n",
        replay_proof.contradiction_contract
    ));
    out.push_str(&render_verification_report(verification_report));
    out.push_str("diagnostics:\n");
    out.push_str(&render_diagnostics(diagnostics));
    out
}

#[must_use]
pub fn render_soc_explain(document: &SocPlanDocument) -> String {
    let mut out = String::new();
    out.push_str("soc integration:\n");
    out.push_str(&format!(
        "  - derivation={} plan_identity={} replay_roots={}\n",
        document.derivation_strategy,
        document.plan_identity,
        document.replay_root_ids.len()
    ));
    out.push_str(&format!(
        "  - requested selectors: regions={} artifact_scopes={} backends={} topology={} caches={} warmups={} readiness={}\n",
        if document.selectors.requested_regions.is_empty() {
            "all".to_owned()
        } else {
            document.selectors.requested_regions.join(",")
        },
        if document.selectors.requested_artifact_scopes.is_empty() {
            "all".to_owned()
        } else {
            document.selectors.requested_artifact_scopes.join(",")
        },
        if document.selectors.requested_backend_families.is_empty() {
            "all".to_owned()
        } else {
            document.selectors.requested_backend_families.join(",")
        },
        if document.selectors.requested_topology_scopes.is_empty() {
            "all".to_owned()
        } else {
            document.selectors.requested_topology_scopes.join(",")
        },
        if document.selectors.requested_cache_namespaces.is_empty() {
            "all".to_owned()
        } else {
            document.selectors.requested_cache_namespaces.join(",")
        },
        if document.selectors.requested_warmup_scopes.is_empty() {
            "all".to_owned()
        } else {
            document.selectors.requested_warmup_scopes.join(",")
        },
        document.selectors.requested_readiness
    ));
    for namespace in &document.namespaces {
        out.push_str(&format!(
            "  - namespace={} mode={:?} subset_build_valid={} direct_entrypoint_invocable={} artifacts={} warmups={} replay_roots={} surfaces={}\n",
            namespace.namespace,
            namespace.materialization_mode,
            namespace.subset_build_valid,
            namespace.direct_entrypoint_invocable,
            if namespace.required_artifacts.is_empty() {
                "none".to_owned()
            } else {
                namespace.required_artifacts.join("|")
            },
            if namespace.warmup_proof_ids.is_empty() {
                "none".to_owned()
            } else {
                namespace.warmup_proof_ids.join("|")
            },
            if namespace.replay_root_ids.is_empty() {
                "none".to_owned()
            } else {
                namespace.replay_root_ids.join("|")
            },
            namespace.source_surface_ids.join("|")
        ));
    }
    out
}

#[must_use]
pub fn render_diagnostics(document: &DiagnosticsDocument) -> String {
    let mut out = String::new();
    for diagnostic in &document.diagnostics {
        let severity = match diagnostic.severity {
            crate::diagnostics::DiagnosticSeverity::Info => "info",
            crate::diagnostics::DiagnosticSeverity::Warning => "warn",
            crate::diagnostics::DiagnosticSeverity::Error => "error",
        };
        out.push_str(&format!(
            "  - [{}] {}: {}\n",
            severity, diagnostic.code, diagnostic.likely_root_cause
        ));
    }
    out
}

#[must_use]
pub fn render_verification_report(report: &VerificationReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("verification {:?}\n", report.status));
    for issue in &report.issues {
        out.push_str(&format!(
            "  - {:?} {}: {}\n",
            issue.severity, issue.code, issue.message
        ));
    }
    if !report.runtime_jit_evidence.is_empty() {
        out.push_str("runtime-jit evidence:\n");
        for evidence in &report.runtime_jit_evidence {
            out.push_str(&format!(
                "  - {} backend={} bounded_by={} mitigation={}\n",
                evidence.surface_name,
                evidence.backend_family,
                evidence.bounded_by.join(","),
                evidence.mitigation
            ));
        }
    }
    if !report.operator_gates.is_empty() {
        out.push_str("operator gates:\n");
        for gate in &report.operator_gates {
            out.push_str(&format!(
                "  - {} compile_free={} forbidden_queues={}\n",
                gate.command,
                gate.compile_free,
                gate.forbidden_queues
                    .iter()
                    .map(|queue| format!("{queue:?}"))
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
    }
    out
}

impl std::fmt::Display for HazardClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
