use crate::{
    DiagnosticsDocument, HazardClass, ResolvedBuildPlan, RewriteTraceDocument, VerificationReport,
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
) -> String {
    let mut out = String::new();
    out.push_str(&render_plan_summary(plan));
    out.push_str(&format!(
        "guarantee correctness {:?} performance {:?}\n",
        plan.guarantee_envelope.achieved_correctness, plan.guarantee_envelope.achieved_performance
    ));
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
