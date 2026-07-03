use serde::{Deserialize, Serialize};

use crate::{
    CanonicalHash, HazardClass, PassTrace, ResolvedBuildPlan, ValidationSeverity, ValidationStatus,
    VerificationReport,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticsDocument {
    pub schema_version: SchemaVersion,
    pub plan_identity: CanonicalHash,
    pub diagnostics: Vec<StructuredDiagnostic>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RewriteTraceDocument {
    pub schema_version: SchemaVersion,
    pub plan_identity: CanonicalHash,
    pub passes: Vec<PassTrace>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaVersion {
    pub major: u16,
    pub minor: u16,
}

impl SchemaVersion {
    pub const CURRENT: Self = Self { major: 1, minor: 0 };

    #[must_use]
    pub const fn current() -> Self {
        Self::CURRENT
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Ord, PartialOrd)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticCategory {
    Environment,
    Closure,
    Replay,
    OperatorDx,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticConfidence {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticEvidence {
    pub key: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuredDiagnostic {
    pub code: String,
    pub category: DiagnosticCategory,
    pub severity: DiagnosticSeverity,
    pub evidence: Vec<DiagnosticEvidence>,
    pub confidence: DiagnosticConfidence,
    pub likely_root_cause: String,
    pub next_action: String,
    pub auto_fix_allowed: bool,
}

impl DiagnosticsDocument {
    #[must_use]
    pub fn from_outcome(
        plan: &ResolvedBuildPlan,
        verification_report: &VerificationReport,
        trace: &[PassTrace],
    ) -> Self {
        let mut diagnostics = verification_report
            .issues
            .iter()
            .map(|issue| StructuredDiagnostic {
                code: issue.code.clone(),
                category: categorize_issue(issue.message.as_str()),
                severity: match issue.severity {
                    ValidationSeverity::Error => DiagnosticSeverity::Error,
                    ValidationSeverity::Warning => DiagnosticSeverity::Warning,
                },
                evidence: vec![DiagnosticEvidence {
                    key: "validation".to_owned(),
                    value: issue.message.clone(),
                }],
                confidence: DiagnosticConfidence::High,
                likely_root_cause: issue.message.clone(),
                next_action: next_action(issue.message.as_str()),
                auto_fix_allowed: false,
            })
            .collect::<Vec<_>>();

        diagnostics.push(match verification_report.status {
            ValidationStatus::Passed => StructuredDiagnostic {
                code: "verified_bundle".to_owned(),
                category: DiagnosticCategory::OperatorDx,
                severity: DiagnosticSeverity::Info,
                evidence: vec![DiagnosticEvidence {
                    key: "rewrite_passes".to_owned(),
                    value: trace.len().to_string(),
                }],
                confidence: DiagnosticConfidence::High,
                likely_root_cause:
                    "Plan, artifact closure, and verification report are internally consistent."
                        .to_owned(),
                next_action:
                    "Replay the bundle on an admissible host or promote it to a regression artifact."
                        .to_owned(),
                auto_fix_allowed: false,
            },
            ValidationStatus::Failed => StructuredDiagnostic {
                code: "verification_failed".to_owned(),
                category: DiagnosticCategory::Replay,
                severity: DiagnosticSeverity::Error,
                evidence: vec![DiagnosticEvidence {
                    key: "plan_identity".to_owned(),
                    value: plan.structural_identity.plan_identity.to_string(),
                }],
                confidence: DiagnosticConfidence::High,
                likely_root_cause: "The requested closure target exceeds current evidence."
                    .to_owned(),
                next_action: "Inspect `sock explain` and rebuild with supported envelopes only."
                    .to_owned(),
                auto_fix_allowed: false,
            },
        });

        if plan
            .guarantee_envelope
            .residual_risks
            .iter()
            .any(|risk| risk.class == HazardClass::ResidualLazyCompile)
        {
            diagnostics.push(StructuredDiagnostic {
                code: "residual_runtime_compile_risk".to_owned(),
                category: DiagnosticCategory::Closure,
                severity: DiagnosticSeverity::Warning,
                evidence: plan
                    .guarantee_envelope
                    .residual_risks
                    .iter()
                    .map(|risk| DiagnosticEvidence {
                        key: risk.class.to_string(),
                        value: risk.summary.clone(),
                    })
                    .collect(),
                confidence: DiagnosticConfidence::High,
                likely_root_cause: "Residual lazy compile risk remains bounded but unresolved."
                    .to_owned(),
                next_action:
                    "Expand warmup obligations before claiming fail-closed runtime behavior."
                        .to_owned(),
                auto_fix_allowed: false,
            });
        }

        diagnostics.sort_by(|left, right| {
            right
                .severity
                .cmp(&left.severity)
                .then(left.code.cmp(&right.code))
        });

        Self {
            schema_version: SchemaVersion::current(),
            plan_identity: plan.structural_identity.plan_identity.clone(),
            diagnostics,
        }
    }
}

impl RewriteTraceDocument {
    #[must_use]
    pub fn new(plan: &ResolvedBuildPlan, passes: Vec<PassTrace>) -> Self {
        Self {
            schema_version: SchemaVersion::current(),
            plan_identity: plan.structural_identity.plan_identity.clone(),
            passes,
        }
    }
}

fn categorize_issue(message: &str) -> DiagnosticCategory {
    if message.contains("FlashInfer") {
        DiagnosticCategory::Environment
    } else if message.contains("runtime JIT")
        || message.contains("Runtime-JIT")
        || message.contains("artifact requirements")
    {
        DiagnosticCategory::Closure
    } else if message.contains("Warmup obligations") {
        DiagnosticCategory::Closure
    } else {
        DiagnosticCategory::OperatorDx
    }
}

fn next_action(message: &str) -> String {
    if message.contains("Warmup obligations") {
        "Add explicit warmup coverage for the missing envelope nodes.".to_owned()
    } else if message.contains("artifact requirements") {
        "Rebuild the bundle so the artifact manifest matches the canonical plan exactly.".to_owned()
    } else if message.contains("Runtime-JIT") || message.contains("runtime JIT") {
        "Bound the residual runtime-JIT surface with explicit artifacts, warmup proofs, or backend identity before shipping."
            .to_owned()
    } else if message.contains("FlashInfer") {
        "Provide the missing capability witness or stop selecting FlashInfer.".to_owned()
    } else {
        "Inspect the replay bundle and rerun verification on an admissible environment.".to_owned()
    }
}
