use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use hex::ToHex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    ArtifactClosure, ArtifactManifestEntry, CanonicalError, CanonicalHash, HazardClass, PassTrace,
    ResolvedBuildPlan, ValidationSeverity, ValidationStatus, VerificationReport, canonical_json,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticsDocument {
    pub schema_version: SchemaVersion,
    pub plan_identity: CanonicalHash,
    pub diagnostics: Vec<StructuredDiagnostic>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactManifestDocument {
    pub schema_version: SchemaVersion,
    pub plan_identity: CanonicalHash,
    pub artifacts: Vec<ArtifactManifestEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RewriteTraceDocument {
    pub schema_version: SchemaVersion,
    pub plan_identity: CanonicalHash,
    pub passes: Vec<PassTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayBundleMetadata {
    pub schema_version: SchemaVersion,
    pub plan_identity: CanonicalHash,
    pub file_digests: BTreeMap<String, String>,
    pub replay_entrypoint: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayBundle {
    pub build_plan: ResolvedBuildPlan,
    pub artifact_closure: ArtifactClosure,
    pub verification_report: VerificationReport,
    pub diagnostics: DiagnosticsDocument,
    pub rewrite_trace: RewriteTraceDocument,
}

#[derive(Debug, Error)]
pub enum ReplayBundleError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("canonical error: {0}")]
    Canonical(#[from] CanonicalError),
    #[error("digest mismatch for {file}")]
    DigestMismatch { file: String },
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

impl ArtifactManifestDocument {
    #[must_use]
    pub fn from_closure(closure: &ArtifactClosure) -> Self {
        let mut artifacts = closure.artifacts.clone();
        artifacts.sort();
        Self {
            schema_version: SchemaVersion::current(),
            plan_identity: closure.plan_identity.clone(),
            artifacts,
        }
    }
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
                likely_root_cause: "Plan, artifact closure, and verification report are internally consistent."
                    .to_owned(),
                next_action: "Replay the bundle on an admissible host or promote it to a regression artifact."
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
                likely_root_cause: "The requested closure target exceeds current evidence.".to_owned(),
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

impl ReplayBundle {
    #[must_use]
    pub fn replay_script(&self) -> String {
        "#!/usr/bin/env sh\nset -eu\nDIR=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\nexec cargo run --quiet --bin sock -- replay --bundle \"$DIR\"\n".to_owned()
    }

    pub fn write_to(&self, dir: &Path) -> Result<ReplayBundleMetadata, ReplayBundleError> {
        fs::create_dir_all(dir)?;
        let artifact_manifest = ArtifactManifestDocument::from_closure(&self.artifact_closure);
        let files = [
            ("buildplan.json", canonical_json(&self.build_plan)?),
            (
                "artifact_manifest.json",
                canonical_json(&artifact_manifest)?,
            ),
            (
                "verification_report.json",
                canonical_json(&self.verification_report)?,
            ),
            ("diagnostics.json", canonical_json(&self.diagnostics)?),
            ("rewrite_trace.json", canonical_json(&self.rewrite_trace)?),
        ];
        let mut file_digests = BTreeMap::new();
        for (name, content) in files {
            fs::write(dir.join(name), content.as_bytes())?;
            file_digests.insert(name.to_owned(), digest(content.as_bytes()));
        }
        let replay_script = self.replay_script();
        fs::write(dir.join("replay.sh"), replay_script.as_bytes())?;
        file_digests.insert("replay.sh".to_owned(), digest(replay_script.as_bytes()));
        let metadata = ReplayBundleMetadata {
            schema_version: SchemaVersion::current(),
            plan_identity: self.build_plan.structural_identity.plan_identity.clone(),
            file_digests,
            replay_entrypoint: "./replay.sh".to_owned(),
        };
        fs::write(
            dir.join("bundle_metadata.json"),
            canonical_json(&metadata)?.as_bytes(),
        )?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = fs::metadata(dir.join("replay.sh"))?.permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(dir.join("replay.sh"), permissions)?;
        }
        Ok(metadata)
    }

    pub fn load_from(dir: &Path) -> Result<Self, ReplayBundleError> {
        let metadata: ReplayBundleMetadata =
            serde_json::from_str(&fs::read_to_string(dir.join("bundle_metadata.json"))?)?;
        for (file, expected) in &metadata.file_digests {
            let actual = digest(fs::read(dir.join(file))?.as_slice());
            if &actual != expected {
                return Err(ReplayBundleError::DigestMismatch { file: file.clone() });
            }
        }

        let build_plan: ResolvedBuildPlan =
            serde_json::from_str(&fs::read_to_string(dir.join("buildplan.json"))?)?;
        let artifact_manifest: ArtifactManifestDocument =
            serde_json::from_str(&fs::read_to_string(dir.join("artifact_manifest.json"))?)?;
        let verification_report: VerificationReport =
            serde_json::from_str(&fs::read_to_string(dir.join("verification_report.json"))?)?;
        let diagnostics: DiagnosticsDocument =
            serde_json::from_str(&fs::read_to_string(dir.join("diagnostics.json"))?)?;
        let rewrite_trace: RewriteTraceDocument =
            serde_json::from_str(&fs::read_to_string(dir.join("rewrite_trace.json"))?)?;

        Ok(Self {
            build_plan,
            artifact_closure: ArtifactClosure {
                plan_identity: artifact_manifest.plan_identity.clone(),
                artifacts: artifact_manifest.artifacts,
            },
            verification_report,
            diagnostics,
            rewrite_trace,
        })
    }
}

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
            pass.pass_name, pass.before_identity, pass.after_identity
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
            DiagnosticSeverity::Info => "info",
            DiagnosticSeverity::Warning => "warn",
            DiagnosticSeverity::Error => "error",
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
    out
}

fn categorize_issue(message: &str) -> DiagnosticCategory {
    if message.contains("FlashInfer") {
        DiagnosticCategory::Environment
    } else if message.contains("Warmup obligations") {
        DiagnosticCategory::Closure
    } else {
        DiagnosticCategory::OperatorDx
    }
}

fn next_action(message: &str) -> String {
    if message.contains("Warmup obligations") {
        "Add explicit warmup coverage for the missing envelope nodes.".to_owned()
    } else if message.contains("FlashInfer") {
        "Provide the missing capability witness or stop selecting FlashInfer.".to_owned()
    } else {
        "Inspect the replay bundle and rerun verification on an admissible environment.".to_owned()
    }
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes).encode_hex::<String>()
}

impl std::fmt::Display for HazardClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
