use serde::{Deserialize, Serialize};

use crate::{CoveragePlane, QueueKind, ValidationLevel, ValidationStatus};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CoverageWitness {
    pub plane: CoveragePlane,
    pub node_name: String,
    pub evidence: String,
    pub coverage_states: Vec<crate::CoverageState>,
    pub artifact_scopes: Vec<String>,
    pub uncovered_residuals: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuaranteeEvidence {
    pub capability_witnesses: Vec<crate::CapabilityWitness>,
    pub artifact_manifest: Vec<crate::ArtifactManifestEntry>,
    pub warmup_obligations: Vec<crate::WarmupObligation>,
    pub coverage_witnesses: Vec<CoverageWitness>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub severity: crate::ValidationSeverity,
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationReport {
    pub level: ValidationLevel,
    pub status: ValidationStatus,
    pub issues: Vec<ValidationIssue>,
    pub phase_timings: Vec<(QueueKind, Option<u64>)>,
    pub runtime_jit_witnesses: Vec<String>,
}
