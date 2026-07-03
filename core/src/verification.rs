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
    pub runtime_jit_evidence: Vec<RuntimeJitEvidence>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RuntimeJitEvidence {
    pub surface_name: String,
    pub backend_family: String,
    pub trigger_shape_or_config: String,
    pub trigger_inputs: Vec<String>,
    pub affected_regions: Vec<String>,
    pub required_artifacts: Vec<String>,
    pub declared_required_warmup_scopes: Vec<String>,
    pub required_warmup_proofs: Vec<String>,
    pub topology_context: String,
    pub bounded_by: Vec<String>,
    pub mitigation: String,
    pub contradiction_reasons: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct OperatorGate {
    pub command: String,
    pub compile_free: bool,
    pub forbidden_queues: Vec<QueueKind>,
    pub rationale: String,
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
    pub runtime_jit_evidence: Vec<RuntimeJitEvidence>,
    pub operator_gates: Vec<OperatorGate>,
}
