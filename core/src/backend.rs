use serde::{Deserialize, Serialize};

use crate::{
    AcceleratorVendor, ArtifactAcquisition, ArtifactClass, ArtifactPortability, CanonicalHash,
    ExecutionTopology, OperatingSystem,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PackagingStrategy {
    PrebuiltOnly,
    PreferPrebuiltThenAot,
    PreferPrebuiltThenAotThenJit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RuntimeJitDisposition {
    Forbidden,
    ShapeBounded,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RuntimeJitPolicy {
    pub disposition: RuntimeJitDisposition,
    pub max_residual_node_count: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CapabilityProvenance {
    pub source: String,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BackendCapability {
    pub family: crate::BackendFamily,
    pub supported_operating_systems: Vec<OperatingSystem>,
    pub supported_accelerator_vendors: Vec<AcceleratorVendor>,
    pub allowed_acquisitions: Vec<ArtifactAcquisition>,
    pub required_witnesses: Vec<String>,
    pub legal_portability: Vec<ArtifactPortability>,
    pub provenance: Vec<CapabilityProvenance>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendCapabilityRegistry {
    pub entries: Vec<BackendCapability>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AdmissibilityVerdict {
    Admissible,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BackendAdmissibilityProof {
    pub verdict: AdmissibilityVerdict,
    pub family: crate::BackendFamily,
    pub acquisition: ArtifactAcquisition,
    pub packaging_strategy: PackagingStrategy,
    pub required_witnesses: Vec<String>,
    pub satisfied_witnesses: Vec<String>,
    pub rejected_reasons: Vec<String>,
    pub provenance: Vec<CapabilityProvenance>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ArtifactAdmissibilityProof {
    pub proof_identity: CanonicalHash,
    pub artifact_scope: String,
    pub class: ArtifactClass,
    pub backend: crate::BackendFamily,
    pub acquisition: ArtifactAcquisition,
    pub portability: ArtifactPortability,
    pub target_abi_identity: CanonicalHash,
    pub target_shape_envelope_identity: CanonicalHash,
    pub target_topology: ExecutionTopology,
    pub required_witnesses: Vec<String>,
    pub satisfied_witnesses: Vec<String>,
    pub fail_closed: bool,
    pub rationale: Vec<String>,
}
