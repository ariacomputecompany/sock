use serde::{Deserialize, Serialize};

use crate::{
    AcceleratorVendor, ArtifactAcquisition, ArtifactClass, ArtifactPortability, CanonicalHash,
    ExecutionTopology, OperatingSystem, SchemaVersion,
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BackendDecisionEntry {
    pub family: crate::BackendFamily,
    pub technically_available: bool,
    pub selected_for_deployment: bool,
    pub reachable_from_model_family: bool,
    pub reachable_from_materialization_plan: bool,
    pub runtime_reachable: bool,
    pub build_technically_possible: bool,
    pub chosen_acquisition: Option<ArtifactAcquisition>,
    pub required_witnesses: Vec<String>,
    pub satisfied_witnesses: Vec<String>,
    pub accepted_reasons: Vec<String>,
    pub rejected_reasons: Vec<String>,
    pub reachable_compile_regions: Vec<String>,
    pub reachable_artifact_scopes: Vec<String>,
    pub reachable_warmup_scopes: Vec<String>,
    pub pass_through_optimizations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BackendExtensionManifest {
    pub extension_key: String,
    pub binary_name: String,
    pub backend_family: crate::BackendFamily,
    pub model_repositories: Vec<String>,
    pub build_technically_possible: bool,
    pub runtime_reachable: bool,
    pub reachable_compile_regions: Vec<String>,
    pub reachable_artifact_scopes: Vec<String>,
    pub artifact_classes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendDecisionPlan {
    pub build_profile_identity: CanonicalHash,
    pub entries: Vec<BackendDecisionEntry>,
    pub extension_manifests: Vec<BackendExtensionManifest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendDecisionDocument {
    pub schema_version: SchemaVersion,
    pub plan_identity: CanonicalHash,
    pub build_profile_identity: CanonicalHash,
    pub entries: Vec<BackendDecisionEntry>,
    pub extension_manifests: Vec<BackendExtensionManifest>,
}

impl BackendDecisionDocument {
    #[must_use]
    pub fn from_plan(plan: &crate::ResolvedBuildPlan) -> Self {
        Self {
            schema_version: SchemaVersion::current(),
            plan_identity: plan.structural_identity.plan_identity.clone(),
            build_profile_identity: plan.backend_decision.build_profile_identity.clone(),
            entries: plan.backend_decision.entries.clone(),
            extension_manifests: plan.backend_decision.extension_manifests.clone(),
        }
    }
}
