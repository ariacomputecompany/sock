use serde::{Deserialize, Serialize};

use crate::{
    AcceleratorVendor, ArtifactAcquisition, ArtifactPortability, ArtifactRequirement,
    BackendFamily, BackendSelection, CanonicalError, CanonicalHash, CompileRegion,
    ExecutionTopology, OperatingSystem, RankDisposition, canonical_hash,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AbiFingerprint {
    pub operating_system: OperatingSystem,
    pub accelerator_vendor: AcceleratorVendor,
    pub gpu_arches: Vec<String>,
    pub cuda_version: String,
    pub driver_version: String,
    pub python_abi: String,
    pub libc_abi: String,
    pub topology: ExecutionTopology,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BackendExtensionFingerprint {
    pub primary_backend: BackendFamily,
    pub secondary_backends: Vec<BackendFamily>,
    pub compile_region_backends: Vec<(String, BackendFamily)>,
    pub compile_region_kinds: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ArtifactReuseBoundary {
    pub scope: String,
    pub backend: BackendFamily,
    pub portability: ArtifactPortability,
    pub rank_disposition: RankDisposition,
    pub acquisition: ArtifactAcquisition,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PortabilityFingerprint {
    pub cache_namespace: String,
    pub allow_cross_machine_reuse: bool,
    pub topology: ExecutionTopology,
    pub artifact_boundaries: Vec<ArtifactReuseBoundary>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuralIdentity {
    pub request_identity: CanonicalHash,
    pub optimization_identity: CanonicalHash,
    pub backend_decision_identity: CanonicalHash,
    pub backend_registry_identity: CanonicalHash,
    pub shape_envelope_identity: CanonicalHash,
    pub compile_region_identity: CanonicalHash,
    pub capability_identity: CanonicalHash,
    pub abi_identity: CanonicalHash,
    pub backend_extension_identity: CanonicalHash,
    pub portability_identity: CanonicalHash,
    pub artifact_identity: CanonicalHash,
    pub evidence_identity: CanonicalHash,
    pub plan_identity: CanonicalHash,
}

#[must_use]
pub fn artifact_manifest_identity(
    primary_backend: BackendFamily,
    requirement: &ArtifactRequirement,
) -> String {
    format!(
        "{}:{:?}:{}",
        primary_backend.as_str(),
        requirement.class,
        requirement.scope
    )
}

pub fn artifact_node_handle(requirement: &ArtifactRequirement) -> Result<String, CanonicalError> {
    Ok(format!(
        "artifact:{}:{}",
        requirement.class.as_str(),
        canonical_hash(requirement)?
    ))
}

pub fn fanout_node_handle(requirement: &ArtifactRequirement) -> Result<String, CanonicalError> {
    Ok(format!(
        "fanout:{}:{}",
        requirement.class.as_str(),
        canonical_hash(requirement)?
    ))
}

impl BackendExtensionFingerprint {
    #[must_use]
    pub fn from_plan(
        selected_backends: &BackendSelection,
        compile_regions: &[CompileRegion],
    ) -> Self {
        Self {
            primary_backend: selected_backends.primary.family,
            secondary_backends: selected_backends
                .secondary
                .iter()
                .map(|candidate| candidate.family)
                .collect(),
            compile_region_backends: compile_regions
                .iter()
                .map(|region| (region.name.clone(), region.family))
                .collect(),
            compile_region_kinds: compile_regions
                .iter()
                .map(|region| (region.name.clone(), format!("{:?}", region.kind)))
                .collect(),
        }
    }
}

impl PortabilityFingerprint {
    #[must_use]
    pub fn from_plan(
        cache_namespace: String,
        allow_cross_machine_reuse: bool,
        topology: ExecutionTopology,
        artifact_requirements: &[ArtifactRequirement],
    ) -> Self {
        Self {
            cache_namespace,
            allow_cross_machine_reuse,
            topology,
            artifact_boundaries: artifact_requirements
                .iter()
                .map(|requirement| ArtifactReuseBoundary {
                    scope: requirement.scope.clone(),
                    backend: requirement.backend,
                    portability: requirement.portability,
                    rank_disposition: requirement.rank_disposition,
                    acquisition: requirement.acquisition,
                })
                .collect(),
        }
    }
}
