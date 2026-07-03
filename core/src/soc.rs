use serde::{Deserialize, Serialize};

use crate::{CanonicalHash, IntegrationScopeKind, SchemaVersion};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SocMaterializationMode {
    EagerBlocking,
    EagerDeferred,
    Lazy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SocSelectorSnapshot {
    pub requested_regions: Vec<String>,
    pub requested_artifact_scopes: Vec<String>,
    pub requested_backend_families: Vec<String>,
    pub requested_topology_scopes: Vec<String>,
    pub requested_cache_namespaces: Vec<String>,
    pub requested_warmup_scopes: Vec<String>,
    pub requested_readiness: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SocNamespacePlan {
    pub namespace: String,
    pub scope_kind: IntegrationScopeKind,
    pub materialization_mode: SocMaterializationMode,
    pub subset_build_valid: bool,
    pub direct_entrypoint_invocable: bool,
    pub artifact_scopes: Vec<String>,
    pub artifact_classes: Vec<String>,
    pub required_artifacts: Vec<String>,
    pub warmup_scopes: Vec<String>,
    pub warmup_proof_ids: Vec<String>,
    pub replay_root_ids: Vec<String>,
    pub source_surface_ids: Vec<String>,
    pub source_callables: Vec<String>,
    pub rationale: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SocPlanDocument {
    pub schema_version: SchemaVersion,
    pub plan_identity: CanonicalHash,
    pub derivation_strategy: String,
    pub selectors: SocSelectorSnapshot,
    pub namespaces: Vec<SocNamespacePlan>,
    pub replay_root_ids: Vec<String>,
    pub shared_abstractions: Vec<String>,
}
