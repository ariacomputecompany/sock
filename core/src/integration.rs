use serde::{Deserialize, Serialize};

use crate::{BackendFamily, CanonicalHash, SchemaVersion, SourceEvidence};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntegrationScopeKind {
    CompileRegion,
    CacheSurface,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VllmCallableTarget {
    pub module: String,
    pub callable: String,
    pub summary: String,
    pub evidence: SourceEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VllmIsolationDisposition {
    Standalone,
    ContextBound,
    NonIsolatable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VllmIsolationContract {
    pub disposition: VllmIsolationDisposition,
    pub subset_build_valid: bool,
    pub direct_entrypoint_invocable: bool,
    pub required_context: Vec<String>,
    pub blockers: Vec<String>,
    pub evidence: SourceEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VllmIntegrationSurface {
    pub id: String,
    pub scope_kind: IntegrationScopeKind,
    pub scope_name: String,
    pub backend: Option<BackendFamily>,
    pub cache_namespace: Option<String>,
    pub warmup_scope: Option<String>,
    pub rationale: String,
    pub preserved_inputs: Vec<String>,
    pub preserved_abstractions: Vec<String>,
    pub isolation: VllmIsolationContract,
    pub primary: VllmCallableTarget,
    pub auxiliary: Vec<VllmCallableTarget>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VllmReplayRootKind {
    CompileRegion,
    CacheSurface,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VllmReplayRoot {
    pub id: String,
    pub root_kind: VllmReplayRootKind,
    pub surface_id: String,
    pub scope_name: String,
    pub root_key: CanonicalHash,
    pub cache_namespace: Option<String>,
    pub warmup_scope: Option<String>,
    pub replay_boundary: String,
    pub manifest_paths: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VllmIntegrationDocument {
    pub schema_version: SchemaVersion,
    pub plan_identity: CanonicalHash,
    pub engine_root: String,
    pub engine_revision: String,
    pub surfaces: Vec<VllmIntegrationSurface>,
    pub replay_roots: Vec<VllmReplayRoot>,
}
