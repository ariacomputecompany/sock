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
    pub primary: VllmCallableTarget,
    pub auxiliary: Vec<VllmCallableTarget>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VllmIntegrationDocument {
    pub schema_version: SchemaVersion,
    pub plan_identity: CanonicalHash,
    pub engine_root: String,
    pub engine_revision: String,
    pub surfaces: Vec<VllmIntegrationSurface>,
}
