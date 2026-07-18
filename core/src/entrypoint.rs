use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{CanonicalHash, SchemaVersion, VllmCallableTarget, VllmIsolationContract};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VllmContextKind {
    None,
    Worker,
    PiecewiseBackend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VllmCallStrategy {
    ModuleFunction,
    ModuleFunctionWithContext,
    ContextMethod,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VllmEntrypoint {
    pub schema_version: SchemaVersion,
    pub plan_identity: CanonicalHash,
    pub engine_root: String,
    pub engine_revision: String,
    pub id: String,
    pub surface_id: String,
    pub scope_name: String,
    pub isolation: VllmIsolationContract,
    pub context_kind: VllmContextKind,
    pub call_strategy: VllmCallStrategy,
    pub callable: VllmCallableTarget,
    pub args: BTreeMap<String, Value>,
    pub required_env: Vec<String>,
    pub preserved_inputs: Vec<String>,
    pub preserved_abstractions: Vec<String>,
    pub summary: String,
    pub manifest_path: String,
    pub wrapper_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VllmEntrypointDocument {
    pub schema_version: SchemaVersion,
    pub plan_identity: CanonicalHash,
    pub engine_root: String,
    pub engine_revision: String,
    pub entrypoints: Vec<VllmEntrypoint>,
}
