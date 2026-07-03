use serde::{Deserialize, Serialize};

use crate::{
    ArtifactAcquisition, ArtifactClass, BackendFamily, CanonicalHash, MaterializationNodeKind,
    QueueKind, SchemaVersion, SourceAnchor,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterializationDisposition {
    Executed,
    Reused,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterializedArtifactRecord {
    pub storage_key: String,
    pub manifest_identity: String,
    pub scope: String,
    pub class: ArtifactClass,
    pub backend: BackendFamily,
    pub acquisition: ArtifactAcquisition,
    pub disposition: MaterializationDisposition,
    pub relative_path: String,
    pub bytes_written: u64,
    pub compile_ms: u64,
    pub transfer_ms: u64,
    pub source_anchors: Vec<SourceAnchor>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterializationNodeRecord {
    pub node_name: String,
    pub wave: u32,
    pub kind: MaterializationNodeKind,
    pub queue: QueueKind,
    pub disposition: MaterializationDisposition,
    pub dependency_nodes: Vec<String>,
    pub outputs: Vec<String>,
    pub relative_path: String,
    pub duration_ms: u64,
    pub bytes_written: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterializationWaveRecord {
    pub wave_name: String,
    pub queue: QueueKind,
    pub node_names: Vec<String>,
    pub relative_path: String,
    pub duration_ms: u64,
    pub bytes_written: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterializationExecutionReport {
    pub schema_version: SchemaVersion,
    pub plan_identity: CanonicalHash,
    pub artifact_root: String,
    pub node_root: String,
    pub wave_root: String,
    pub artifact_count: u32,
    pub reused_artifact_count: u32,
    pub total_bytes_written: u64,
    pub total_compile_ms: u64,
    pub total_transfer_ms: u64,
    pub artifacts: Vec<MaterializedArtifactRecord>,
    pub nodes: Vec<MaterializationNodeRecord>,
    pub waves: Vec<MaterializationWaveRecord>,
}
