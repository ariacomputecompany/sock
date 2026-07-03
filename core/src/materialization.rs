use serde::{Deserialize, Serialize};

use crate::{
    ArtifactAcquisition, ArtifactClass, BackendFamily, CanonicalHash, FanoutStrategy,
    MaterializationNodeKind, QueueDiscipline, QueueKind, RankDisposition, RegionCacheSharing,
    SchemaVersion, SourceAnchor, ValidationStatus,
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
    pub region_stable_identity: Option<CanonicalHash>,
    pub region_equivalence_identity: Option<CanonicalHash>,
    pub cache_sharing: Option<RegionCacheSharing>,
    pub cache_namespace: String,
    pub invalidation_domain: String,
    pub acquisition: ArtifactAcquisition,
    pub rank_disposition: RankDisposition,
    pub preferred_fanout_strategy: FanoutStrategy,
    pub disposition: MaterializationDisposition,
    pub relative_path: String,
    pub cache_relative_path: String,
    pub bytes_written: u64,
    pub deserialization_ms: u64,
    pub rank_count: u16,
    pub compile_ms: u64,
    pub transfer_ms: u64,
    pub rebuild_ms: u64,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterializationSchedulingMode {
    Sequential,
    Parallel,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterializationWaveRecord {
    pub wave_name: String,
    pub queue: QueueKind,
    pub discipline: QueueDiscipline,
    pub scheduling_mode: MaterializationSchedulingMode,
    pub max_parallelism: u16,
    pub node_names: Vec<String>,
    pub relative_path: String,
    pub duration_ms: u64,
    pub bytes_written: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClosureExpansionRecord {
    pub requested_regions: Vec<String>,
    pub requested_artifact_scopes: Vec<String>,
    pub requested_backend_families: Vec<String>,
    pub requested_cache_namespaces: Vec<String>,
    pub requested_warmup_scopes: Vec<String>,
    pub expanded_regions: Vec<String>,
    pub expanded_artifact_scopes: Vec<String>,
    pub expanded_warmup_scopes: Vec<String>,
    pub deterministically_closed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservedReadinessLevel {
    EarlyServe,
    Correctness,
    Performance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadinessObservation {
    pub requested_readiness: ObservedReadinessLevel,
    pub achieved_readiness: ObservedReadinessLevel,
    pub blocking_warmups_complete: bool,
    pub early_serve_frontier_complete: bool,
    pub deferred_warmups_complete: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeJitObservationStatus {
    Bounded,
    Contradicted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StartupClosureOutcome {
    FullCompileClosure,
    PartialCompileClosure,
    ClosureByAssumption,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeJitObservation {
    pub surface_name: String,
    pub status: RuntimeJitObservationStatus,
    pub observed_artifacts: Vec<String>,
    pub observed_warmup_proofs: Vec<String>,
    pub contradiction_reasons: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterializationExecutionReport {
    pub schema_version: SchemaVersion,
    pub plan_identity: CanonicalHash,
    pub artifact_root: String,
    pub cache_root: String,
    pub node_root: String,
    pub wave_root: String,
    pub artifact_count: u32,
    pub executed_artifact_count: u32,
    pub reused_artifact_count: u32,
    pub wall_clock_ms: u64,
    pub total_bytes_written: u64,
    pub total_compile_ms: u64,
    pub total_transfer_ms: u64,
    pub total_rebuild_ms: u64,
    pub unique_artifact_count: u32,
    pub duplicate_artifact_count: u32,
    pub unique_artifact_bytes: u64,
    pub duplicate_artifact_bytes: u64,
    pub artifact_deserialization_ms: u64,
    pub duplicate_rank_local_compile_count: u32,
    pub duplicate_rank_local_load_count: u32,
    pub closure_expansion: ClosureExpansionRecord,
    pub closure_outcome: StartupClosureOutcome,
    pub readiness: ReadinessObservation,
    pub runtime_jit_observations: Vec<RuntimeJitObservation>,
    pub verify_replay_compile_free: bool,
    pub verify_replay_status: ValidationStatus,
    pub artifacts: Vec<MaterializedArtifactRecord>,
    pub nodes: Vec<MaterializationNodeRecord>,
    pub waves: Vec<MaterializationWaveRecord>,
}
