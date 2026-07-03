use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{ArtifactPortability, BackendFamily, CoveragePlane, RankDisposition, TargetEngine};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigInputSource {
    CliFlag,
    PythonApi,
    EnvironmentVariable,
    RuntimeState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    Info,
    Warning,
    Risk,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JitRiskLevel {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompileRegionKind {
    RepeatedTransformerBlockBody,
    DecodeMicrograph,
    PrefillMicrograph,
    AttentionKvBoundary,
    MoeSpecialtyPath,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SourceAnchor {
    pub file: String,
    pub line: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SourceEvidence {
    pub summary: String,
    pub anchors: Vec<SourceAnchor>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectiveConfigInput {
    pub name: String,
    pub source: ConfigInputSource,
    pub compile_relevance: String,
    pub identity_affecting: bool,
    pub evidence: SourceEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompileAffectingKnob {
    pub name: String,
    pub category: String,
    pub description: String,
    pub evidence: SourceEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreservedEngineAbstraction {
    pub name: String,
    pub description: String,
    pub evidence: SourceEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterBackendBinding {
    Primary,
    Fixed(BackendFamily),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegionCacheSharing {
    NamespaceLocal,
    ContentAddressed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterCompileRegion {
    pub name: String,
    pub canonical_name: String,
    pub kind: CompileRegionKind,
    pub backend_binding: AdapterBackendBinding,
    pub repeated: bool,
    pub regional_compile_candidate: bool,
    pub boundaries: Vec<String>,
    pub rationale: String,
    pub invalidation_domain: String,
    pub shape_planes: Vec<CoveragePlane>,
    pub artifact_portability: ArtifactPortability,
    pub rank_disposition: RankDisposition,
    pub topology_sensitive: bool,
    pub cache_namespace: String,
    pub warmup_scope: String,
    pub cache_sharing: RegionCacheSharing,
    pub portability_scope: String,
    pub topology_scope: String,
    pub closure_verification_criteria: Vec<String>,
    pub evidence: SourceEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidualRuntimeJitSurface {
    pub name: String,
    pub risk: JitRiskLevel,
    pub trigger_shape_or_config: String,
    pub backend_family: String,
    pub topology_context: String,
    pub warmup_gap: String,
    pub mitigation: String,
    pub trigger_inputs: Vec<String>,
    pub affected_regions: Vec<String>,
    pub required_artifacts: Vec<String>,
    pub required_warmup_scopes: Vec<String>,
    pub topology_sensitive: bool,
    pub evidence: SourceEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheOwnershipSurface {
    pub name: String,
    pub artifact_scopes: Vec<String>,
    pub ownership_inputs: Vec<String>,
    pub portability: ArtifactPortability,
    pub rank_disposition: RankDisposition,
    pub topology_sensitive: bool,
    pub rationale: String,
    pub evidence: SourceEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterDiagnostic {
    pub severity: DiagnosticSeverity,
    pub title: String,
    pub message: String,
    pub evidence: SourceEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterHook {
    DiscoverEngineSurface,
    ExtractEffectiveConfig,
    EnumerateExecutionPaths,
    EnumerateMaterializableArtifacts,
    ResolveBackendOptions,
    BuildWarmupCoverage,
    ObserveRuntimeMaterialization,
    VerifyClosureClaims,
    RenderExplain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterBoundary {
    CompileGraphBoundary,
    CompileRegionBoundary,
    CustomOpBoundary,
    CacheOwnershipBoundary,
    CUDAGraphBoundary,
    WarmupBoundary,
    TopologyBoundary,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineAdapterContract {
    pub hooks: Vec<AdapterHook>,
    pub boundaries: Vec<AdapterBoundary>,
    pub guarantee_limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterSurvey {
    pub engine: TargetEngine,
    pub engine_root: String,
    pub engine_revision: String,
    pub contract: EngineAdapterContract,
    pub config_inputs: Vec<EffectiveConfigInput>,
    pub compile_knobs: Vec<CompileAffectingKnob>,
    pub preserved_abstractions: Vec<PreservedEngineAbstraction>,
    pub compile_regions: Vec<AdapterCompileRegion>,
    pub cache_ownership_surfaces: Vec<CacheOwnershipSurface>,
    pub residual_jit_surfaces: Vec<ResidualRuntimeJitSurface>,
    pub diagnostics: Vec<AdapterDiagnostic>,
}

#[derive(Debug, Error)]
pub enum AdapterError {
    #[error("failed to read source file `{file}`: {source}")]
    ReadSource {
        file: String,
        source: std::io::Error,
    },
    #[error("missing pattern `{pattern}` in `{file}`")]
    MissingPattern { file: String, pattern: String },
}

pub type AdapterResult<T> = Result<T, AdapterError>;

pub trait EngineAdapter {
    fn target(&self) -> TargetEngine;
    fn survey(&self) -> AdapterResult<AdapterSurvey>;
    fn render_explain(&self, survey: &AdapterSurvey) -> String;
}
