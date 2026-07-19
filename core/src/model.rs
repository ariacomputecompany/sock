use std::cmp::Ordering;
use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::CanonicalHash;
use crate::adapter::{CompileRegionKind, RegionCacheSharing, SourceEvidence};
use crate::backend::{
    ArtifactAdmissibilityProof, BackendAdmissibilityProof, BackendCapabilityRegistry,
    BackendDecisionPlan,
};
use crate::identity::StructuralIdentity;
use crate::request::{GuaranteeTarget, NormalizedRequest};
use crate::rewrite::PassTrace;
use crate::runtime::{
    NodeExecutionContract, RuntimeRoi, WarmupCoverageProof, WaveExecutionContract,
};
use crate::verification::{GuaranteeEvidence, OperatorGate, ValidationIssue, VerificationReport};
use crate::{OptimizationEnvelope, artifact_manifest_identity};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TargetEngine {
    Vllm,
}

impl TargetEngine {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Vllm => "vllm",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OperatingSystem {
    Linux,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AcceleratorVendor {
    Nvidia,
    Amd,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BackendFamily {
    FlashInfer,
    Triton,
    AotInductor,
    CudaGraphs,
}

impl BackendFamily {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FlashInfer => "flashinfer",
            Self::Triton => "triton",
            Self::AotInductor => "aot-inductor",
            Self::CudaGraphs => "cuda-graphs",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ArtifactClass {
    CompiledGraph,
    TritonBinary,
    ExtensionBinary,
    BackendPackageInput,
    AutotuneResult,
    CudaGraphCapture,
    TopologyScopedCache,
}

impl ArtifactClass {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CompiledGraph => "compiled-graph",
            Self::TritonBinary => "triton-binary",
            Self::ExtensionBinary => "extension-binary",
            Self::BackendPackageInput => "backend-package-input",
            Self::AutotuneResult => "autotune-result",
            Self::CudaGraphCapture => "cuda-graph-capture",
            Self::TopologyScopedCache => "topology-scoped-cache",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HazardClass {
    StaleCache,
    BackendLegality,
    ResidualLazyCompile,
    TopologyMismatch,
    ReplayInsufficiency,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CoveragePlane {
    Correctness,
    Performance,
    CudaGraph,
    BackendSpecialization,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RangeIntent {
    ExactHotShape,
    SymbolicRange,
    FallbackRange,
    UncoveredResidual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ArtifactAcquisition {
    VendorPrebuilt,
    UpstreamCacheBundle,
    LocalAotBuild,
    LocalSourceBuild,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QueueKind {
    Compile,
    Assemble,
    ArtifactIo,
    Warmup,
    Verify,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MaterializationNodeKind {
    Compile,
    Assemble,
    Materialize,
    Transfer,
    Warmup,
    Verify,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CoverageState {
    Compiled,
    Executed,
    Captured,
    Autotuned,
    VerifiedNoNewCompile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ArtifactPortability {
    HostLocalOnly,
    AbiClusterPortable,
    GpuArchitectureFamilyPortable,
    TopologyScoped,
    ShapeEnvelopeScoped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RankDisposition {
    Shared,
    RankLocal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GuaranteeDimension {
    Environment,
    Kernel,
    Shape,
    Runtime,
    Topology,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GuaranteeLevel {
    Advisory,
    WarmupBounded,
    ShapeBoundedAot,
    StrictNoSurpriseJit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FailureMode {
    FailClosed,
    FailOpen,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ValidationLevel {
    Structural,
    Semantic,
    WitnessBacked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ValidationSeverity {
    Error,
    Warning,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ValidationStatus {
    Passed,
    Failed,
}

impl GuaranteeLevel {
    #[must_use]
    pub fn rank(self) -> u8 {
        match self {
            Self::Advisory => 0,
            Self::WarmupBounded => 1,
            Self::ShapeBoundedAot => 2,
            Self::StrictNoSurpriseJit => 3,
        }
    }
}

impl PartialOrd for GuaranteeLevel {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GuaranteeLevel {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank().cmp(&other.rank())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CapabilityWitness {
    pub key: String,
    pub value: String,
    pub provenance: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BackendCandidate {
    pub family: BackendFamily,
    pub acquisition: ArtifactAcquisition,
    pub reason: String,
    pub admissibility: BackendAdmissibilityProof,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendSelection {
    pub primary: BackendCandidate,
    pub secondary: Vec<BackendCandidate>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CompileRegion {
    pub name: String,
    pub kind: CompileRegionKind,
    pub family: BackendFamily,
    pub reusable: bool,
    pub regional_compile_candidate: bool,
    pub boundaries: Vec<String>,
    pub rationale: String,
    pub invalidation_domain: String,
    pub shape_planes: Vec<CoveragePlane>,
    pub stable_identity: CanonicalHash,
    pub equivalence_identity: CanonicalHash,
    pub cache_namespace: String,
    pub cache_sharing: RegionCacheSharing,
    pub portability: ArtifactPortability,
    pub rank_disposition: RankDisposition,
    pub topology_sensitive: bool,
    pub portability_scope: String,
    pub topology_scope: String,
    pub warmup_scope: String,
    pub closure_verification_criteria: Vec<String>,
    pub evidence: SourceEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ShapeEnvelopeNode {
    pub name: String,
    pub plane: CoveragePlane,
    pub intent: RangeIntent,
    pub range: crate::ShapeRange,
    pub exact_shape: Option<crate::ShapePoint>,
    pub required_backends: Vec<BackendFamily>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShapeEnvelope {
    pub nodes: Vec<ShapeEnvelopeNode>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WarmupObligation {
    pub node_name: String,
    pub region_name: String,
    pub step_count: u32,
    pub plane: CoveragePlane,
    pub blocking: bool,
    pub required_artifacts: Vec<String>,
    pub rank_scope: Vec<u16>,
    pub requires_capture: bool,
    pub requires_autotune: bool,
    pub proof: WarmupCoverageProof,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ArtifactRequirement {
    pub class: ArtifactClass,
    pub backend: BackendFamily,
    pub acquisition: ArtifactAcquisition,
    pub scope: String,
    pub portability: ArtifactPortability,
    pub rank_disposition: RankDisposition,
    pub expected_bytes: Option<u64>,
    pub expected_compile_ms: Option<u64>,
    pub expected_transfer_ms: Option<u64>,
    pub admissibility: ArtifactAdmissibilityProof,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MaterializationNode {
    pub name: String,
    pub wave: u32,
    pub kind: MaterializationNodeKind,
    pub queue: QueueKind,
    pub plane: CoveragePlane,
    pub dependency_nodes: Vec<String>,
    pub consumes: Vec<String>,
    pub produces: Vec<String>,
    pub rank_scope: Vec<u16>,
    pub invalidation_domain: String,
    pub replay_boundary: String,
    pub expected_compile_ms: Option<u64>,
    pub expected_bytes_written: Option<u64>,
    pub expected_transfer_ms: Option<u64>,
    pub residual_jit_risk_removed: u16,
    pub execution_contract: NodeExecutionContract,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LeaderAssignment {
    pub artifact_scope: String,
    pub leader_rank: u16,
    pub follower_ranks: Vec<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WaveEstimate {
    pub expected_compile_ms: Option<u64>,
    pub expected_bytes_written: Option<u64>,
    pub expected_transfer_ms: Option<u64>,
    pub fanout_count: u16,
    pub residual_jit_risk_removed: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MaterializationWave {
    pub name: String,
    pub queue: QueueKind,
    pub node_names: Vec<String>,
    pub estimate: WaveEstimate,
    pub hazard_repairs: Vec<String>,
    pub execution_contract: WaveExecutionContract,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterializationGraph {
    pub nodes: Vec<MaterializationNode>,
    pub waves: Vec<MaterializationWave>,
    pub leader_assignments: Vec<LeaderAssignment>,
    pub early_serve_frontier: Vec<String>,
    pub late_bindings: Vec<(String, String)>,
    pub runtime_roi: Vec<RuntimeRoi>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ResidualRuntimeRisk {
    pub class: HazardClass,
    pub summary: String,
    pub bounded_to: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuaranteeEnvelope {
    pub requested_correctness: GuaranteeTarget,
    pub requested_performance: GuaranteeTarget,
    pub achieved_correctness: GuaranteeLevel,
    pub achieved_performance: GuaranteeLevel,
    pub covered_dimensions: Vec<GuaranteeDimension>,
    pub covered_shapes: Vec<String>,
    pub residual_risks: Vec<ResidualRuntimeRisk>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ArtifactManifestEntry {
    pub identity: String,
    pub class: ArtifactClass,
    pub backend: BackendFamily,
    pub scope: String,
    pub admissibility: ArtifactAdmissibilityProof,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactClosure {
    pub plan_identity: CanonicalHash,
    pub artifacts: Vec<ArtifactManifestEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedBuildPlan {
    pub normalized_request: NormalizedRequest,
    pub requested_readiness: Option<String>,
    pub optimization_envelope: OptimizationEnvelope,
    pub backend_decision: BackendDecisionPlan,
    pub backend_registry: BackendCapabilityRegistry,
    pub selected_backends: BackendSelection,
    pub compile_regions: Vec<CompileRegion>,
    pub shape_envelope: ShapeEnvelope,
    pub artifact_requirements: Vec<ArtifactRequirement>,
    pub warmup_obligations: Vec<WarmupObligation>,
    pub materialization_graph: MaterializationGraph,
    pub guarantee_envelope: GuaranteeEnvelope,
    pub guarantee_evidence: GuaranteeEvidence,
    pub rewrite_trace: Vec<PassTrace>,
    pub structural_identity: StructuralIdentity,
}

impl ResolvedBuildPlan {
    #[must_use]
    pub fn validate(&self) -> VerificationReport {
        let mut issues = Vec::new();
        let selected_backend_families = std::iter::once(self.selected_backends.primary.family)
            .chain(
                self.selected_backends
                    .secondary
                    .iter()
                    .map(|candidate| candidate.family),
            )
            .collect::<BTreeSet<_>>();
        let expected_artifact_manifest = self
            .artifact_requirements
            .iter()
            .map(|requirement| ArtifactManifestEntry {
                identity: artifact_manifest_identity(
                    self.selected_backends.primary.family,
                    requirement,
                ),
                class: requirement.class,
                backend: requirement.backend,
                scope: requirement.scope.clone(),
                admissibility: requirement.admissibility.clone(),
            })
            .collect::<Vec<_>>();
        let mut actual_artifact_manifest = self.guarantee_evidence.artifact_manifest.clone();
        let mut sorted_expected_artifact_manifest = expected_artifact_manifest.clone();
        sorted_expected_artifact_manifest.sort();
        actual_artifact_manifest.sort();
        let operator_gates = operator_gates();
        let selected_backend_names =
            std::iter::once(self.selected_backends.primary.family.as_str())
                .chain(
                    self.selected_backends
                        .secondary
                        .iter()
                        .map(|candidate| candidate.family.as_str()),
                )
                .collect::<BTreeSet<_>>();

        let widened_compile_regions = self
            .compile_regions
            .iter()
            .filter(|region| !selected_backend_families.contains(&region.family))
            .map(|region| format!("{}:{:?}", region.name, region.family))
            .collect::<Vec<_>>();
        if !widened_compile_regions.is_empty() {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                code: "compile_region_backend_out_of_selection".to_owned(),
                message: format!(
                    "Compile regions widened beyond the selected backend set: {}",
                    widened_compile_regions.join(", ")
                ),
            });
        }

        let widened_artifact_requirements = self
            .artifact_requirements
            .iter()
            .filter(|artifact| !selected_backend_families.contains(&artifact.backend))
            .map(|artifact| format!("{}:{:?}", artifact.scope, artifact.backend))
            .collect::<Vec<_>>();
        if !widened_artifact_requirements.is_empty() {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                code: "artifact_backend_out_of_selection".to_owned(),
                message: format!(
                    "Artifact requirements widened beyond the selected backend set: {}",
                    widened_artifact_requirements.join(", ")
                ),
            });
        }

        if self.selected_backends.primary.family == BackendFamily::FlashInfer
            && !self
                .guarantee_evidence
                .capability_witnesses
                .iter()
                .any(|witness| witness.key == "flashinfer.prebuilt")
        {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                code: "missing_flashinfer_witness".to_owned(),
                message: "FlashInfer selected without a prebuilt capability witness".to_owned(),
            });
        }

        if self.guarantee_envelope.achieved_correctness
            < self.guarantee_envelope.requested_correctness.level
            && self.guarantee_envelope.requested_correctness.failure_mode == FailureMode::FailClosed
        {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                code: "correctness_target_unmet".to_owned(),
                message: "Achieved correctness guarantee is below the requested fail-closed target"
                    .to_owned(),
            });
        }

        if self.guarantee_envelope.achieved_performance
            < self.guarantee_envelope.requested_performance.level
            && self.guarantee_envelope.requested_performance.failure_mode == FailureMode::FailClosed
        {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                code: "performance_target_unmet".to_owned(),
                message: "Achieved performance guarantee is below the requested fail-closed target"
                    .to_owned(),
            });
        }

        let coverage_nodes = self
            .shape_envelope
            .nodes
            .iter()
            .map(|node| node.name.as_str())
            .collect::<BTreeSet<_>>();
        let warmup_nodes = self
            .warmup_obligations
            .iter()
            .map(|obligation| obligation.node_name.as_str())
            .collect::<BTreeSet<_>>();
        let missing_warmup = coverage_nodes
            .difference(&warmup_nodes)
            .filter(|node| {
                self.shape_envelope
                    .nodes
                    .iter()
                    .find(|shape| shape.name == **node)
                    .is_some_and(|shape| shape.intent != RangeIntent::UncoveredResidual)
            })
            .cloned()
            .collect::<Vec<_>>();
        if !missing_warmup.is_empty() && self.requested_readiness.as_deref() != Some("early_serve")
        {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                code: "warmup_incomplete".to_owned(),
                message: format!(
                    "Warmup obligations missing for envelope nodes: {}",
                    missing_warmup.join(", ")
                ),
            });
        }

        if self
            .guarantee_evidence
            .artifact_manifest
            .iter()
            .any(|artifact| artifact.scope.is_empty())
        {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                code: "artifact_scope_missing".to_owned(),
                message: "Artifact manifest contains an unscoped artifact".to_owned(),
            });
        }

        if actual_artifact_manifest != sorted_expected_artifact_manifest {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                code: "artifact_manifest_invalid_reuse".to_owned(),
                message:
                    "Artifact manifest does not exactly match the canonical artifact requirements."
                        .to_owned(),
            });
        }

        for obligation in &self.warmup_obligations {
            if obligation.proof.node_name != obligation.node_name
                || obligation.proof.region_name != obligation.region_name
            {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    code: "warmup_proof_mismatch".to_owned(),
                    message: format!(
                        "Warmup proof does not match obligation {}:{}",
                        obligation.node_name, obligation.region_name
                    ),
                });
            }
        }

        if self
            .materialization_graph
            .early_serve_frontier
            .iter()
            .any(|frontier| {
                !self
                    .materialization_graph
                    .nodes
                    .iter()
                    .any(|node| &node.name == frontier)
            })
        {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                code: "early_serve_frontier_invalid".to_owned(),
                message: "Early-serve frontier references unknown materialization nodes".to_owned(),
            });
        }

        let mut materialization_nodes = BTreeSet::new();
        let duplicate_nodes = self
            .materialization_graph
            .nodes
            .iter()
            .filter_map(|node| {
                if materialization_nodes.insert(node.name.as_str()) {
                    None
                } else {
                    Some(node.name.clone())
                }
            })
            .collect::<Vec<_>>();
        if !duplicate_nodes.is_empty() {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                code: "materialization_node_duplicate".to_owned(),
                message: format!(
                    "Materialization graph contains duplicate node names: {}",
                    duplicate_nodes.join(", ")
                ),
            });
        }

        let residual_shape_nodes = self
            .shape_envelope
            .nodes
            .iter()
            .filter(|node| node.intent == RangeIntent::UncoveredResidual)
            .count();

        if residual_shape_nodes > 0
            && self
                .normalized_request
                .backend_policy
                .runtime_jit_policy
                .disposition
                == crate::RuntimeJitDisposition::Forbidden
        {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                code: "residual_jit_forbidden".to_owned(),
                message: "Residual runtime JIT is present while policy requires fail-closed prebuilt/AoT closure."
                    .to_owned(),
            });
        }

        if self
            .normalized_request
            .backend_policy
            .runtime_jit_policy
            .disposition
            == crate::RuntimeJitDisposition::ShapeBounded
            && self.normalized_request.backend_policy.packaging_strategy
                != crate::PackagingStrategy::PreferPrebuiltThenAotThenJit
        {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                code: "bounded_jit_policy_mismatch".to_owned(),
                message:
                    "Shape-bounded residual JIT requires the explicit prebuilt->AoT->bounded-JIT packaging strategy."
                        .to_owned(),
            });
        }

        if residual_shape_nodes
            > usize::from(
                self.normalized_request
                    .backend_policy
                    .runtime_jit_policy
                    .max_residual_node_count,
            )
        {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                code: "residual_jit_budget_exceeded".to_owned(),
                message: format!(
                    "Residual runtime JIT nodes {} exceed policy budget {}",
                    residual_shape_nodes,
                    self.normalized_request
                        .backend_policy
                        .runtime_jit_policy
                        .max_residual_node_count
                ),
            });
        }

        let total_warmup_steps = self
            .warmup_obligations
            .iter()
            .map(|obligation| obligation.step_count)
            .sum::<u32>();
        if total_warmup_steps > self.optimization_envelope.max_warmup_steps {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                code: "optimization_warmup_budget_exceeded".to_owned(),
                message: format!(
                    "Warmup steps {} exceed optimization budget {} for {}",
                    total_warmup_steps,
                    self.optimization_envelope.max_warmup_steps,
                    self.optimization_envelope.level.as_str()
                ),
            });
        }

        if self.artifact_requirements.len()
            > usize::try_from(
                self.optimization_envelope
                    .artifact_budget
                    .max_artifact_count,
            )
            .expect("artifact budget fits usize")
        {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                code: "optimization_artifact_budget_exceeded".to_owned(),
                message: format!(
                    "Artifact count {} exceeds optimization budget {} for {}",
                    self.artifact_requirements.len(),
                    self.optimization_envelope
                        .artifact_budget
                        .max_artifact_count,
                    self.optimization_envelope.level.as_str()
                ),
            });
        }

        let rank_local_artifacts = self
            .artifact_requirements
            .iter()
            .filter(|artifact| artifact.rank_disposition == RankDisposition::RankLocal)
            .count();
        if rank_local_artifacts
            > usize::try_from(
                self.optimization_envelope
                    .artifact_budget
                    .max_rank_local_artifacts,
            )
            .expect("rank-local budget fits usize")
        {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                code: "optimization_rank_local_budget_exceeded".to_owned(),
                message: format!(
                    "Rank-local artifact count {} exceeds optimization budget {} for {}",
                    rank_local_artifacts,
                    self.optimization_envelope
                        .artifact_budget
                        .max_rank_local_artifacts,
                    self.optimization_envelope.level.as_str()
                ),
            });
        }

        if self.selected_backends.primary.admissibility.verdict
            != crate::AdmissibilityVerdict::Admissible
        {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                code: "backend_selection_not_admissible".to_owned(),
                message: format!(
                    "Primary backend {:?} is not admissible: {}",
                    self.selected_backends.primary.family,
                    self.selected_backends
                        .primary
                        .admissibility
                        .rejected_reasons
                        .join(", ")
                ),
            });
        }

        for artifact in &self.artifact_requirements {
            if !artifact.admissibility.fail_closed {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    code: "artifact_not_fail_closed".to_owned(),
                    message: format!(
                        "Artifact {} does not carry a fail-closed admissibility proof",
                        artifact.scope
                    ),
                });
            }
        }

        let capability_witnesses = self
            .guarantee_evidence
            .capability_witnesses
            .iter()
            .map(|witness| (witness.key.as_str(), witness.value.as_str()))
            .collect::<Vec<_>>();
        for witness_key in &self
            .selected_backends
            .primary
            .admissibility
            .required_witnesses
        {
            match capability_witnesses
                .iter()
                .find(|(key, _)| key == witness_key)
                .copied()
            {
                Some((_, value))
                    if value.eq_ignore_ascii_case("missing")
                        || value.eq_ignore_ascii_case("absent")
                        || value.eq_ignore_ascii_case("false") =>
                {
                    issues.push(ValidationIssue {
                        severity: ValidationSeverity::Error,
                        code: "capability_witness_contradiction".to_owned(),
                        message: format!(
                            "Capability witness {} contradicts backend admissibility.",
                            witness_key
                        ),
                    });
                }
                None => issues.push(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    code: "capability_witness_missing".to_owned(),
                    message: format!(
                        "Capability witness {} required by backend admissibility is missing.",
                        witness_key
                    ),
                }),
                Some(_) => {}
            }
        }

        if self
            .guarantee_envelope
            .residual_risks
            .iter()
            .any(|risk| risk.class == HazardClass::ResidualLazyCompile)
            && self.guarantee_evidence.runtime_jit_evidence.is_empty()
        {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                code: "runtime_jit_evidence_missing".to_owned(),
                message:
                    "Residual runtime JIT risk exists but no structured runtime-JIT evidence was captured."
                        .to_owned(),
            });
        }

        for evidence in &self.guarantee_evidence.runtime_jit_evidence {
            if evidence.bounded_by.is_empty() {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    code: "runtime_jit_evidence_unbounded".to_owned(),
                    message: format!(
                        "Runtime-JIT surface {} is not bounded by any proof or artifact identity.",
                        evidence.surface_name
                    ),
                });
            }
            if !evidence.contradiction_reasons.is_empty() {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    code: "runtime_jit_evidence_contradiction".to_owned(),
                    message: format!(
                        "Runtime-JIT surface {} has contradiction evidence: {}",
                        evidence.surface_name,
                        evidence.contradiction_reasons.join(", ")
                    ),
                });
            }
            for region in &evidence.affected_regions {
                if !self
                    .compile_regions
                    .iter()
                    .any(|candidate| &candidate.name == region)
                {
                    issues.push(ValidationIssue {
                        severity: ValidationSeverity::Error,
                        code: "runtime_jit_region_unknown".to_owned(),
                        message: format!(
                            "Runtime-JIT surface {} references unknown compile region {}.",
                            evidence.surface_name, region
                        ),
                    });
                }
            }
            for artifact in &evidence.required_artifacts {
                if !self
                    .artifact_requirements
                    .iter()
                    .any(|candidate| &candidate.scope == artifact)
                {
                    issues.push(ValidationIssue {
                        severity: ValidationSeverity::Error,
                        code: "runtime_jit_artifact_unknown".to_owned(),
                        message: format!(
                            "Runtime-JIT surface {} references unknown artifact scope {}.",
                            evidence.surface_name, artifact
                        ),
                    });
                }
            }
            for proof in &evidence.required_warmup_proofs {
                if !self
                    .warmup_obligations
                    .iter()
                    .any(|obligation| &obligation.proof.proof_id == proof)
                {
                    issues.push(ValidationIssue {
                        severity: ValidationSeverity::Error,
                        code: "runtime_jit_warmup_unknown".to_owned(),
                        message: format!(
                            "Runtime-JIT surface {} references unknown warmup proof {}.",
                            evidence.surface_name, proof
                        ),
                    });
                }
            }
        }

        for gate in &operator_gates {
            if gate.compile_free && gate.forbidden_queues.is_empty() {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    code: "operator_gate_incomplete".to_owned(),
                    message: format!(
                        "Operator gate {} claims compile-free execution without forbidden queues.",
                        gate.command
                    ),
                });
            }
        }

        for pass in &self.rewrite_trace {
            if let Err(message) = pass.validate() {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    code: "rewrite_pass_contract_violation".to_owned(),
                    message,
                });
            }
        }

        for entry in &self.backend_decision.entries {
            let selected = selected_backend_names.contains(entry.family.as_str());
            if entry.selected_for_deployment != selected {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    code: "backend_decision_selection_mismatch".to_owned(),
                    message: format!(
                        "Backend decision entry {} disagrees with selected backend set.",
                        entry.family.as_str()
                    ),
                });
            }
            if entry.reachable_from_materialization_plan
                && entry.reachable_compile_regions.is_empty()
                && entry.reachable_artifact_scopes.is_empty()
            {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    code: "backend_decision_plan_reachability_unbacked".to_owned(),
                    message: format!(
                        "Backend decision entry {} claims plan reachability without compile regions or artifact scopes.",
                        entry.family.as_str()
                    ),
                });
            }
        }

        let status = if issues
            .iter()
            .any(|issue| issue.severity == ValidationSeverity::Error)
        {
            ValidationStatus::Failed
        } else {
            ValidationStatus::Passed
        };

        VerificationReport {
            level: ValidationLevel::WitnessBacked,
            status,
            issues,
            phase_timings: self
                .materialization_graph
                .waves
                .iter()
                .map(|wave| (wave.queue, wave.estimate.expected_compile_ms))
                .collect(),
            runtime_jit_witnesses: self
                .guarantee_envelope
                .residual_risks
                .iter()
                .filter(|risk| risk.class == HazardClass::ResidualLazyCompile)
                .map(|risk| risk.summary.clone())
                .collect(),
            runtime_jit_evidence: self.guarantee_evidence.runtime_jit_evidence.clone(),
            operator_gates,
        }
    }
}

fn operator_gates() -> Vec<OperatorGate> {
    let forbidden_queues = vec![
        QueueKind::Compile,
        QueueKind::Assemble,
        QueueKind::ArtifactIo,
        QueueKind::Warmup,
    ];
    vec![
        OperatorGate {
            command: "verify".to_owned(),
            compile_free: true,
            forbidden_queues: forbidden_queues.clone(),
            rationale:
                "Bundle verification must be purely structural and witness-backed; it may not materialize or compile anything."
                    .to_owned(),
        },
        OperatorGate {
            command: "replay".to_owned(),
            compile_free: true,
            forbidden_queues,
            rationale:
                "Replay must consume recorded bundle state only and never perform new artifact, compile, or warmup work."
                    .to_owned(),
        },
    ]
}

#[cfg(test)]
mod tests {
    use crate::{RawRequest, canonical_json, parse_canonical_json};

    fn base_request() -> RawRequest {
        RawRequest {
            engine: crate::TargetEngine::Vllm,
            model: crate::ModelRef {
                repository: "meta-llama/Llama-3.1-8B-Instruct".to_owned(),
                revision: "main".to_owned(),
            },
            engine_source: crate::EngineSource {
                kind: "vendored".to_owned(),
                revision: "deadbeef".to_owned(),
            },
            environment: crate::RequestedEnvironment {
                operating_system: crate::OperatingSystem::Linux,
                accelerator_vendor: crate::AcceleratorVendor::Nvidia,
                gpu_arches: vec!["sm90".to_owned(), "sm80".to_owned()],
                cuda_version: "12.4".to_owned(),
                driver_version: "550.54".to_owned(),
                python_abi: "cp311".to_owned(),
                libc_abi: "glibc-2.35".to_owned(),
            },
            topology: crate::ExecutionTopology {
                tensor_parallelism: 2,
                pipeline_parallelism: 1,
                replicas: 1,
            },
            kv_layout_policy: crate::KvLayoutPolicy::standard(),
            backend_policy: crate::BackendPolicy {
                preferred_families: vec![
                    crate::BackendFamily::Triton,
                    crate::BackendFamily::FlashInfer,
                    crate::BackendFamily::Triton,
                ],
                packaging_strategy: crate::PackagingStrategy::PreferPrebuiltThenAot,
                runtime_jit_policy: crate::RuntimeJitPolicy {
                    disposition: crate::RuntimeJitDisposition::Forbidden,
                    max_residual_node_count: 0,
                },
                correctness_target: crate::GuaranteeTarget {
                    level: crate::GuaranteeLevel::ShapeBoundedAot,
                    failure_mode: crate::FailureMode::FailClosed,
                },
                performance_target: crate::GuaranteeTarget {
                    level: crate::GuaranteeLevel::WarmupBounded,
                    failure_mode: crate::FailureMode::FailOpen,
                },
            },
            shape_policy: crate::ShapePolicy {
                correctness_range: crate::ShapeRange {
                    min_batch_size: 1,
                    max_batch_size: 8,
                    min_sequence_length: 1,
                    max_sequence_length: 4096,
                },
                performance_range: crate::ShapeRange {
                    min_batch_size: 1,
                    max_batch_size: 4,
                    min_sequence_length: 1,
                    max_sequence_length: 2048,
                },
                hot_shapes: vec![
                    crate::ShapePoint {
                        batch_size: 1,
                        sequence_length: 128,
                        plane: crate::CoveragePlane::Performance,
                    },
                    crate::ShapePoint {
                        batch_size: 4,
                        sequence_length: 2048,
                        plane: crate::CoveragePlane::Performance,
                    },
                ],
                cuda_graph_shapes: vec![crate::ShapePoint {
                    batch_size: 1,
                    sequence_length: 128,
                    plane: crate::CoveragePlane::CudaGraph,
                }],
            },
            cache_policy: crate::CachePolicy {
                namespace: "prod".to_owned(),
                allow_cross_machine_reuse: false,
            },
            warmup_policy: crate::WarmupPolicy {
                max_warmup_steps: 6,
                verify_cuda_graph_capture: true,
            },
            optimization_policy: crate::OptimizationPolicy {
                level: crate::OptimizationLevel::O2,
            },
            layered_config: vec![crate::ConfigLayer {
                name: "project".to_owned(),
                precedence: 1,
                entries: vec![crate::ConfigEntry {
                    key: "tensor_parallel_size".to_owned(),
                    value: "2".to_owned(),
                }],
            }],
        }
    }

    #[test]
    fn equivalent_requests_share_identity() {
        let mut reordered = base_request();
        reordered.environment.gpu_arches.reverse();
        reordered.backend_policy.preferred_families.reverse();

        let a = base_request().normalize().expect("normalized");
        let b = reordered.normalize().expect("normalized");

        assert_eq!(a.identity, b.identity);
    }

    #[test]
    fn topology_changes_invalidate_identity() {
        let a = base_request().normalize().expect("normalized");
        let mut changed = base_request();
        changed.topology.tensor_parallelism = 4;
        let b = changed.normalize().expect("normalized");

        assert_ne!(a.identity, b.identity);
    }

    #[test]
    fn kv_layout_changes_invalidate_identity() {
        let a = base_request().normalize().expect("normalized");
        let mut changed = base_request();
        changed.kv_layout_policy = crate::KvLayoutPolicy::tmh_accounting();
        let b = changed.normalize().expect("normalized");

        assert_ne!(a.identity, b.identity);
    }

    #[test]
    fn canonical_round_trip_is_stable() {
        let request = base_request().normalize().expect("normalized");
        let rendered = canonical_json(&request).expect("json");
        let reparsed: crate::NormalizedRequest = parse_canonical_json(&rendered).expect("parse");

        assert_eq!(request, reparsed);
    }
}
