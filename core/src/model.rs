use std::cmp::Ordering;
use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::CanonicalHash;
use crate::adapter::{CompileRegionKind, SourceEvidence};
use crate::backend::{
    ArtifactAdmissibilityProof, BackendAdmissibilityProof, BackendCapabilityRegistry,
};
use crate::identity::StructuralIdentity;
use crate::request::{GuaranteeTarget, NormalizedRequest};
use crate::rewrite::PassTrace;
use crate::runtime::{
    NodeExecutionContract, RuntimeRoi, WarmupCoverageProof, WaveExecutionContract,
};
use crate::verification::{GuaranteeEvidence, ValidationIssue, VerificationReport};

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
        if !missing_warmup.is_empty() {
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

        for pass in &self.rewrite_trace {
            if let Err(message) = pass.validate() {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    code: "rewrite_pass_contract_violation".to_owned(),
                    message,
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
        }
    }
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
            layered_config: vec![
                crate::ConfigLayer {
                    name: "env".to_owned(),
                    precedence: 0,
                    entries: vec![crate::ConfigEntry {
                        key: "VLLM_USE_V1".to_owned(),
                        value: "1".to_owned(),
                    }],
                },
                crate::ConfigLayer {
                    name: "project".to_owned(),
                    precedence: 1,
                    entries: vec![crate::ConfigEntry {
                        key: "tensor_parallel_size".to_owned(),
                        value: "2".to_owned(),
                    }],
                },
            ],
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
    fn canonical_round_trip_is_stable() {
        let request = base_request().normalize().expect("normalized");
        let rendered = canonical_json(&request).expect("json");
        let reparsed: crate::NormalizedRequest = parse_canonical_json(&rendered).expect("parse");

        assert_eq!(request, reparsed);
    }
}
