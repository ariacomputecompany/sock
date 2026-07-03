use std::cmp::Ordering;
use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::canonical::{CanonicalError, CanonicalHash, canonical_hash};

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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ConfigLayer {
    pub name: String,
    pub precedence: u8,
    pub entries: Vec<ConfigEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ConfigEntry {
    pub key: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ModelRef {
    pub repository: String,
    pub revision: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EngineSource {
    pub kind: String,
    pub revision: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RequestedEnvironment {
    pub operating_system: OperatingSystem,
    pub accelerator_vendor: AcceleratorVendor,
    pub gpu_arches: Vec<String>,
    pub cuda_version: String,
    pub driver_version: String,
    pub python_abi: String,
    pub libc_abi: String,
}

impl RequestedEnvironment {
    pub fn canonicalize(&mut self) {
        self.gpu_arches.sort();
        self.gpu_arches.dedup();
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ExecutionTopology {
    pub tensor_parallelism: u16,
    pub pipeline_parallelism: u16,
    pub replicas: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GuaranteeTarget {
    pub level: GuaranteeLevel,
    pub failure_mode: FailureMode,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendPolicy {
    pub preferred_families: Vec<BackendFamily>,
    pub require_prebuilt_artifacts: bool,
    pub allow_runtime_jit: bool,
    pub correctness_target: GuaranteeTarget,
    pub performance_target: GuaranteeTarget,
}

impl BackendPolicy {
    pub fn canonicalize(&mut self) {
        self.preferred_families.sort();
        self.preferred_families.dedup();
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ShapeRange {
    pub min_batch_size: u32,
    pub max_batch_size: u32,
    pub min_sequence_length: u32,
    pub max_sequence_length: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ShapePoint {
    pub batch_size: u32,
    pub sequence_length: u32,
    pub plane: CoveragePlane,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShapePolicy {
    pub correctness_range: ShapeRange,
    pub performance_range: ShapeRange,
    pub hot_shapes: Vec<ShapePoint>,
    pub cuda_graph_shapes: Vec<ShapePoint>,
}

impl ShapePolicy {
    pub fn canonicalize(&mut self) {
        self.hot_shapes.sort();
        self.hot_shapes.dedup();
        self.cuda_graph_shapes.sort();
        self.cuda_graph_shapes.dedup();
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CachePolicy {
    pub namespace: String,
    pub allow_cross_machine_reuse: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WarmupPolicy {
    pub max_warmup_steps: u32,
    pub verify_cuda_graph_capture: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RawRequest {
    pub engine: TargetEngine,
    pub model: ModelRef,
    pub engine_source: EngineSource,
    pub environment: RequestedEnvironment,
    pub topology: ExecutionTopology,
    pub backend_policy: BackendPolicy,
    pub shape_policy: ShapePolicy,
    pub cache_policy: CachePolicy,
    pub warmup_policy: WarmupPolicy,
    pub layered_config: Vec<ConfigLayer>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizedRequest {
    pub engine: TargetEngine,
    pub model: ModelRef,
    pub engine_source: EngineSource,
    pub environment: RequestedEnvironment,
    pub topology: ExecutionTopology,
    pub backend_policy: BackendPolicy,
    pub shape_policy: ShapePolicy,
    pub cache_policy: CachePolicy,
    pub warmup_policy: WarmupPolicy,
    pub layered_config: Vec<ConfigLayer>,
    pub identity: CanonicalHash,
}

impl RawRequest {
    pub fn normalize(mut self) -> Result<NormalizedRequest, CanonicalError> {
        self.environment.canonicalize();
        self.backend_policy.canonicalize();
        self.shape_policy.canonicalize();
        self.layered_config
            .sort_by_key(|layer| (layer.precedence, layer.name.clone()));
        for layer in &mut self.layered_config {
            layer.entries.sort();
            layer.entries.dedup();
        }

        let body = NormalizedRequestBody {
            engine: self.engine,
            model: self.model,
            engine_source: self.engine_source,
            environment: self.environment,
            topology: self.topology,
            backend_policy: self.backend_policy,
            shape_policy: self.shape_policy,
            cache_policy: self.cache_policy,
            warmup_policy: self.warmup_policy,
            layered_config: self.layered_config,
        };
        let identity = canonical_hash(&body)?;

        Ok(NormalizedRequest {
            engine: body.engine,
            model: body.model,
            engine_source: body.engine_source,
            environment: body.environment,
            topology: body.topology,
            backend_policy: body.backend_policy,
            shape_policy: body.shape_policy,
            cache_policy: body.cache_policy,
            warmup_policy: body.warmup_policy,
            layered_config: body.layered_config,
            identity,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct NormalizedRequestBody {
    engine: TargetEngine,
    model: ModelRef,
    engine_source: EngineSource,
    environment: RequestedEnvironment,
    topology: ExecutionTopology,
    backend_policy: BackendPolicy,
    shape_policy: ShapePolicy,
    cache_policy: CachePolicy,
    warmup_policy: WarmupPolicy,
    layered_config: Vec<ConfigLayer>,
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
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendSelection {
    pub primary: BackendCandidate,
    pub secondary: Vec<BackendCandidate>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CompileRegion {
    pub name: String,
    pub family: BackendFamily,
    pub reusable: bool,
    pub shape_planes: Vec<CoveragePlane>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ShapeEnvelopeNode {
    pub name: String,
    pub plane: CoveragePlane,
    pub intent: RangeIntent,
    pub range: ShapeRange,
    pub exact_shape: Option<ShapePoint>,
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
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ArtifactRequirement {
    pub class: ArtifactClass,
    pub backend: BackendFamily,
    pub acquisition: ArtifactAcquisition,
    pub scope: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MaterializationNode {
    pub name: String,
    pub wave: u32,
    pub consumes: Vec<String>,
    pub produces: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterializationGraph {
    pub nodes: Vec<MaterializationNode>,
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
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactClosure {
    pub plan_identity: CanonicalHash,
    pub artifacts: Vec<ArtifactManifestEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CoverageWitness {
    pub plane: CoveragePlane,
    pub node_name: String,
    pub evidence: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuaranteeEvidence {
    pub capability_witnesses: Vec<CapabilityWitness>,
    pub artifact_manifest: Vec<ArtifactManifestEntry>,
    pub warmup_obligations: Vec<WarmupObligation>,
    pub coverage_witnesses: Vec<CoverageWitness>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub severity: ValidationSeverity,
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationReport {
    pub level: ValidationLevel,
    pub status: ValidationStatus,
    pub issues: Vec<ValidationIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PassTrace {
    pub pass_name: String,
    pub before_identity: String,
    pub after_identity: String,
    pub matched_rules: Vec<String>,
    pub repairs: Vec<String>,
    pub invalidated_assumptions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuralIdentity {
    pub request_identity: CanonicalHash,
    pub shape_envelope_identity: CanonicalHash,
    pub compile_region_identity: CanonicalHash,
    pub capability_identity: CanonicalHash,
    pub abi_identity: CanonicalHash,
    pub artifact_identity: CanonicalHash,
    pub evidence_identity: CanonicalHash,
    pub plan_identity: CanonicalHash,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedBuildPlan {
    pub normalized_request: NormalizedRequest,
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{canonical_json, parse_canonical_json};

    fn base_request() -> RawRequest {
        RawRequest {
            engine: TargetEngine::Vllm,
            model: ModelRef {
                repository: "meta-llama/Llama-3.1-8B-Instruct".to_owned(),
                revision: "main".to_owned(),
            },
            engine_source: EngineSource {
                kind: "vendored".to_owned(),
                revision: "deadbeef".to_owned(),
            },
            environment: RequestedEnvironment {
                operating_system: OperatingSystem::Linux,
                accelerator_vendor: AcceleratorVendor::Nvidia,
                gpu_arches: vec!["sm90".to_owned(), "sm80".to_owned()],
                cuda_version: "12.4".to_owned(),
                driver_version: "550.54".to_owned(),
                python_abi: "cp311".to_owned(),
                libc_abi: "glibc-2.35".to_owned(),
            },
            topology: ExecutionTopology {
                tensor_parallelism: 2,
                pipeline_parallelism: 1,
                replicas: 1,
            },
            backend_policy: BackendPolicy {
                preferred_families: vec![
                    BackendFamily::Triton,
                    BackendFamily::FlashInfer,
                    BackendFamily::Triton,
                ],
                require_prebuilt_artifacts: true,
                allow_runtime_jit: false,
                correctness_target: GuaranteeTarget {
                    level: GuaranteeLevel::ShapeBoundedAot,
                    failure_mode: FailureMode::FailClosed,
                },
                performance_target: GuaranteeTarget {
                    level: GuaranteeLevel::WarmupBounded,
                    failure_mode: FailureMode::FailOpen,
                },
            },
            shape_policy: ShapePolicy {
                correctness_range: ShapeRange {
                    min_batch_size: 1,
                    max_batch_size: 8,
                    min_sequence_length: 1,
                    max_sequence_length: 4096,
                },
                performance_range: ShapeRange {
                    min_batch_size: 1,
                    max_batch_size: 4,
                    min_sequence_length: 1,
                    max_sequence_length: 2048,
                },
                hot_shapes: vec![
                    ShapePoint {
                        batch_size: 1,
                        sequence_length: 128,
                        plane: CoveragePlane::Performance,
                    },
                    ShapePoint {
                        batch_size: 4,
                        sequence_length: 2048,
                        plane: CoveragePlane::Performance,
                    },
                ],
                cuda_graph_shapes: vec![ShapePoint {
                    batch_size: 1,
                    sequence_length: 128,
                    plane: CoveragePlane::CudaGraph,
                }],
            },
            cache_policy: CachePolicy {
                namespace: "prod".to_owned(),
                allow_cross_machine_reuse: false,
            },
            warmup_policy: WarmupPolicy {
                max_warmup_steps: 6,
                verify_cuda_graph_capture: true,
            },
            layered_config: vec![
                ConfigLayer {
                    name: "env".to_owned(),
                    precedence: 0,
                    entries: vec![ConfigEntry {
                        key: "VLLM_USE_V1".to_owned(),
                        value: "1".to_owned(),
                    }],
                },
                ConfigLayer {
                    name: "project".to_owned(),
                    precedence: 1,
                    entries: vec![ConfigEntry {
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
        let reparsed: NormalizedRequest = parse_canonical_json(&rendered).expect("parse");

        assert_eq!(request, reparsed);
    }
}
