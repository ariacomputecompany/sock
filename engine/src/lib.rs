use anyhow::Result;
use sock_core::{
    AcceleratorVendor, ArtifactAcquisition, ArtifactClass, ArtifactClosure, ArtifactManifestEntry,
    BackendCandidate, BackendFamily, BackendSelection, CapabilityWitness, CompileRegion,
    CoveragePlane, GuaranteeDimension, GuaranteeEnvelope, GuaranteeEvidence,
    GuaranteeLevel, HazardClass, MaterializationGraph, MaterializationNode, NormalizedRequest,
    PassTrace, RangeIntent, RawRequest, ResidualRuntimeRisk, ResolvedBuildPlan, ShapeEnvelope,
    ShapeEnvelopeNode, VerificationReport, WarmupObligation, canonical_hash,
};

pub mod vllm;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannerHostSnapshot {
    pub operating_system: sock_core::OperatingSystem,
    pub accelerator_vendor: AcceleratorVendor,
    pub gpu_arches: Vec<String>,
    pub cuda_version: String,
    pub driver_version: String,
    pub python_abi: String,
    pub libc_abi: String,
    pub flashinfer_prebuilt_available: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanningOutcome {
    pub plan: ResolvedBuildPlan,
    pub closure: ArtifactClosure,
    pub verification: VerificationReport,
}

#[derive(Debug, Clone)]
pub struct Planner {
    host: PlannerHostSnapshot,
}

impl Planner {
    #[must_use]
    pub fn new(host: PlannerHostSnapshot) -> Self {
        Self { host }
    }

    pub fn resolve(&self, request: RawRequest) -> Result<PlanningOutcome> {
        let normalized_request = request.normalize()?;
        let selected_backends = self.select_backends(&normalized_request);
        let compile_regions = self.compile_regions(&selected_backends);
        let shape_envelope = self.shape_envelope(&normalized_request, &selected_backends);
        let warmup_obligations = self.warmup_obligations(&shape_envelope, &compile_regions);
        let materialization_graph = self.materialization_graph(&warmup_obligations);
        let capability_witnesses = self.capability_witnesses(&selected_backends);
        let artifacts = self.artifacts(&selected_backends, &shape_envelope);
        let guarantee_envelope =
            self.guarantee_envelope(&normalized_request, &selected_backends, &shape_envelope);
        let guarantee_evidence = GuaranteeEvidence {
            capability_witnesses,
            artifact_manifest: artifacts.clone(),
            warmup_obligations: warmup_obligations.clone(),
            coverage_witnesses: shape_envelope
                .nodes
                .iter()
                .map(|node| sock_core::CoverageWitness {
                    plane: node.plane,
                    node_name: node.name.clone(),
                    evidence: format!("planned:{}", node.name),
                })
                .collect(),
        };
        let rewrite_trace = self.rewrite_trace(&normalized_request, &selected_backends);
        let structural_identity = self.structural_identity(
            &normalized_request,
            &shape_envelope,
            &compile_regions,
            &guarantee_evidence.capability_witnesses,
            &artifacts,
            &guarantee_evidence,
        )?;

        let plan = ResolvedBuildPlan {
            normalized_request,
            selected_backends,
            compile_regions,
            shape_envelope,
            artifact_requirements: artifacts
                .iter()
                .map(|artifact| sock_core::ArtifactRequirement {
                    class: artifact.class,
                    backend: artifact.backend,
                    acquisition: match artifact.backend {
                        BackendFamily::FlashInfer => ArtifactAcquisition::VendorPrebuilt,
                        BackendFamily::Triton => ArtifactAcquisition::LocalAotBuild,
                        BackendFamily::AotInductor => ArtifactAcquisition::LocalAotBuild,
                        BackendFamily::CudaGraphs => ArtifactAcquisition::LocalSourceBuild,
                    },
                    scope: artifact.scope.clone(),
                })
                .collect(),
            warmup_obligations,
            materialization_graph,
            guarantee_envelope,
            guarantee_evidence,
            rewrite_trace,
            structural_identity,
        };

        let verification = plan.validate();
        let closure = ArtifactClosure {
            plan_identity: plan.structural_identity.plan_identity.clone(),
            artifacts,
        };

        Ok(PlanningOutcome {
            plan,
            closure,
            verification,
        })
    }

    fn select_backends(&self, request: &NormalizedRequest) -> BackendSelection {
        let mut preferred = request.backend_policy.preferred_families.clone();
        if preferred.is_empty() {
            preferred.push(BackendFamily::Triton);
        }

        let primary_family = if self.host.flashinfer_prebuilt_available
            && preferred.contains(&BackendFamily::FlashInfer)
        {
            BackendFamily::FlashInfer
        } else {
            preferred[0]
        };

        let primary = BackendCandidate {
            family: primary_family,
            acquisition: match primary_family {
                BackendFamily::FlashInfer => ArtifactAcquisition::VendorPrebuilt,
                BackendFamily::Triton => ArtifactAcquisition::LocalAotBuild,
                BackendFamily::AotInductor => ArtifactAcquisition::LocalAotBuild,
                BackendFamily::CudaGraphs => ArtifactAcquisition::LocalSourceBuild,
            },
            reason: "selected from preferred backend policy under current host capabilities"
                .to_owned(),
        };

        let secondary = preferred
            .into_iter()
            .filter(|family| *family != primary.family)
            .map(|family| BackendCandidate {
                family,
                acquisition: ArtifactAcquisition::LocalSourceBuild,
                reason: "eligible fallback under the declared policy".to_owned(),
            })
            .collect();

        BackendSelection { primary, secondary }
    }

    fn compile_regions(&self, backends: &BackendSelection) -> Vec<CompileRegion> {
        vec![
            CompileRegion {
                name: "prefill-region".to_owned(),
                family: backends.primary.family,
                reusable: true,
                shape_planes: vec![CoveragePlane::Correctness, CoveragePlane::Performance],
            },
            CompileRegion {
                name: "decode-region".to_owned(),
                family: backends.primary.family,
                reusable: true,
                shape_planes: vec![CoveragePlane::Correctness, CoveragePlane::Performance],
            },
        ]
    }

    fn shape_envelope(
        &self,
        request: &NormalizedRequest,
        backends: &BackendSelection,
    ) -> ShapeEnvelope {
        let mut nodes = vec![
            ShapeEnvelopeNode {
                name: "correctness-range".to_owned(),
                plane: CoveragePlane::Correctness,
                intent: RangeIntent::SymbolicRange,
                range: request.shape_policy.correctness_range.clone(),
                exact_shape: None,
                required_backends: vec![backends.primary.family],
            },
            ShapeEnvelopeNode {
                name: "performance-range".to_owned(),
                plane: CoveragePlane::Performance,
                intent: RangeIntent::FallbackRange,
                range: request.shape_policy.performance_range.clone(),
                exact_shape: None,
                required_backends: vec![backends.primary.family],
            },
        ];

        for (index, shape) in request.shape_policy.hot_shapes.iter().enumerate() {
            nodes.push(ShapeEnvelopeNode {
                name: format!("hot-shape-{index}"),
                plane: shape.plane,
                intent: RangeIntent::ExactHotShape,
                range: sock_core::ShapeRange {
                    min_batch_size: shape.batch_size,
                    max_batch_size: shape.batch_size,
                    min_sequence_length: shape.sequence_length,
                    max_sequence_length: shape.sequence_length,
                },
                exact_shape: Some(shape.clone()),
                required_backends: vec![backends.primary.family],
            });
        }

        ShapeEnvelope { nodes }
    }

    fn warmup_obligations(
        &self,
        envelope: &ShapeEnvelope,
        regions: &[CompileRegion],
    ) -> Vec<WarmupObligation> {
        envelope
            .nodes
            .iter()
            .filter(|node| node.intent != RangeIntent::UncoveredResidual)
            .map(|node| WarmupObligation {
                node_name: node.name.clone(),
                region_name: regions
                    .first()
                    .map(|region| region.name.clone())
                    .unwrap_or_else(|| "prefill-region".to_owned()),
                step_count: 1,
            })
            .collect()
    }

    fn materialization_graph(&self, obligations: &[WarmupObligation]) -> MaterializationGraph {
        let mut nodes = Vec::new();
        nodes.push(MaterializationNode {
            name: "compile-primary-backend".to_owned(),
            wave: 0,
            consumes: vec!["normalized-request".to_owned()],
            produces: vec!["compiled-artifacts".to_owned()],
        });
        nodes.push(MaterializationNode {
            name: "materialize-artifacts".to_owned(),
            wave: 1,
            consumes: vec!["compiled-artifacts".to_owned()],
            produces: vec!["artifact-closure".to_owned()],
        });
        for (index, obligation) in obligations.iter().enumerate() {
            nodes.push(MaterializationNode {
                name: format!("warmup-{}", obligation.node_name),
                wave: 2 + index as u32,
                consumes: vec!["artifact-closure".to_owned()],
                produces: vec![format!("coverage:{}", obligation.node_name)],
            });
        }
        MaterializationGraph { nodes }
    }

    fn capability_witnesses(&self, backends: &BackendSelection) -> Vec<CapabilityWitness> {
        let mut witnesses = vec![
            CapabilityWitness {
                key: "host.os".to_owned(),
                value: "linux".to_owned(),
                provenance: "planner-host-snapshot".to_owned(),
            },
            CapabilityWitness {
                key: "host.accelerator".to_owned(),
                value: match self.host.accelerator_vendor {
                    AcceleratorVendor::Nvidia => "nvidia".to_owned(),
                },
                provenance: "planner-host-snapshot".to_owned(),
            },
        ];

        if backends.primary.family == BackendFamily::FlashInfer
            && self.host.flashinfer_prebuilt_available
        {
            witnesses.push(CapabilityWitness {
                key: "flashinfer.prebuilt".to_owned(),
                value: "true".to_owned(),
                provenance: "planner-host-snapshot".to_owned(),
            });
        }

        witnesses
    }

    fn artifacts(
        &self,
        backends: &BackendSelection,
        envelope: &ShapeEnvelope,
    ) -> Vec<ArtifactManifestEntry> {
        let mut artifacts = vec![ArtifactManifestEntry {
            identity: format!("{}:primary", backends.primary.family.as_str()),
            class: match backends.primary.family {
                BackendFamily::FlashInfer => ArtifactClass::BackendPackageInput,
                BackendFamily::Triton | BackendFamily::AotInductor => ArtifactClass::CompiledGraph,
                BackendFamily::CudaGraphs => ArtifactClass::CudaGraphCapture,
            },
            backend: backends.primary.family,
            scope: "host".to_owned(),
        }];

        if envelope
            .nodes
            .iter()
            .any(|node| node.plane == CoveragePlane::CudaGraph)
        {
            artifacts.push(ArtifactManifestEntry {
                identity: "cuda-graph:capture".to_owned(),
                class: ArtifactClass::CudaGraphCapture,
                backend: BackendFamily::CudaGraphs,
                scope: "host".to_owned(),
            });
        }

        artifacts
    }

    fn guarantee_envelope(
        &self,
        request: &NormalizedRequest,
        backends: &BackendSelection,
        envelope: &ShapeEnvelope,
    ) -> GuaranteeEnvelope {
        let achieved_correctness = request.backend_policy.correctness_target.level;
        let achieved_performance = if backends.primary.family == BackendFamily::FlashInfer
            && self.host.flashinfer_prebuilt_available
            && !request.backend_policy.allow_runtime_jit
        {
            GuaranteeLevel::StrictNoSurpriseJit
        } else {
            request.backend_policy.performance_target.level
        };

        GuaranteeEnvelope {
            requested_correctness: request.backend_policy.correctness_target.clone(),
            requested_performance: request.backend_policy.performance_target.clone(),
            achieved_correctness,
            achieved_performance,
            covered_dimensions: vec![
                GuaranteeDimension::Environment,
                GuaranteeDimension::Kernel,
                GuaranteeDimension::Shape,
                GuaranteeDimension::Runtime,
                GuaranteeDimension::Topology,
            ],
            covered_shapes: envelope
                .nodes
                .iter()
                .map(|node| node.name.clone())
                .collect(),
            residual_risks: if request.backend_policy.allow_runtime_jit {
                vec![ResidualRuntimeRisk {
                    class: HazardClass::ResidualLazyCompile,
                    summary: "runtime jit allowed by policy".to_owned(),
                    bounded_to: Some("declared envelope".to_owned()),
                }]
            } else {
                Vec::new()
            },
        }
    }

    fn rewrite_trace(
        &self,
        request: &NormalizedRequest,
        backends: &BackendSelection,
    ) -> Vec<PassTrace> {
        vec![
            PassTrace {
                pass_name: "normalize-request".to_owned(),
                before_identity: "raw-request".to_owned(),
                after_identity: request.identity.to_string(),
                matched_rules: vec![
                    "config-layer-sort".to_owned(),
                    "shape-policy-sort".to_owned(),
                ],
                repairs: Vec::new(),
                invalidated_assumptions: Vec::new(),
            },
            PassTrace {
                pass_name: "select-backend".to_owned(),
                before_identity: request.identity.to_string(),
                after_identity: format!("backend:{}", backends.primary.family.as_str()),
                matched_rules: vec!["prefer-prebuilt-flashinfer".to_owned()],
                repairs: Vec::new(),
                invalidated_assumptions: Vec::new(),
            },
        ]
    }

    fn structural_identity(
        &self,
        request: &NormalizedRequest,
        envelope: &ShapeEnvelope,
        regions: &[CompileRegion],
        witnesses: &[CapabilityWitness],
        artifacts: &[ArtifactManifestEntry],
        evidence: &GuaranteeEvidence,
    ) -> Result<sock_core::StructuralIdentity> {
        let shape_envelope_identity = canonical_hash(envelope)?;
        let compile_region_identity = canonical_hash(regions)?;
        let capability_identity = canonical_hash(witnesses)?;
        let abi_identity = canonical_hash(&(
            &self.host.gpu_arches,
            &self.host.cuda_version,
            &self.host.driver_version,
            &self.host.python_abi,
            &self.host.libc_abi,
        ))?;
        let artifact_identity = canonical_hash(artifacts)?;
        let evidence_identity = canonical_hash(evidence)?;
        let plan_identity = canonical_hash(&(
            request.identity.as_str(),
            shape_envelope_identity.as_str(),
            compile_region_identity.as_str(),
            capability_identity.as_str(),
            abi_identity.as_str(),
            artifact_identity.as_str(),
            evidence_identity.as_str(),
        ))?;

        Ok(sock_core::StructuralIdentity {
            request_identity: request.identity.clone(),
            shape_envelope_identity,
            compile_region_identity,
            capability_identity,
            abi_identity,
            artifact_identity,
            evidence_identity,
            plan_identity,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sock_core::{
        BackendPolicy, CachePolicy, ConfigEntry, ConfigLayer, EngineSource, ExecutionTopology,
        FailureMode, GuaranteeTarget, ModelRef, OperatingSystem, RawRequest, RequestedEnvironment,
        ShapePoint, ShapePolicy, ShapeRange, TargetEngine, ValidationStatus, WarmupPolicy,
    };

    fn request() -> RawRequest {
        RawRequest {
            engine: TargetEngine::Vllm,
            model: ModelRef {
                repository: "meta-llama/Llama-3.1-8B-Instruct".to_owned(),
                revision: "main".to_owned(),
            },
            engine_source: EngineSource {
                kind: "vendored".to_owned(),
                revision: "test".to_owned(),
            },
            environment: RequestedEnvironment {
                operating_system: OperatingSystem::Linux,
                accelerator_vendor: AcceleratorVendor::Nvidia,
                gpu_arches: vec!["sm90".to_owned()],
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
                preferred_families: vec![BackendFamily::FlashInfer, BackendFamily::Triton],
                require_prebuilt_artifacts: true,
                allow_runtime_jit: false,
                correctness_target: GuaranteeTarget {
                    level: GuaranteeLevel::ShapeBoundedAot,
                    failure_mode: FailureMode::FailClosed,
                },
                performance_target: GuaranteeTarget {
                    level: GuaranteeLevel::WarmupBounded,
                    failure_mode: FailureMode::FailClosed,
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
                hot_shapes: vec![ShapePoint {
                    batch_size: 1,
                    sequence_length: 128,
                    plane: CoveragePlane::Performance,
                }],
                cuda_graph_shapes: vec![],
            },
            cache_policy: CachePolicy {
                namespace: "prod".to_owned(),
                allow_cross_machine_reuse: false,
            },
            warmup_policy: WarmupPolicy {
                max_warmup_steps: 4,
                verify_cuda_graph_capture: true,
            },
            layered_config: vec![ConfigLayer {
                name: "env".to_owned(),
                precedence: 0,
                entries: vec![ConfigEntry {
                    key: "VLLM_USE_V1".to_owned(),
                    value: "1".to_owned(),
                }],
            }],
        }
    }

    #[test]
    fn planner_produces_verifiable_outcome() {
        let planner = Planner::new(PlannerHostSnapshot {
            operating_system: OperatingSystem::Linux,
            accelerator_vendor: AcceleratorVendor::Nvidia,
            gpu_arches: vec!["sm90".to_owned()],
            cuda_version: "12.4".to_owned(),
            driver_version: "550.54".to_owned(),
            python_abi: "cp311".to_owned(),
            libc_abi: "glibc-2.35".to_owned(),
            flashinfer_prebuilt_available: true,
        });

        let outcome = planner.resolve(request()).expect("planner should succeed");

        assert_eq!(
            outcome.plan.selected_backends.primary.family,
            BackendFamily::FlashInfer
        );
        assert_eq!(outcome.verification.status, ValidationStatus::Passed);
        assert!(
            !outcome
                .plan
                .structural_identity
                .plan_identity
                .as_str()
                .is_empty()
        );
    }
}
