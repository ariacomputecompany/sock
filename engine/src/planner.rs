use std::collections::BTreeMap;

use sock_core::{
    AbiFingerprint, AdapterError, AdapterSurvey, ArtifactAcquisition, ArtifactClass,
    ArtifactClosure, ArtifactManifestEntry, ArtifactPortability, ArtifactRequirement,
    BackendCandidate, BackendExtensionFingerprint, BackendFamily, BackendSelection, CanonicalError,
    CapabilityWitness, CompileRegion, CompileRegionKind, CoveragePlane, CoverageState,
    CoverageWitness, EngineAdapter, ExecutionTopology, GuaranteeDimension, GuaranteeEnvelope,
    GuaranteeEvidence, GuaranteeLevel, HazardClass, LeaderAssignment, MaterializationGraph,
    MaterializationNode, MaterializationNodeKind, MaterializationWave, NormalizedRequest,
    OperatingSystem, PassTrace, PortabilityFingerprint, QueueKind, RangeIntent, RankDisposition,
    RawRequest, ResidualRuntimeRisk, ResolvedBuildPlan, RewritePassContract, RewritePhase,
    ShapeEnvelope, ShapeEnvelopeNode, ShapePoint, ShapeRange, StructuralIdentity, ValidationStatus,
    WarmupObligation, WaveEstimate, canonical_hash,
};
use thiserror::Error;

use crate::vllm_adapter::VllmAdapter;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannerHostSnapshot {
    pub operating_system: OperatingSystem,
    pub accelerator_vendor: sock_core::AcceleratorVendor,
    pub gpu_arches: Vec<String>,
    pub cuda_version: String,
    pub driver_version: String,
    pub python_abi: String,
    pub libc_abi: String,
    pub flashinfer_prebuilt_available: bool,
}

#[derive(Debug)]
pub struct PlanningOutcome {
    pub plan: ResolvedBuildPlan,
    pub closure: ArtifactClosure,
    pub verification: sock_core::VerificationReport,
    pub adapter_survey: AdapterSurvey,
}

#[derive(Debug, Error)]
pub enum PlanError {
    #[error("canonicalization failed: {0}")]
    Canonical(#[from] CanonicalError),
    #[error("adapter survey failed: {0}")]
    Adapter(#[from] AdapterError),
    #[error("planner validation failed: {0}")]
    Validation(String),
}

pub struct Planner {
    host: PlannerHostSnapshot,
}

impl Planner {
    #[must_use]
    pub fn new(host: PlannerHostSnapshot) -> Self {
        Self { host }
    }

    pub fn resolve(&self, raw: RawRequest) -> Result<PlanningOutcome, PlanError> {
        let normalized = raw.normalize()?;
        let adapter_survey = VllmAdapter::default().survey()?;
        let capability_witnesses = self.capability_witnesses(&normalized, &adapter_survey);
        let selected_backends =
            self.select_backends(&normalized, &capability_witnesses, &adapter_survey)?;
        let compile_regions = self.compile_regions(&selected_backends, &adapter_survey);
        let shape_envelope = self.shape_envelope(&normalized, &selected_backends);
        let artifact_requirements =
            self.artifact_requirements(&normalized.topology, &selected_backends, &compile_regions);
        let warmup_obligations =
            self.warmup_obligations(&normalized, &shape_envelope, &compile_regions);
        let residual_risks = self.residual_risks(
            &normalized,
            &shape_envelope,
            &selected_backends,
            &adapter_survey,
        );
        let materialization_graph = self.materialization_graph(
            &normalized.topology,
            &artifact_requirements,
            &warmup_obligations,
        );
        let guarantee_envelope =
            self.guarantee_envelope(&normalized, &shape_envelope, residual_risks.clone());
        let artifact_manifest = artifact_requirements
            .iter()
            .map(|requirement| ArtifactManifestEntry {
                identity: format!(
                    "{}:{:?}:{}",
                    selected_backends.primary.family.as_str(),
                    requirement.class,
                    requirement.scope
                ),
                class: requirement.class,
                backend: requirement.backend,
                scope: requirement.scope.clone(),
            })
            .collect::<Vec<_>>();
        let coverage_witnesses =
            self.coverage_witnesses(&shape_envelope, &artifact_manifest, &residual_risks);
        let guarantee_evidence = GuaranteeEvidence {
            capability_witnesses: capability_witnesses.clone(),
            artifact_manifest: artifact_manifest.clone(),
            warmup_obligations: warmup_obligations.clone(),
            coverage_witnesses,
        };
        let rewrite_trace = self.rewrite_trace(
            &normalized,
            &selected_backends,
            &compile_regions,
            &shape_envelope,
            &artifact_requirements,
            &warmup_obligations,
            &materialization_graph,
            &guarantee_envelope,
            &adapter_survey,
        )?;
        let structural_identity = self.structural_identity(
            &normalized,
            &capability_witnesses,
            &compile_regions,
            &shape_envelope,
            &artifact_requirements,
            &guarantee_evidence,
            &selected_backends,
            &materialization_graph,
            &guarantee_envelope,
        )?;
        let plan = ResolvedBuildPlan {
            normalized_request: normalized,
            selected_backends,
            compile_regions,
            shape_envelope,
            artifact_requirements,
            warmup_obligations,
            materialization_graph,
            guarantee_envelope,
            guarantee_evidence,
            rewrite_trace,
            structural_identity,
        };
        let verification = plan.validate();
        if verification.status == ValidationStatus::Failed {
            return Err(PlanError::Validation(
                verification
                    .issues
                    .iter()
                    .map(|issue| format!("{}: {}", issue.code, issue.message))
                    .collect::<Vec<_>>()
                    .join("; "),
            ));
        }
        let closure = ArtifactClosure {
            plan_identity: plan.structural_identity.plan_identity.clone(),
            artifacts: artifact_manifest,
        };
        Ok(PlanningOutcome {
            plan,
            closure,
            verification,
            adapter_survey,
        })
    }

    fn capability_witnesses(
        &self,
        normalized: &NormalizedRequest,
        adapter_survey: &AdapterSurvey,
    ) -> Vec<CapabilityWitness> {
        let mut witnesses = vec![
            witness(
                "platform.os",
                format!("{:?}", normalized.environment.operating_system),
                "requested-environment",
            ),
            witness(
                "platform.gpu_vendor",
                format!("{:?}", normalized.environment.accelerator_vendor),
                "requested-environment",
            ),
            witness(
                "platform.gpu_arches",
                normalized.environment.gpu_arches.join(","),
                "requested-environment",
            ),
            witness(
                "platform.cuda",
                normalized.environment.cuda_version.clone(),
                "requested-environment",
            ),
            witness(
                "platform.python_abi",
                normalized.environment.python_abi.clone(),
                "requested-environment",
            ),
            witness(
                "engine.revision",
                adapter_survey.engine_revision.clone(),
                "vllm-adapter-survey",
            ),
            witness(
                "adapter.compile_region_count",
                adapter_survey.compile_regions.len().to_string(),
                "vllm-adapter-survey",
            ),
            witness(
                "adapter.residual_surface_count",
                adapter_survey.residual_jit_surfaces.len().to_string(),
                "vllm-adapter-survey",
            ),
        ];
        if self.host.flashinfer_prebuilt_available {
            witnesses.push(witness(
                "flashinfer.prebuilt",
                "available",
                "host-discovery",
            ));
        }
        witnesses.extend(adapter_survey.diagnostics.iter().map(|diagnostic| {
            witness(
                &format!("adapter.diagnostic.{}", slug(&diagnostic.title)),
                format!("{:?}", diagnostic.severity).to_lowercase(),
                "vllm-adapter-survey",
            )
        }));
        witnesses.sort();
        witnesses
    }

    fn select_backends(
        &self,
        normalized: &NormalizedRequest,
        witnesses: &[CapabilityWitness],
        adapter_survey: &AdapterSurvey,
    ) -> Result<BackendSelection, PlanError> {
        let linux_nvidia = normalized.environment.operating_system == OperatingSystem::Linux
            && normalized.environment.accelerator_vendor == sock_core::AcceleratorVendor::Nvidia;
        if !linux_nvidia {
            return Err(PlanError::Validation(
                "V1 planning is locked to vLLM on NVIDIA/Linux".to_owned(),
            ));
        }
        let mut viable = Vec::new();
        for family in &normalized.backend_policy.preferred_families {
            match family {
                BackendFamily::FlashInfer => {
                    if witnesses.iter().any(|w| w.key == "flashinfer.prebuilt") {
                        viable.push(BackendCandidate {
                            family: *family,
                            acquisition: ArtifactAcquisition::VendorPrebuilt,
                            reason: format!(
                                "prebuilt FlashInfer artifact is admissible for this ABI and SM envelope; adapter surfaced {} compile-affecting inputs",
                                adapter_survey.config_inputs.len()
                            ),
                        });
                    } else if !normalized.backend_policy.require_prebuilt_artifacts {
                        viable.push(BackendCandidate {
                            family: *family,
                            acquisition: ArtifactAcquisition::LocalSourceBuild,
                            reason: "FlashInfer remains legal via local build".to_owned(),
                        });
                    }
                }
                BackendFamily::Triton => viable.push(BackendCandidate {
                    family: *family,
                    acquisition: ArtifactAcquisition::LocalAotBuild,
                    reason: "Triton regional compilation is legal for the requested shape envelope"
                        .to_owned(),
                }),
                BackendFamily::AotInductor => viable.push(BackendCandidate {
                    family: *family,
                    acquisition: ArtifactAcquisition::UpstreamCacheBundle,
                    reason: "AoTInductor stable-region reuse is admissible".to_owned(),
                }),
                BackendFamily::CudaGraphs => viable.push(BackendCandidate {
                    family: *family,
                    acquisition: ArtifactAcquisition::LocalAotBuild,
                    reason: "CUDA Graph captures are planned after kernel closure".to_owned(),
                }),
            }
        }
        let primary = viable.first().cloned().ok_or_else(|| {
            PlanError::Validation(
                "no admissible backend remained after legality filtering".to_owned(),
            )
        })?;
        Ok(BackendSelection {
            primary,
            secondary: viable.into_iter().skip(1).collect(),
        })
    }

    fn compile_regions(
        &self,
        selected_backends: &BackendSelection,
        adapter_survey: &AdapterSurvey,
    ) -> Vec<CompileRegion> {
        let mut regions = adapter_survey
            .compile_regions
            .iter()
            .filter(|region| region.regional_compile_candidate)
            .map(|region| CompileRegion {
                name: canonical_region_name(region.kind),
                kind: region.kind,
                family: region_family(region.kind, selected_backends.primary.family),
                reusable: region.repeated || region.regional_compile_candidate,
                regional_compile_candidate: region.regional_compile_candidate,
                boundaries: region.boundaries.clone(),
                rationale: region.rationale.clone(),
                invalidation_domain: region.name.clone(),
                shape_planes: region_shape_planes(region.kind),
                evidence: region.evidence.clone(),
            })
            .collect::<Vec<_>>();
        regions.sort();
        regions
    }

    fn shape_envelope(
        &self,
        normalized: &NormalizedRequest,
        selected_backends: &BackendSelection,
    ) -> ShapeEnvelope {
        let mut nodes = vec![
            ShapeEnvelopeNode {
                name: "correctness-range".to_owned(),
                plane: CoveragePlane::Correctness,
                intent: RangeIntent::SymbolicRange,
                range: normalized.shape_policy.correctness_range.clone(),
                exact_shape: None,
                required_backends: vec![selected_backends.primary.family],
            },
            ShapeEnvelopeNode {
                name: "performance-range".to_owned(),
                plane: CoveragePlane::Performance,
                intent: RangeIntent::FallbackRange,
                range: normalized.shape_policy.performance_range.clone(),
                exact_shape: None,
                required_backends: vec![selected_backends.primary.family],
            },
        ];
        for hot in &normalized.shape_policy.hot_shapes {
            nodes.push(hot_shape_node("hot", hot, selected_backends.primary.family));
        }
        for graph in &normalized.shape_policy.cuda_graph_shapes {
            nodes.push(hot_shape_node(
                "cuda-graph",
                graph,
                BackendFamily::CudaGraphs,
            ));
        }
        if normalized.backend_policy.allow_runtime_jit {
            nodes.push(ShapeEnvelopeNode {
                name: "residual-dynamic-tail".to_owned(),
                plane: CoveragePlane::Performance,
                intent: RangeIntent::UncoveredResidual,
                range: normalized.shape_policy.correctness_range.clone(),
                exact_shape: None,
                required_backends: vec![selected_backends.primary.family],
            });
        }
        nodes.sort();
        ShapeEnvelope { nodes }
    }

    fn artifact_requirements(
        &self,
        topology: &ExecutionTopology,
        selected_backends: &BackendSelection,
        compile_regions: &[CompileRegion],
    ) -> Vec<ArtifactRequirement> {
        let mut requirements = compile_regions
            .iter()
            .flat_map(|region| {
                [
                    ArtifactRequirement {
                        class: ArtifactClass::CompiledGraph,
                        backend: region.family,
                        acquisition: selected_backends.primary.acquisition,
                        scope: region.name.clone(),
                        portability: ArtifactPortability::GpuArchitectureFamilyPortable,
                        rank_disposition: RankDisposition::Shared,
                        expected_bytes: Some(24_000_000),
                        expected_compile_ms: Some(2_500),
                        expected_transfer_ms: Some(if total_rank_count(topology) > 1 {
                            220
                        } else {
                            0
                        }),
                    },
                    ArtifactRequirement {
                        class: ArtifactClass::TritonBinary,
                        backend: region.family,
                        acquisition: selected_backends.primary.acquisition,
                        scope: region.name.clone(),
                        portability: ArtifactPortability::AbiClusterPortable,
                        rank_disposition: RankDisposition::Shared,
                        expected_bytes: Some(4_000_000),
                        expected_compile_ms: Some(800),
                        expected_transfer_ms: Some(if total_rank_count(topology) > 1 {
                            90
                        } else {
                            0
                        }),
                    },
                ]
            })
            .collect::<Vec<_>>();
        requirements.push(ArtifactRequirement {
            class: ArtifactClass::CudaGraphCapture,
            backend: BackendFamily::CudaGraphs,
            acquisition: ArtifactAcquisition::LocalAotBuild,
            scope: "decode_attention".to_owned(),
            portability: ArtifactPortability::TopologyScoped,
            rank_disposition: RankDisposition::RankLocal,
            expected_bytes: Some(2_000_000),
            expected_compile_ms: Some(300),
            expected_transfer_ms: Some(25),
        });
        requirements.sort();
        requirements
    }

    fn warmup_obligations(
        &self,
        normalized: &NormalizedRequest,
        shape_envelope: &ShapeEnvelope,
        compile_regions: &[CompileRegion],
    ) -> Vec<WarmupObligation> {
        let all_ranks = total_ranks(&normalized.topology);
        let mut obligations = Vec::new();
        for node in &shape_envelope.nodes {
            if node.intent == RangeIntent::UncoveredResidual {
                continue;
            }
            let steps = match node.intent {
                RangeIntent::ExactHotShape => 1,
                RangeIntent::SymbolicRange => normalized.warmup_policy.max_warmup_steps.min(3),
                RangeIntent::FallbackRange => normalized.warmup_policy.max_warmup_steps.min(2),
                RangeIntent::UncoveredResidual => 0,
            };
            for region in compile_regions {
                if region.shape_planes.contains(&node.plane)
                    || node.plane == CoveragePlane::Correctness
                {
                    obligations.push(WarmupObligation {
                        node_name: node.name.clone(),
                        region_name: region.name.clone(),
                        step_count: steps,
                        plane: node.plane,
                        blocking: node.plane == CoveragePlane::Correctness,
                        required_artifacts: vec![region.name.clone()],
                        rank_scope: all_ranks.clone(),
                        requires_capture: node.plane == CoveragePlane::CudaGraph,
                        requires_autotune: region.family == BackendFamily::FlashInfer,
                    });
                }
            }
        }
        obligations.sort();
        obligations
    }

    fn residual_risks(
        &self,
        normalized: &NormalizedRequest,
        shape_envelope: &ShapeEnvelope,
        selected_backends: &BackendSelection,
        adapter_survey: &AdapterSurvey,
    ) -> Vec<ResidualRuntimeRisk> {
        let mut risks = adapter_survey
            .residual_jit_surfaces
            .iter()
            .filter(|surface| surface_applies(surface.backend_family.as_str(), selected_backends))
            .map(|surface| ResidualRuntimeRisk {
                class: HazardClass::ResidualLazyCompile,
                summary: format!("{}: {}", surface.name, surface.warmup_gap),
                bounded_to: Some(surface.mitigation.clone()),
            })
            .collect::<Vec<_>>();
        if shape_envelope
            .nodes
            .iter()
            .any(|node| node.intent == RangeIntent::UncoveredResidual)
        {
            risks.push(ResidualRuntimeRisk {
                class: HazardClass::ResidualLazyCompile,
                summary: format!(
                    "{} retains a bounded dynamic tail outside the compiled hot-shape set",
                    selected_backends.primary.family.as_str()
                ),
                bounded_to: Some("residual-dynamic-tail".to_owned()),
            });
        }
        if normalized.cache_policy.allow_cross_machine_reuse {
            risks.push(ResidualRuntimeRisk {
                class: HazardClass::StaleCache,
                summary: "cross-machine cache reuse must be bounded by ABI-cluster identity"
                    .to_owned(),
                bounded_to: None,
            });
        }
        risks
    }

    fn materialization_graph(
        &self,
        topology: &ExecutionTopology,
        artifact_requirements: &[ArtifactRequirement],
        warmup_obligations: &[WarmupObligation],
    ) -> MaterializationGraph {
        let mut nodes = Vec::new();
        let rank_zero = vec![0];
        for requirement in artifact_requirements {
            let queue = match requirement.acquisition {
                ArtifactAcquisition::VendorPrebuilt | ArtifactAcquisition::UpstreamCacheBundle => {
                    QueueKind::ArtifactIo
                }
                ArtifactAcquisition::LocalAotBuild | ArtifactAcquisition::LocalSourceBuild => {
                    QueueKind::Compile
                }
            };
            let kind = match requirement.acquisition {
                ArtifactAcquisition::VendorPrebuilt | ArtifactAcquisition::UpstreamCacheBundle => {
                    MaterializationNodeKind::Materialize
                }
                ArtifactAcquisition::LocalAotBuild | ArtifactAcquisition::LocalSourceBuild => {
                    MaterializationNodeKind::Compile
                }
            };
            let plane = if requirement.class == ArtifactClass::CudaGraphCapture {
                CoveragePlane::CudaGraph
            } else {
                CoveragePlane::Performance
            };
            nodes.push(MaterializationNode {
                name: format!("artifact:{}", requirement.scope),
                wave: if requirement.class == ArtifactClass::CudaGraphCapture {
                    2
                } else {
                    1
                },
                kind,
                queue,
                plane,
                dependency_nodes: Vec::new(),
                consumes: Vec::new(),
                produces: vec![requirement.scope.clone()],
                rank_scope: rank_zero.clone(),
                invalidation_domain: requirement.scope.clone(),
                replay_boundary: format!("artifact:{}", requirement.scope),
                expected_compile_ms: requirement.expected_compile_ms,
                expected_bytes_written: requirement.expected_bytes,
                expected_transfer_ms: requirement.expected_transfer_ms,
                residual_jit_risk_removed: 1,
            });
            if total_rank_count(topology) > 1
                && requirement.rank_disposition == RankDisposition::Shared
            {
                nodes.push(MaterializationNode {
                    name: format!("fanout:{}", requirement.scope),
                    wave: 2,
                    kind: MaterializationNodeKind::Transfer,
                    queue: QueueKind::ArtifactIo,
                    plane,
                    dependency_nodes: vec![format!("artifact:{}", requirement.scope)],
                    consumes: vec![requirement.scope.clone()],
                    produces: vec![format!("distributed:{}", requirement.scope)],
                    rank_scope: total_ranks(topology),
                    invalidation_domain: requirement.scope.clone(),
                    replay_boundary: format!("fanout:{}", requirement.scope),
                    expected_compile_ms: Some(0),
                    expected_bytes_written: requirement.expected_bytes,
                    expected_transfer_ms: requirement.expected_transfer_ms,
                    residual_jit_risk_removed: 1,
                });
            }
        }
        for obligation in warmup_obligations {
            nodes.push(MaterializationNode {
                name: format!("warmup:{}:{}", obligation.node_name, obligation.region_name),
                wave: if obligation.blocking { 3 } else { 4 },
                kind: MaterializationNodeKind::Warmup,
                queue: QueueKind::Warmup,
                plane: obligation.plane,
                dependency_nodes: obligation
                    .required_artifacts
                    .iter()
                    .map(|artifact| format!("artifact:{artifact}"))
                    .collect(),
                consumes: obligation.required_artifacts.clone(),
                produces: vec![format!(
                    "coverage:{}:{}",
                    obligation.node_name, obligation.region_name
                )],
                rank_scope: obligation.rank_scope.clone(),
                invalidation_domain: obligation.node_name.clone(),
                replay_boundary: format!("warmup:{}", obligation.node_name),
                expected_compile_ms: Some(0),
                expected_bytes_written: Some(0),
                expected_transfer_ms: Some(0),
                residual_jit_risk_removed: 1,
            });
        }
        let leader_assignments = artifact_requirements
            .iter()
            .filter(|requirement| {
                requirement.rank_disposition == RankDisposition::Shared
                    && total_rank_count(topology) > 1
            })
            .map(|requirement| LeaderAssignment {
                artifact_scope: requirement.scope.clone(),
                leader_rank: 0,
                follower_ranks: (1..total_rank_count(topology)).collect(),
            })
            .collect::<Vec<_>>();
        let waves = build_waves(&nodes);
        MaterializationGraph {
            nodes,
            waves,
            leader_assignments,
            early_serve_frontier: vec!["correctness-range".to_owned()],
            late_bindings: vec![(
                "performance".to_owned(),
                "strict-no-surprise-jit".to_owned(),
            )],
        }
    }

    fn guarantee_envelope(
        &self,
        normalized: &NormalizedRequest,
        shape_envelope: &ShapeEnvelope,
        residual_risks: Vec<ResidualRuntimeRisk>,
    ) -> GuaranteeEnvelope {
        let has_residual = shape_envelope
            .nodes
            .iter()
            .any(|node| node.intent == RangeIntent::UncoveredResidual);
        let achieved_correctness = if has_residual {
            GuaranteeLevel::WarmupBounded
        } else {
            GuaranteeLevel::ShapeBoundedAot
        };
        let achieved_performance =
            if !has_residual && normalized.warmup_policy.verify_cuda_graph_capture {
                GuaranteeLevel::StrictNoSurpriseJit
            } else if !has_residual {
                GuaranteeLevel::ShapeBoundedAot
            } else {
                GuaranteeLevel::WarmupBounded
            };
        GuaranteeEnvelope {
            requested_correctness: normalized.backend_policy.correctness_target.clone(),
            requested_performance: normalized.backend_policy.performance_target.clone(),
            achieved_correctness,
            achieved_performance,
            covered_dimensions: vec![
                GuaranteeDimension::Environment,
                GuaranteeDimension::Kernel,
                GuaranteeDimension::Shape,
                GuaranteeDimension::Runtime,
                GuaranteeDimension::Topology,
            ],
            covered_shapes: shape_envelope
                .nodes
                .iter()
                .map(|node| node.name.clone())
                .collect(),
            residual_risks,
        }
    }

    fn coverage_witnesses(
        &self,
        shape_envelope: &ShapeEnvelope,
        artifact_manifest: &[ArtifactManifestEntry],
        residual_risks: &[ResidualRuntimeRisk],
    ) -> Vec<CoverageWitness> {
        shape_envelope
            .nodes
            .iter()
            .map(|node| CoverageWitness {
                plane: node.plane,
                node_name: node.name.clone(),
                evidence: format!(
                    "coverage derived from canonical warmup obligations for {}",
                    node.name
                ),
                coverage_states: coverage_states_for_plane(node.plane),
                artifact_scopes: artifact_manifest
                    .iter()
                    .map(|artifact| artifact.scope.clone())
                    .collect(),
                uncovered_residuals: residual_risks
                    .iter()
                    .filter_map(|risk| risk.bounded_to.clone())
                    .collect(),
            })
            .collect()
    }

    fn rewrite_trace(
        &self,
        normalized: &NormalizedRequest,
        selected_backends: &BackendSelection,
        compile_regions: &[CompileRegion],
        shape_envelope: &ShapeEnvelope,
        artifact_requirements: &[ArtifactRequirement],
        warmup_obligations: &[WarmupObligation],
        materialization_graph: &MaterializationGraph,
        guarantee_envelope: &GuaranteeEnvelope,
        adapter_survey: &AdapterSurvey,
    ) -> Result<Vec<PassTrace>, CanonicalError> {
        let request_id = normalized.identity.to_string();
        let backend_id = canonical_hash(selected_backends)?.to_string();
        let region_id = canonical_hash(compile_regions)?.to_string();
        let shape_id = canonical_hash(shape_envelope)?.to_string();
        let artifact_id = canonical_hash(artifact_requirements)?.to_string();
        let warmup_id = canonical_hash(warmup_obligations)?.to_string();
        let wave_id = canonical_hash(materialization_graph)?.to_string();
        let guarantee_id = canonical_hash(guarantee_envelope)?.to_string();
        Ok(vec![
            pass(
                RewritePassContract::new(
                    "parse-cleanup",
                    RewritePhase::NormalizeRequest,
                    vec!["raw-request"],
                    vec!["normalized-request"],
                    vec!["request-canonicalization"],
                ),
                &request_id,
                &request_id,
                vec!["raw-request-ingested"],
                vec![],
                vec![],
                vec!["request-canonicalization"],
            ),
            pass(
                RewritePassContract::new(
                    "layered-config-normalization",
                    RewritePhase::NormalizeRequest,
                    vec!["normalized-request", "layered-config"],
                    vec!["normalized-config"],
                    vec!["request-canonicalization", "config-precedence-order"],
                ),
                &request_id,
                &request_id,
                vec!["config-layers-sorted", "duplicate-entries-removed"],
                vec![],
                vec![],
                vec!["request-canonicalization", "config-precedence-order"],
            ),
            pass(
                RewritePassContract::new(
                    "survey-vllm-surface",
                    RewritePhase::SurveyEngine,
                    vec!["normalized-request", "vendored-vllm-source"],
                    vec!["adapter-survey"],
                    vec!["source-anchored-engine-seam"],
                ),
                &request_id,
                &request_id,
                vec![
                    "config-input-extraction",
                    "compile-region-mining",
                    "residual-jit-surface-mining",
                ],
                vec![],
                vec![&format!(
                    "adapter-survey:{}-regions-{}-residuals",
                    adapter_survey.compile_regions.len(),
                    adapter_survey.residual_jit_surfaces.len()
                )],
                vec!["source-anchored-engine-seam"],
            ),
            pass(
                RewritePassContract::new(
                    "backend-legality-filtering",
                    RewritePhase::SelectBackends,
                    vec!["adapter-survey", "capability-witnesses", "backend-policy"],
                    vec!["backend-selection"],
                    vec!["fail-closed-legality"],
                ),
                &request_id,
                &backend_id,
                vec!["v1-linux-nvidia-constraint", "prebuilt-first-resolution"],
                vec![],
                vec!["illegal-backends-pruned"],
                vec!["fail-closed-legality"],
            ),
            pass(
                RewritePassContract::new(
                    "compile-region-discovery",
                    RewritePhase::DiscoverCompileRegions,
                    vec!["backend-selection", "adapter-survey"],
                    vec!["compile-regions"],
                    vec!["region-source-evidence"],
                ),
                &backend_id,
                &region_id,
                vec!["regional-transformer-segmentation"],
                vec![],
                vec![],
                vec!["region-source-evidence"],
            ),
            pass(
                RewritePassContract::new(
                    "shape-envelope-lattice-construction",
                    RewritePhase::BuildShapeEnvelope,
                    vec!["compile-regions", "shape-policy"],
                    vec!["shape-envelope"],
                    vec!["correctness-range-preserved", "hot-shape-bounds-preserved"],
                ),
                &region_id,
                &shape_id,
                vec!["hot-shape-specialization", "range-node-deduplication"],
                vec![],
                vec![],
                vec!["correctness-range-preserved", "hot-shape-bounds-preserved"],
            ),
            pass(
                RewritePassContract::new(
                    "warmup-coverage-elaboration",
                    RewritePhase::ElaborateWarmup,
                    vec!["shape-envelope", "compile-regions", "warmup-policy"],
                    vec!["warmup-obligations"],
                    vec!["non-residual-shapes-covered"],
                ),
                &shape_id,
                &warmup_id,
                vec!["warmup-obligations-attached"],
                vec![],
                vec![],
                vec!["non-residual-shapes-covered"],
            ),
            pass(
                RewritePassContract::new(
                    "materialization-wave-planning",
                    RewritePhase::PlanMaterialization,
                    vec!["artifact-requirements", "warmup-obligations", "topology"],
                    vec!["materialization-graph"],
                    vec!["dependency-order-preserved", "replay-boundaries-assigned"],
                ),
                &warmup_id,
                &wave_id,
                vec!["phase-local-cache-skipping"],
                vec![],
                vec![],
                vec!["dependency-order-preserved", "replay-boundaries-assigned"],
            ),
            pass(
                RewritePassContract::new(
                    "guarantee-envelope-shaping",
                    RewritePhase::ShapeGuarantees,
                    vec!["materialization-graph", "shape-envelope", "residual-risks"],
                    vec!["guarantee-envelope"],
                    vec!["correctness-performance-split"],
                ),
                &wave_id,
                &guarantee_id,
                vec!["correctness-performance-split"],
                vec![],
                vec![],
                vec!["correctness-performance-split"],
            ),
            pass(
                RewritePassContract::new(
                    "artifact-emission-shaping",
                    RewritePhase::EmitArtifacts,
                    vec!["guarantee-envelope", "artifact-requirements"],
                    vec!["artifact-closure"],
                    vec!["closure-manifest-derived-from-plan"],
                ),
                &guarantee_id,
                &artifact_id,
                vec!["closure-manifest-derived-from-plan"],
                vec![],
                vec![],
                vec!["closure-manifest-derived-from-plan"],
            ),
        ])
    }

    fn structural_identity(
        &self,
        normalized: &NormalizedRequest,
        capability_witnesses: &[CapabilityWitness],
        compile_regions: &[CompileRegion],
        shape_envelope: &ShapeEnvelope,
        artifact_requirements: &[ArtifactRequirement],
        guarantee_evidence: &GuaranteeEvidence,
        selected_backends: &BackendSelection,
        materialization_graph: &MaterializationGraph,
        guarantee_envelope: &GuaranteeEnvelope,
    ) -> Result<StructuralIdentity, CanonicalError> {
        let request_identity = normalized.identity.clone();
        let shape_envelope_identity = canonical_hash(shape_envelope)?;
        let compile_region_identity = canonical_hash(compile_regions)?;
        let capability_identity = canonical_hash(capability_witnesses)?;
        let abi_identity = canonical_hash(&AbiFingerprint {
            operating_system: normalized.environment.operating_system,
            accelerator_vendor: normalized.environment.accelerator_vendor,
            gpu_arches: normalized.environment.gpu_arches.clone(),
            cuda_version: normalized.environment.cuda_version.clone(),
            driver_version: normalized.environment.driver_version.clone(),
            python_abi: normalized.environment.python_abi.clone(),
            libc_abi: normalized.environment.libc_abi.clone(),
            topology: normalized.topology.clone(),
        })?;
        let backend_extension_identity = canonical_hash(&BackendExtensionFingerprint::from_plan(
            selected_backends,
            compile_regions,
        ))?;
        let portability_identity = canonical_hash(&PortabilityFingerprint::from_plan(
            normalized.cache_policy.namespace.clone(),
            normalized.cache_policy.allow_cross_machine_reuse,
            normalized.topology.clone(),
            artifact_requirements,
        ))?;
        let artifact_identity = canonical_hash(artifact_requirements)?;
        let evidence_identity = canonical_hash(guarantee_evidence)?;
        let plan_identity = canonical_hash(&(
            selected_backends,
            compile_regions,
            shape_envelope,
            artifact_requirements,
            materialization_graph,
            guarantee_envelope,
            &request_identity,
            &capability_identity,
            &abi_identity,
            &backend_extension_identity,
            &portability_identity,
        ))?;
        Ok(StructuralIdentity {
            request_identity,
            shape_envelope_identity,
            compile_region_identity,
            capability_identity,
            abi_identity,
            backend_extension_identity,
            portability_identity,
            artifact_identity,
            evidence_identity,
            plan_identity,
        })
    }
}

fn witness(key: &str, value: impl Into<String>, provenance: &str) -> CapabilityWitness {
    CapabilityWitness {
        key: key.to_owned(),
        value: value.into(),
        provenance: provenance.to_owned(),
    }
}

fn hot_shape_node(prefix: &str, shape: &ShapePoint, backend: BackendFamily) -> ShapeEnvelopeNode {
    ShapeEnvelopeNode {
        name: format!("{prefix}-b{}-s{}", shape.batch_size, shape.sequence_length),
        plane: shape.plane,
        intent: RangeIntent::ExactHotShape,
        range: ShapeRange {
            min_batch_size: shape.batch_size,
            max_batch_size: shape.batch_size,
            min_sequence_length: shape.sequence_length,
            max_sequence_length: shape.sequence_length,
        },
        exact_shape: Some(shape.clone()),
        required_backends: vec![backend],
    }
}

fn canonical_region_name(kind: CompileRegionKind) -> String {
    match kind {
        CompileRegionKind::RepeatedTransformerBlockBody => "transformer_block_body".to_owned(),
        CompileRegionKind::DecodeMicrograph => "decode_attention".to_owned(),
        CompileRegionKind::PrefillMicrograph => "prefill_attention".to_owned(),
        CompileRegionKind::AttentionKvBoundary => "kv_cache_update".to_owned(),
        CompileRegionKind::MoeSpecialtyPath => "moe_specialty_path".to_owned(),
    }
}

fn region_family(kind: CompileRegionKind, primary: BackendFamily) -> BackendFamily {
    match kind {
        CompileRegionKind::DecodeMicrograph => BackendFamily::CudaGraphs,
        CompileRegionKind::MoeSpecialtyPath => BackendFamily::AotInductor,
        CompileRegionKind::RepeatedTransformerBlockBody
        | CompileRegionKind::PrefillMicrograph
        | CompileRegionKind::AttentionKvBoundary => primary,
    }
}

fn region_shape_planes(kind: CompileRegionKind) -> Vec<CoveragePlane> {
    match kind {
        CompileRegionKind::RepeatedTransformerBlockBody => {
            vec![CoveragePlane::Correctness, CoveragePlane::Performance]
        }
        CompileRegionKind::DecodeMicrograph => vec![
            CoveragePlane::Correctness,
            CoveragePlane::Performance,
            CoveragePlane::CudaGraph,
        ],
        CompileRegionKind::PrefillMicrograph => {
            vec![CoveragePlane::Correctness, CoveragePlane::Performance]
        }
        CompileRegionKind::AttentionKvBoundary => vec![
            CoveragePlane::Correctness,
            CoveragePlane::BackendSpecialization,
        ],
        CompileRegionKind::MoeSpecialtyPath => vec![CoveragePlane::BackendSpecialization],
    }
}

fn surface_applies(backend_family: &str, selected_backends: &BackendSelection) -> bool {
    match backend_family {
        "flashinfer" => selected_backends.primary.family == BackendFamily::FlashInfer,
        "triton" => {
            selected_backends.primary.family == BackendFamily::Triton
                || selected_backends
                    .secondary
                    .iter()
                    .any(|candidate| candidate.family == BackendFamily::Triton)
        }
        "inductor" => {
            selected_backends.primary.family == BackendFamily::AotInductor
                || selected_backends
                    .secondary
                    .iter()
                    .any(|candidate| candidate.family == BackendFamily::AotInductor)
        }
        "torch.compile" => true,
        _ => false,
    }
}

fn slug(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}

fn total_rank_count(topology: &ExecutionTopology) -> u16 {
    topology
        .tensor_parallelism
        .saturating_mul(topology.pipeline_parallelism)
        .max(1)
}

fn total_ranks(topology: &ExecutionTopology) -> Vec<u16> {
    (0..total_rank_count(topology)).collect()
}

fn coverage_states_for_plane(plane: CoveragePlane) -> Vec<CoverageState> {
    match plane {
        CoveragePlane::Correctness => vec![CoverageState::Compiled, CoverageState::Executed],
        CoveragePlane::Performance => vec![
            CoverageState::Compiled,
            CoverageState::Executed,
            CoverageState::VerifiedNoNewCompile,
        ],
        CoveragePlane::CudaGraph => vec![CoverageState::Compiled, CoverageState::Captured],
        CoveragePlane::BackendSpecialization => {
            vec![CoverageState::Compiled, CoverageState::Autotuned]
        }
    }
}

fn build_waves(nodes: &[MaterializationNode]) -> Vec<MaterializationWave> {
    let mut buckets: BTreeMap<(u32, QueueKind), Vec<&MaterializationNode>> = BTreeMap::new();
    for node in nodes {
        buckets
            .entry((node.wave, node.queue))
            .or_default()
            .push(node);
    }
    buckets
        .into_iter()
        .map(|((wave, queue), members)| MaterializationWave {
            name: format!("wave-{wave}-{queue:?}"),
            queue,
            node_names: members.iter().map(|node| node.name.clone()).collect(),
            estimate: WaveEstimate {
                expected_compile_ms: Some(
                    members
                        .iter()
                        .filter_map(|node| node.expected_compile_ms)
                        .sum(),
                ),
                expected_bytes_written: Some(
                    members
                        .iter()
                        .filter_map(|node| node.expected_bytes_written)
                        .sum(),
                ),
                expected_transfer_ms: Some(
                    members
                        .iter()
                        .filter_map(|node| node.expected_transfer_ms)
                        .sum(),
                ),
                fanout_count: members
                    .iter()
                    .flat_map(|node| node.rank_scope.iter().copied())
                    .max()
                    .map_or(0, |rank| rank.saturating_add(1)),
                residual_jit_risk_removed: members
                    .iter()
                    .map(|node| node.residual_jit_risk_removed)
                    .sum(),
            },
            hazard_repairs: Vec::new(),
        })
        .collect()
}

fn pass(
    contract: RewritePassContract,
    before: &str,
    after: &str,
    matched_rules: Vec<&str>,
    repairs: Vec<&str>,
    invalidated_assumptions: Vec<&str>,
    validated_invariants: Vec<&str>,
) -> PassTrace {
    PassTrace {
        contract,
        before_identity: before.to_owned(),
        after_identity: after.to_owned(),
        matched_rules: matched_rules.into_iter().map(str::to_owned).collect(),
        repairs: repairs.into_iter().map(str::to_owned).collect(),
        invalidated_assumptions: invalidated_assumptions
            .into_iter()
            .map(str::to_owned)
            .collect(),
        validated_invariants: validated_invariants
            .into_iter()
            .map(str::to_owned)
            .collect(),
        violations: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sock_core::{
        AcceleratorVendor, BackendPolicy, CachePolicy, ConfigEntry, ConfigLayer, EngineSource,
        FailureMode, GuaranteeLevel, GuaranteeTarget, ModelRef, OperatingSystem,
        RequestedEnvironment, ShapePoint, ShapePolicy, ShapeRange, TargetEngine, WarmupPolicy,
    };

    fn host() -> PlannerHostSnapshot {
        PlannerHostSnapshot {
            operating_system: OperatingSystem::Linux,
            accelerator_vendor: AcceleratorVendor::Nvidia,
            gpu_arches: vec!["sm90".to_owned()],
            cuda_version: "12.4".to_owned(),
            driver_version: "550.54".to_owned(),
            python_abi: "cp311".to_owned(),
            libc_abi: "glibc-2.35".to_owned(),
            flashinfer_prebuilt_available: true,
        }
    }

    fn request() -> RawRequest {
        RawRequest {
            engine: TargetEngine::Vllm,
            model: ModelRef {
                repository: "meta-llama/Llama-3.1-8B-Instruct".to_owned(),
                revision: "main".to_owned(),
            },
            engine_source: EngineSource {
                kind: "vendored".to_owned(),
                revision: crate::vllm::revision().to_owned(),
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
                preferred_families: vec![
                    BackendFamily::FlashInfer,
                    BackendFamily::Triton,
                    BackendFamily::CudaGraphs,
                ],
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
    fn portability_policy_changes_plan_identity() {
        let planner = Planner::new(host());
        let baseline = planner.resolve(request()).expect("baseline plan");

        let mut changed_request = request();
        changed_request.cache_policy.allow_cross_machine_reuse = true;
        let changed = planner.resolve(changed_request).expect("changed plan");

        assert_ne!(
            baseline.plan.structural_identity.portability_identity,
            changed.plan.structural_identity.portability_identity
        );
        assert_ne!(
            baseline.plan.structural_identity.plan_identity,
            changed.plan.structural_identity.plan_identity
        );
    }

    #[test]
    fn rewrite_trace_contracts_validate() {
        let planner = Planner::new(host());
        let outcome = planner.resolve(request()).expect("plan");

        assert!(!outcome.plan.rewrite_trace.is_empty());
        for pass in &outcome.plan.rewrite_trace {
            pass.validate().expect("rewrite pass contract");
        }
    }
}
