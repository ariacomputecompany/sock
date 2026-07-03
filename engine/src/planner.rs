use std::collections::BTreeMap;

use sock_core::{
    AbiFingerprint, AdapterBackendBinding, AdapterError, AdapterSurvey, AdmissibilityVerdict,
    ArtifactAcquisition, ArtifactAdmissibilityProof, ArtifactClass, ArtifactClosure,
    ArtifactManifestEntry, ArtifactPortability, ArtifactRequirement, BackendAdmissibilityProof,
    BackendCandidate, BackendCapability, BackendCapabilityRegistry, BackendExtensionFingerprint,
    BackendFamily, BackendSelection, CanonicalError, CapabilityProvenance, CapabilityWitness,
    CompileRegion, CoveragePlane, CoverageState, CoverageWitness, EngineAdapter, ExecutionTopology,
    FanoutStrategy, GuaranteeDimension, GuaranteeEnvelope, GuaranteeEvidence, GuaranteeLevel,
    HazardClass, LeaderAssignment, MaterializationGraph, MaterializationNode,
    MaterializationNodeKind, MaterializationWave, NodeExecutionContract, NormalizedRequest,
    OperatingSystem, OptimizationEnvelope, PackagingStrategy, PassTrace, PortabilityFingerprint,
    QueueDiscipline, QueueKind, RangeIntent, RankDisposition, RawRequest, ResidualRuntimeRisk,
    ResolvedBuildPlan, RewritePassContract, RewritePhase, RuntimeJitDisposition,
    RuntimeJitEvidence, RuntimeRoi, ServePhase, ShapeEnvelope, ShapeEnvelopeNode, ShapePoint,
    ShapeRange, StructuralIdentity, ValidationStatus, WarmupContradiction, WarmupCoverageProof,
    WarmupObligation, WaveEstimate, WaveExecutionContract, artifact_manifest_identity,
    artifact_node_handle, canonical_hash, fanout_node_handle,
};
use thiserror::Error;

use crate::{BuildReadiness, BuildScope, vllm_adapter::VllmAdapter};

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
        self.resolve_scoped(raw, &BuildScope::default())
    }

    pub fn resolve_scoped(
        &self,
        raw: RawRequest,
        scope: &BuildScope,
    ) -> Result<PlanningOutcome, PlanError> {
        let normalized = raw.normalize()?;
        let optimization_envelope =
            OptimizationEnvelope::from_level(normalized.optimization_policy.level);
        let adapter_survey = VllmAdapter::default().survey()?;
        let capability_witnesses = self.capability_witnesses(&normalized, &adapter_survey);
        let backend_registry = self.backend_registry(&normalized);
        let selected_backends =
            self.select_backends(&normalized, &capability_witnesses, &backend_registry)?;
        let selected_backends = if optimization_envelope.compile_secondary_backends {
            selected_backends
        } else {
            BackendSelection {
                primary: selected_backends.primary,
                secondary: Vec::new(),
            }
        };
        let compile_regions = self.compile_regions(&selected_backends, &adapter_survey, scope);
        if compile_regions.is_empty() {
            return Err(PlanError::Validation(
                "scoped build request resolved to an empty compile-region closure".to_owned(),
            ));
        }
        let selected_backends = self.expand_selected_backends(
            &normalized,
            &capability_witnesses,
            &backend_registry,
            selected_backends,
            &compile_regions,
        )?;
        let shape_envelope = scoped_shape_envelope(
            self.shape_envelope(&normalized, &selected_backends, &optimization_envelope),
            &compile_regions,
            scope,
        );
        let artifact_requirements = self.artifact_requirements(
            &normalized,
            &capability_witnesses,
            &shape_envelope,
            &selected_backends,
            &compile_regions,
            &adapter_survey,
            &optimization_envelope,
            scope,
        )?;
        let warmup_obligations = self.warmup_obligations(
            &normalized,
            &shape_envelope,
            &compile_regions,
            &optimization_envelope,
            scope,
        );
        let residual_risks = self.residual_risks(
            &normalized,
            &shape_envelope,
            &selected_backends,
            &adapter_survey,
            &compile_regions,
            scope,
        );
        let materialization_graph = self.materialization_graph(
            &normalized.topology,
            &artifact_requirements,
            &warmup_obligations,
        )?;
        let guarantee_envelope =
            self.guarantee_envelope(&normalized, &shape_envelope, residual_risks.clone());
        let artifact_manifest = artifact_requirements
            .iter()
            .map(|requirement| ArtifactManifestEntry {
                identity: artifact_manifest_identity(selected_backends.primary.family, requirement),
                class: requirement.class,
                backend: requirement.backend,
                scope: requirement.scope.clone(),
                admissibility: requirement.admissibility.clone(),
            })
            .collect::<Vec<_>>();
        let coverage_witnesses =
            self.coverage_witnesses(&shape_envelope, &artifact_manifest, &residual_risks);
        let runtime_jit_evidence = self.runtime_jit_evidence(
            &normalized,
            &selected_backends,
            &compile_regions,
            &warmup_obligations,
            &artifact_requirements,
            &adapter_survey,
        );
        let guarantee_evidence = GuaranteeEvidence {
            capability_witnesses: capability_witnesses.clone(),
            artifact_manifest: artifact_manifest.clone(),
            warmup_obligations: warmup_obligations.clone(),
            coverage_witnesses,
            runtime_jit_evidence,
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
            &backend_registry,
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
            requested_readiness: scope.readiness.map(|readiness| match readiness {
                BuildReadiness::EarlyServe => "early_serve".to_owned(),
                BuildReadiness::Correctness => "correctness".to_owned(),
                BuildReadiness::Performance => "performance".to_owned(),
            }),
            optimization_envelope,
            backend_registry,
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

    fn backend_registry(&self, normalized: &NormalizedRequest) -> BackendCapabilityRegistry {
        let mut entries = vec![
            BackendCapability {
                family: BackendFamily::FlashInfer,
                supported_operating_systems: vec![OperatingSystem::Linux],
                supported_accelerator_vendors: vec![sock_core::AcceleratorVendor::Nvidia],
                allowed_acquisitions: vec![
                    ArtifactAcquisition::VendorPrebuilt,
                    ArtifactAcquisition::LocalSourceBuild,
                ],
                required_witnesses: vec!["flashinfer.prebuilt".to_owned()],
                legal_portability: vec![
                    ArtifactPortability::GpuArchitectureFamilyPortable,
                    ArtifactPortability::AbiClusterPortable,
                ],
                provenance: vec![
                    provenance(
                        "vllm-adapter-survey",
                        "FlashInfer backend surfaced by adapter",
                    ),
                    provenance(
                        "host-discovery",
                        "prebuilt cubin witness available on target host",
                    ),
                ],
            },
            BackendCapability {
                family: BackendFamily::Triton,
                supported_operating_systems: vec![OperatingSystem::Linux],
                supported_accelerator_vendors: vec![sock_core::AcceleratorVendor::Nvidia],
                allowed_acquisitions: vec![ArtifactAcquisition::LocalAotBuild],
                required_witnesses: Vec::new(),
                legal_portability: vec![
                    ArtifactPortability::AbiClusterPortable,
                    ArtifactPortability::GpuArchitectureFamilyPortable,
                ],
                provenance: vec![provenance(
                    "vllm-adapter-survey",
                    "Triton regional compilation remains legal for vLLM attention and warmup paths",
                )],
            },
            BackendCapability {
                family: BackendFamily::AotInductor,
                supported_operating_systems: vec![OperatingSystem::Linux],
                supported_accelerator_vendors: vec![sock_core::AcceleratorVendor::Nvidia],
                allowed_acquisitions: vec![ArtifactAcquisition::UpstreamCacheBundle],
                required_witnesses: Vec::new(),
                legal_portability: vec![ArtifactPortability::AbiClusterPortable],
                provenance: vec![provenance(
                    "vllm-adapter-survey",
                    "AoTInductor reuse is bounded to stable region bundles",
                )],
            },
            BackendCapability {
                family: BackendFamily::CudaGraphs,
                supported_operating_systems: vec![OperatingSystem::Linux],
                supported_accelerator_vendors: vec![sock_core::AcceleratorVendor::Nvidia],
                allowed_acquisitions: vec![ArtifactAcquisition::LocalAotBuild],
                required_witnesses: Vec::new(),
                legal_portability: vec![ArtifactPortability::TopologyScoped],
                provenance: vec![provenance(
                    "vllm-adapter-survey",
                    "CUDA Graph captures are topology-scoped materialized artifacts",
                )],
            },
        ];
        entries.retain(|entry| {
            entry
                .supported_operating_systems
                .contains(&normalized.environment.operating_system)
                && entry
                    .supported_accelerator_vendors
                    .contains(&normalized.environment.accelerator_vendor)
        });
        entries.sort_by_key(|entry| entry.family.as_str().to_owned());
        BackendCapabilityRegistry { entries }
    }

    fn select_backends(
        &self,
        normalized: &NormalizedRequest,
        witnesses: &[CapabilityWitness],
        registry: &BackendCapabilityRegistry,
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
            let Some(capability) = registry
                .entries
                .iter()
                .find(|entry| entry.family == *family)
            else {
                continue;
            };
            let proofs = self.backend_proofs(normalized, witnesses, capability);
            if let Some(proof) = proofs
                .iter()
                .find(|proof| proof.verdict == AdmissibilityVerdict::Admissible)
                .cloned()
            {
                viable.push(BackendCandidate {
                    family: *family,
                    acquisition: proof.acquisition,
                    reason: backend_reason(&proof),
                    admissibility: proof,
                });
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
        scope: &BuildScope,
    ) -> Vec<CompileRegion> {
        let mut regions = adapter_survey
            .compile_regions
            .iter()
            .filter(|region| region.regional_compile_candidate)
            .filter(|region| scope.allows_region(&region.canonical_name))
            .filter(|region| scope.allows_cache_namespace(&region.cache_namespace))
            .filter(|region| scope.allows_warmup_scope(&region.warmup_scope))
            .filter(|region| scope.allows_rank_disposition(region.rank_disposition))
            .filter(|region| {
                scope.allows_backend_family(resolve_backend_binding(
                    region.backend_binding,
                    selected_backends,
                ))
            })
            .map(|region| CompileRegion {
                name: region.canonical_name.clone(),
                kind: region.kind,
                family: resolve_backend_binding(region.backend_binding, selected_backends),
                reusable: region.repeated || region.regional_compile_candidate,
                regional_compile_candidate: region.regional_compile_candidate,
                boundaries: region.boundaries.clone(),
                rationale: region.rationale.clone(),
                invalidation_domain: region.invalidation_domain.clone(),
                shape_planes: region.shape_planes.clone(),
                evidence: region.evidence.clone(),
            })
            .collect::<Vec<_>>();
        regions.sort();
        regions
    }

    fn expand_selected_backends(
        &self,
        normalized: &NormalizedRequest,
        witnesses: &[CapabilityWitness],
        registry: &BackendCapabilityRegistry,
        selected_backends: BackendSelection,
        compile_regions: &[CompileRegion],
    ) -> Result<BackendSelection, PlanError> {
        let mut candidates = std::iter::once(selected_backends.primary.clone())
            .chain(selected_backends.secondary.iter().cloned())
            .collect::<Vec<_>>();
        for family in compile_regions.iter().map(|region| region.family) {
            if candidates
                .iter()
                .any(|candidate| candidate.family == family)
            {
                continue;
            }
            let capability = registry
                .entries
                .iter()
                .find(|entry| entry.family == family)
                .ok_or_else(|| {
                    PlanError::Validation(format!(
                        "compile-region backend {} is absent from the registry",
                        family.as_str()
                    ))
                })?;
            let proof = self
                .backend_proofs(normalized, witnesses, capability)
                .into_iter()
                .find(|proof| proof.verdict == AdmissibilityVerdict::Admissible)
                .ok_or_else(|| {
                    PlanError::Validation(format!(
                        "compile-region backend {} is not admissible for this environment",
                        family.as_str()
                    ))
                })?;
            candidates.push(BackendCandidate {
                family,
                acquisition: proof.acquisition,
                reason: backend_reason(&proof),
                admissibility: proof,
            });
        }
        let primary = candidates.first().cloned().ok_or_else(|| {
            PlanError::Validation(
                "no admissible backend remained after compile-region expansion".to_owned(),
            )
        })?;
        Ok(BackendSelection {
            primary,
            secondary: candidates.into_iter().skip(1).collect(),
        })
    }

    fn backend_proofs(
        &self,
        normalized: &NormalizedRequest,
        witnesses: &[CapabilityWitness],
        capability: &BackendCapability,
    ) -> Vec<BackendAdmissibilityProof> {
        capability
            .allowed_acquisitions
            .iter()
            .copied()
            .map(|acquisition| {
                let mut rejected_reasons = Vec::new();
                let mut satisfied_witnesses = Vec::new();
                let required_witnesses = capability.required_witnesses.clone();

                if !capability
                    .supported_operating_systems
                    .contains(&normalized.environment.operating_system)
                {
                    rejected_reasons.push("unsupported operating system".to_owned());
                }
                if !capability
                    .supported_accelerator_vendors
                    .contains(&normalized.environment.accelerator_vendor)
                {
                    rejected_reasons.push("unsupported accelerator vendor".to_owned());
                }

                for witness_key in &required_witnesses {
                    if witnesses.iter().any(|witness| witness.key == *witness_key) {
                        satisfied_witnesses.push(witness_key.clone());
                    } else {
                        rejected_reasons.push(format!("missing witness `{witness_key}`"));
                    }
                }

                match normalized.backend_policy.packaging_strategy {
                    PackagingStrategy::PrebuiltOnly => {
                        if acquisition != ArtifactAcquisition::VendorPrebuilt {
                            rejected_reasons
                                .push("policy requires vendor prebuilt closure only".to_owned());
                        }
                    }
                    PackagingStrategy::PreferPrebuiltThenAot => {
                        if matches!(acquisition, ArtifactAcquisition::LocalSourceBuild) {
                            rejected_reasons.push(
                                "policy stops at prebuilt and stable AoT materialization"
                                    .to_owned(),
                            );
                        }
                    }
                    PackagingStrategy::PreferPrebuiltThenAotThenJit => {}
                }

                let verdict = if rejected_reasons.is_empty() {
                    AdmissibilityVerdict::Admissible
                } else {
                    AdmissibilityVerdict::Rejected
                };

                BackendAdmissibilityProof {
                    verdict,
                    family: capability.family,
                    acquisition,
                    packaging_strategy: normalized.backend_policy.packaging_strategy,
                    required_witnesses,
                    satisfied_witnesses,
                    rejected_reasons,
                    provenance: capability.provenance.clone(),
                }
            })
            .collect()
    }

    fn artifact_admissibility(
        &self,
        normalized: &NormalizedRequest,
        witnesses: &[CapabilityWitness],
        backend: BackendFamily,
        acquisition: ArtifactAcquisition,
        class: ArtifactClass,
        scope: String,
        portability: ArtifactPortability,
        abi_identity: &sock_core::CanonicalHash,
        shape_envelope_identity: &sock_core::CanonicalHash,
        rationale: Vec<String>,
    ) -> ArtifactAdmissibilityProof {
        let required_witnesses = if backend == BackendFamily::FlashInfer
            && acquisition == ArtifactAcquisition::VendorPrebuilt
        {
            vec!["flashinfer.prebuilt".to_owned()]
        } else {
            Vec::new()
        };
        let satisfied_witnesses = required_witnesses
            .iter()
            .filter(|witness_key| witnesses.iter().any(|witness| witness.key == **witness_key))
            .cloned()
            .collect::<Vec<_>>();
        let fail_closed = required_witnesses.len() == satisfied_witnesses.len()
            && required_witnesses.len() == satisfied_witnesses.len();
        let proof_identity = canonical_hash(&(
            backend,
            acquisition,
            class,
            &scope,
            portability,
            abi_identity,
            shape_envelope_identity,
            &normalized.topology,
            &required_witnesses,
            &satisfied_witnesses,
            fail_closed,
            &rationale,
        ))
        .expect("artifact admissibility proof identity should hash");

        ArtifactAdmissibilityProof {
            proof_identity,
            artifact_scope: scope,
            class,
            backend,
            acquisition,
            portability,
            target_abi_identity: abi_identity.clone(),
            target_shape_envelope_identity: shape_envelope_identity.clone(),
            target_topology: normalized.topology.clone(),
            required_witnesses,
            satisfied_witnesses,
            fail_closed,
            rationale,
        }
    }

    fn shape_envelope(
        &self,
        normalized: &NormalizedRequest,
        selected_backends: &BackendSelection,
        optimization_envelope: &OptimizationEnvelope,
    ) -> ShapeEnvelope {
        let mut nodes = vec![ShapeEnvelopeNode {
            name: "correctness-range".to_owned(),
            plane: CoveragePlane::Correctness,
            intent: RangeIntent::SymbolicRange,
            range: normalized.shape_policy.correctness_range.clone(),
            exact_shape: None,
            required_backends: vec![selected_backends.primary.family],
        }];
        if optimization_envelope.include_performance_warmup {
            nodes.push(ShapeEnvelopeNode {
                name: "performance-range".to_owned(),
                plane: CoveragePlane::Performance,
                intent: RangeIntent::FallbackRange,
                range: normalized.shape_policy.performance_range.clone(),
                exact_shape: None,
                required_backends: vec![selected_backends.primary.family],
            });
            for hot in &normalized.shape_policy.hot_shapes {
                nodes.push(hot_shape_node("hot", hot, selected_backends.primary.family));
            }
        }
        if optimization_envelope.include_cuda_graphs {
            for graph in &normalized.shape_policy.cuda_graph_shapes {
                nodes.push(hot_shape_node(
                    "cuda-graph",
                    graph,
                    BackendFamily::CudaGraphs,
                ));
            }
        }
        if normalized.backend_policy.runtime_jit_policy.disposition
            == RuntimeJitDisposition::ShapeBounded
        {
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
        normalized: &NormalizedRequest,
        witnesses: &[CapabilityWitness],
        shape_envelope: &ShapeEnvelope,
        selected_backends: &BackendSelection,
        compile_regions: &[CompileRegion],
        adapter_survey: &AdapterSurvey,
        optimization_envelope: &OptimizationEnvelope,
        scope: &BuildScope,
    ) -> Result<Vec<ArtifactRequirement>, PlanError> {
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
        let shape_envelope_identity = canonical_hash(shape_envelope)?;
        let mut requirements = compile_regions
            .iter()
            .filter(|region| scope.allows_artifact_scope(&region.name))
            .flat_map(|region| {
                let (graph_portability, graph_rank_disposition, graph_cache_namespace) =
                    cache_traits(&region.name, adapter_survey);
                let acquisition = region_acquisition(region.family, selected_backends);
                [
                    ArtifactRequirement {
                        class: ArtifactClass::CompiledGraph,
                        backend: region.family,
                        acquisition,
                        scope: region.name.clone(),
                        portability: graph_portability,
                        rank_disposition: graph_rank_disposition,
                        expected_bytes: Some(24_000_000),
                        expected_compile_ms: Some(2_500),
                        expected_transfer_ms: Some(if total_rank_count(&normalized.topology) > 1 {
                            220
                        } else {
                            0
                        }),
                        admissibility: self.artifact_admissibility(
                            normalized,
                            witnesses,
                            region.family,
                            acquisition,
                            ArtifactClass::CompiledGraph,
                            region.name.clone(),
                            graph_portability,
                            &abi_identity,
                            &shape_envelope_identity,
                            vec![
                                format!(
                                    "compiled graph ownership follows vLLM cache namespace {}",
                                    graph_cache_namespace
                                ),
                                format!(
                                    "region {} is bounded by the selected shape envelope",
                                    region.name
                                ),
                            ],
                        ),
                    },
                    ArtifactRequirement {
                        class: ArtifactClass::TritonBinary,
                        backend: region.family,
                        acquisition,
                        scope: region.name.clone(),
                        portability: graph_portability,
                        rank_disposition: graph_rank_disposition,
                        expected_bytes: Some(4_000_000),
                        expected_compile_ms: Some(800),
                        expected_transfer_ms: Some(if total_rank_count(&normalized.topology) > 1 {
                            90
                        } else {
                            0
                        }),
                        admissibility: self.artifact_admissibility(
                            normalized,
                            witnesses,
                            region.family,
                            acquisition,
                            ArtifactClass::TritonBinary,
                            region.name.clone(),
                            graph_portability,
                            &abi_identity,
                            &shape_envelope_identity,
                            vec![
                                format!(
                                    "cache namespace {} shapes binary reuse and ownership",
                                    graph_cache_namespace
                                ),
                                format!(
                                    "compile region {} is source-anchored in the adapter survey",
                                    region.name
                                ),
                            ],
                        ),
                    },
                ]
            })
            .collect::<Vec<_>>();
        if optimization_envelope.include_cuda_graphs
            && compile_regions
                .iter()
                .any(|region| region.name == "decode_attention")
            && scope.allows_artifact_scope("decode_attention")
        {
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
                admissibility: self.artifact_admissibility(
                    normalized,
                    witnesses,
                    BackendFamily::CudaGraphs,
                    ArtifactAcquisition::LocalAotBuild,
                    ArtifactClass::CudaGraphCapture,
                    "decode_attention".to_owned(),
                    ArtifactPortability::TopologyScoped,
                    &abi_identity,
                    &shape_envelope_identity,
                    vec![
                        "CUDA graph captures are only legal on the planned topology".to_owned(),
                        "capture reuse is bounded to the exact graph envelope".to_owned(),
                    ],
                ),
            });
        }
        requirements.sort();
        Ok(requirements)
    }

    fn warmup_obligations(
        &self,
        normalized: &NormalizedRequest,
        shape_envelope: &ShapeEnvelope,
        compile_regions: &[CompileRegion],
        optimization_envelope: &OptimizationEnvelope,
        scope: &BuildScope,
    ) -> Vec<WarmupObligation> {
        let all_ranks = total_ranks(&normalized.topology);
        let mut obligations = Vec::new();
        for node in &shape_envelope.nodes {
            if node.intent == RangeIntent::UncoveredResidual {
                continue;
            }
            let steps = match node.intent {
                RangeIntent::ExactHotShape => 1_u32.min(optimization_envelope.max_warmup_steps),
                RangeIntent::SymbolicRange => optimization_envelope.max_warmup_steps.min(3),
                RangeIntent::FallbackRange => optimization_envelope.max_warmup_steps.min(2),
                RangeIntent::UncoveredResidual => 0,
            };
            for region in compile_regions {
                if node.plane == CoveragePlane::Performance
                    && !optimization_envelope.include_performance_warmup
                {
                    continue;
                }
                if node.plane == CoveragePlane::CudaGraph
                    && !optimization_envelope.include_cuda_graphs
                {
                    continue;
                }
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
                        requires_capture: node.plane == CoveragePlane::CudaGraph
                            && optimization_envelope.include_cuda_graphs,
                        requires_autotune: region.family == BackendFamily::FlashInfer
                            && optimization_envelope.enable_autotune,
                        proof: warmup_proof(node, region, &all_ranks),
                    });
                }
            }
        }
        if let Some(readiness) = scope.readiness {
            obligations.retain(|obligation| match readiness {
                BuildReadiness::EarlyServe => false,
                BuildReadiness::Correctness => obligation.blocking,
                BuildReadiness::Performance => true,
            });
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
        compile_regions: &[CompileRegion],
        scope: &BuildScope,
    ) -> Vec<ResidualRuntimeRisk> {
        let mut risks = adapter_survey
            .residual_jit_surfaces
            .iter()
            .filter(|surface| surface_applies(surface.backend_family.as_str(), selected_backends))
            .filter(|surface| {
                scope.is_unscoped()
                    || surface.affected_regions.iter().any(|region| {
                        compile_regions
                            .iter()
                            .any(|candidate| &candidate.name == region)
                    })
            })
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

    fn runtime_jit_evidence(
        &self,
        normalized: &NormalizedRequest,
        selected_backends: &BackendSelection,
        compile_regions: &[CompileRegion],
        warmup_obligations: &[WarmupObligation],
        artifact_requirements: &[ArtifactRequirement],
        adapter_survey: &AdapterSurvey,
    ) -> Vec<RuntimeJitEvidence> {
        adapter_survey
            .residual_jit_surfaces
            .iter()
            .filter(|surface| surface_applies(surface.backend_family.as_str(), selected_backends))
            .filter(|surface| {
                compile_regions.iter().any(|region| {
                    surface
                        .affected_regions
                        .iter()
                        .any(|name| name == &region.name)
                }) || surface.backend_family == "torch.compile"
            })
            .map(|surface| {
                let affected_regions = surface
                    .affected_regions
                    .iter()
                    .filter(|region| {
                        compile_regions
                            .iter()
                            .any(|candidate| &candidate.name == *region)
                    })
                    .cloned()
                    .collect();
                let required_artifacts = surface
                    .required_artifacts
                    .iter()
                    .filter(|scope| {
                        artifact_requirements
                            .iter()
                            .any(|candidate| &candidate.scope == *scope)
                    })
                    .cloned()
                    .collect();
                let required_warmup_proofs = surface
                    .required_warmup_scopes
                    .iter()
                    .flat_map(|scope| warmup_proof_ids_for_scope(scope, warmup_obligations))
                    .collect();

                RuntimeJitEvidence {
                    surface_name: surface.name.clone(),
                    backend_family: surface.backend_family.clone(),
                    trigger_shape_or_config: surface.trigger_shape_or_config.clone(),
                    trigger_inputs: surface.trigger_inputs.clone(),
                    affected_regions,
                    required_artifacts,
                    declared_required_warmup_scopes: surface.required_warmup_scopes.clone(),
                    required_warmup_proofs,
                    topology_context: surface.topology_context.clone(),
                    bounded_by: bounded_by(
                        normalized,
                        selected_backends,
                        compile_regions,
                        warmup_obligations,
                        artifact_requirements,
                        surface,
                    ),
                    mitigation: surface.mitigation.clone(),
                    contradiction_reasons: contradiction_reasons(normalized, surface),
                }
            })
            .collect()
    }

    fn materialization_graph(
        &self,
        topology: &ExecutionTopology,
        artifact_requirements: &[ArtifactRequirement],
        warmup_obligations: &[WarmupObligation],
    ) -> Result<MaterializationGraph, CanonicalError> {
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
            let artifact_node = artifact_node_handle(requirement)?;
            nodes.push(MaterializationNode {
                name: artifact_node.clone(),
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
                produces: vec![artifact_node.clone()],
                rank_scope: rank_zero.clone(),
                invalidation_domain: requirement.scope.clone(),
                replay_boundary: artifact_node.clone(),
                expected_compile_ms: requirement.expected_compile_ms,
                expected_bytes_written: requirement.expected_bytes,
                expected_transfer_ms: requirement.expected_transfer_ms,
                residual_jit_risk_removed: 1,
                execution_contract: NodeExecutionContract {
                    queue,
                    discipline: node_discipline(requirement.rank_disposition, queue),
                    serve_phase: ServePhase::EarlyServeReady,
                    fanout_strategy: artifact_fanout_strategy(requirement),
                    dependency_barrier: Vec::new(),
                },
            });
            if total_rank_count(topology) > 1
                && requirement.rank_disposition == RankDisposition::Shared
            {
                let fanout_node = fanout_node_handle(requirement)?;
                nodes.push(MaterializationNode {
                    name: fanout_node.clone(),
                    wave: 2,
                    kind: MaterializationNodeKind::Transfer,
                    queue: QueueKind::ArtifactIo,
                    plane,
                    dependency_nodes: vec![artifact_node.clone()],
                    consumes: vec![artifact_node.clone()],
                    produces: vec![fanout_node.clone()],
                    rank_scope: total_ranks(topology),
                    invalidation_domain: requirement.scope.clone(),
                    replay_boundary: fanout_node.clone(),
                    expected_compile_ms: Some(0),
                    expected_bytes_written: requirement.expected_bytes,
                    expected_transfer_ms: requirement.expected_transfer_ms,
                    residual_jit_risk_removed: 1,
                    execution_contract: NodeExecutionContract {
                        queue: QueueKind::ArtifactIo,
                        discipline: QueueDiscipline::LeaderThenBroadcast,
                        serve_phase: ServePhase::EarlyServeReady,
                        fanout_strategy: FanoutStrategy::BroadcastFromLeader,
                        dependency_barrier: vec![artifact_node],
                    },
                });
            }
        }
        for obligation in warmup_obligations {
            let dependency_nodes = warmup_dependency_nodes(
                topology,
                artifact_requirements,
                &obligation.required_artifacts,
            )?;
            nodes.push(MaterializationNode {
                name: format!("warmup:{}:{}", obligation.node_name, obligation.region_name),
                wave: if obligation.blocking { 3 } else { 4 },
                kind: MaterializationNodeKind::Warmup,
                queue: QueueKind::Warmup,
                plane: obligation.plane,
                dependency_nodes: dependency_nodes.clone(),
                consumes: dependency_nodes.clone(),
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
                execution_contract: NodeExecutionContract {
                    queue: QueueKind::Warmup,
                    discipline: QueueDiscipline::ParallelPerRank,
                    serve_phase: if obligation.blocking {
                        ServePhase::PreServeBlocking
                    } else {
                        ServePhase::DeferredPerformance
                    },
                    fanout_strategy: FanoutStrategy::None,
                    dependency_barrier: dependency_nodes,
                },
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
        Ok(MaterializationGraph {
            nodes,
            waves,
            leader_assignments,
            early_serve_frontier: warmup_obligations
                .iter()
                .filter(|obligation| obligation.blocking)
                .map(|obligation| {
                    format!("warmup:{}:{}", obligation.node_name, obligation.region_name)
                })
                .collect(),
            late_bindings: vec![(
                "performance".to_owned(),
                "strict-no-surprise-jit".to_owned(),
            )],
            runtime_roi: artifact_requirements.iter().map(runtime_roi).collect(),
        })
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
        backend_registry: &BackendCapabilityRegistry,
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
        let optimization_identity = canonical_hash(&normalized.optimization_policy)?;
        let shape_envelope_identity = canonical_hash(shape_envelope)?;
        let compile_region_identity = canonical_hash(compile_regions)?;
        let capability_identity = canonical_hash(capability_witnesses)?;
        let backend_registry_identity = canonical_hash(backend_registry)?;
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
            &optimization_identity,
            selected_backends,
            compile_regions,
            shape_envelope,
            artifact_requirements,
            materialization_graph,
            guarantee_envelope,
            &request_identity,
            &backend_registry_identity,
            &capability_identity,
            &abi_identity,
            &backend_extension_identity,
            &portability_identity,
        ))?;
        Ok(StructuralIdentity {
            request_identity,
            optimization_identity,
            backend_registry_identity,
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

fn provenance(source: &str, detail: &str) -> CapabilityProvenance {
    CapabilityProvenance {
        source: source.to_owned(),
        detail: detail.to_owned(),
    }
}

fn backend_reason(proof: &BackendAdmissibilityProof) -> String {
    format!(
        "{:?} via {:?} is admissible under {:?}; witnesses={}",
        proof.family,
        proof.acquisition,
        proof.packaging_strategy,
        if proof.satisfied_witnesses.is_empty() {
            "none".to_owned()
        } else {
            proof.satisfied_witnesses.join(",")
        }
    )
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

fn scoped_shape_envelope(
    shape_envelope: ShapeEnvelope,
    compile_regions: &[CompileRegion],
    scope: &BuildScope,
) -> ShapeEnvelope {
    let mut nodes = shape_envelope
        .nodes
        .into_iter()
        .filter(|node| {
            compile_regions
                .iter()
                .any(|region| region.shape_planes.contains(&node.plane))
        })
        .filter(|node| match scope.readiness {
            Some(BuildReadiness::EarlyServe) => true,
            Some(BuildReadiness::Correctness) => node.plane == CoveragePlane::Correctness,
            Some(BuildReadiness::Performance) | None => true,
        })
        .collect::<Vec<_>>();
    nodes.sort();
    ShapeEnvelope { nodes }
}

fn resolve_backend_binding(
    binding: AdapterBackendBinding,
    selected_backends: &BackendSelection,
) -> BackendFamily {
    match binding {
        AdapterBackendBinding::Primary => selected_backends.primary.family,
        AdapterBackendBinding::Fixed(family) => family,
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

fn bounded_by(
    normalized: &NormalizedRequest,
    selected_backends: &BackendSelection,
    compile_regions: &[CompileRegion],
    warmup_obligations: &[WarmupObligation],
    artifact_requirements: &[ArtifactRequirement],
    surface: &sock_core::ResidualRuntimeJitSurface,
) -> Vec<String> {
    let backend_family = surface.backend_family.as_str();
    let regions = compile_regions
        .iter()
        .filter(|region| {
            region.family.as_str() == backend_family
                && surface
                    .affected_regions
                    .iter()
                    .any(|name| name == &region.name)
        })
        .map(|region| format!("region:{}", region.name));
    let warmups = warmup_obligations
        .iter()
        .filter(|obligation| {
            surface
                .required_warmup_scopes
                .iter()
                .any(|scope| scope == &obligation.region_name)
        })
        .map(|obligation| obligation.proof.proof_id.clone());
    let artifacts = artifact_requirements
        .iter()
        .filter(|requirement| {
            requirement.backend.as_str() == backend_family
                && surface
                    .required_artifacts
                    .iter()
                    .any(|scope| scope == &requirement.scope)
        })
        .filter_map(|requirement| artifact_node_handle(requirement).ok());
    let mut bounded = regions
        .chain(warmups)
        .chain(artifacts)
        .chain(
            surface
                .trigger_inputs
                .iter()
                .cloned()
                .map(|input| format!("trigger:{input}")),
        )
        .collect::<Vec<_>>();
    if selected_backends.primary.family.as_str() == backend_family {
        bounded.push(format!(
            "selected_backend:primary:{}",
            selected_backends.primary.family.as_str()
        ));
    }
    bounded.extend(
        selected_backends
            .secondary
            .iter()
            .filter(|candidate| candidate.family.as_str() == backend_family)
            .map(|candidate| format!("selected_backend:secondary:{}", candidate.family.as_str())),
    );

    if backend_family == "torch.compile" {
        bounded.push(format!(
            "packaging:{:?}",
            normalized.backend_policy.packaging_strategy
        ));
        bounded.push(format!(
            "runtime_jit_policy:{:?}",
            normalized.backend_policy.runtime_jit_policy.disposition
        ));
        bounded.push(format!(
            "primary_backend:{}",
            selected_backends.primary.family.as_str()
        ));
    }

    bounded.sort();
    bounded.dedup();
    bounded
}

fn contradiction_reasons(
    normalized: &NormalizedRequest,
    surface: &sock_core::ResidualRuntimeJitSurface,
) -> Vec<String> {
    let mut contradictions = Vec::new();
    if surface.topology_sensitive
        && normalized.topology.tensor_parallelism == 1
        && surface.topology_context.contains("distributed")
    {
        contradictions.push(
            "surface claims distributed-only topology sensitivity but planned topology is single-rank"
                .to_owned(),
        );
    }
    contradictions
}

fn region_acquisition(
    region_family: BackendFamily,
    selected_backends: &BackendSelection,
) -> ArtifactAcquisition {
    if selected_backends.primary.family == region_family {
        selected_backends.primary.acquisition
    } else {
        selected_backends
            .secondary
            .iter()
            .find(|candidate| candidate.family == region_family)
            .map(|candidate| candidate.acquisition)
            .unwrap_or(ArtifactAcquisition::LocalAotBuild)
    }
}

fn cache_traits(
    region_name: &str,
    adapter_survey: &AdapterSurvey,
) -> (ArtifactPortability, RankDisposition, String) {
    adapter_survey
        .compile_regions
        .iter()
        .find(|region| region.canonical_name == region_name)
        .map(|region| {
            (
                region.artifact_portability,
                region.rank_disposition,
                region.cache_namespace.clone(),
            )
        })
        .unwrap_or((
            ArtifactPortability::AbiClusterPortable,
            RankDisposition::Shared,
            "compile-cache".to_owned(),
        ))
}

fn warmup_proof_ids_for_scope(scope: &str, warmup_obligations: &[WarmupObligation]) -> Vec<String> {
    warmup_obligations
        .iter()
        .filter(|obligation| obligation.region_name == scope)
        .map(|obligation| obligation.proof.proof_id.clone())
        .collect()
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
            execution_contract: WaveExecutionContract {
                wave_name: format!("wave-{wave}-{queue:?}"),
                queue,
                discipline: wave_discipline(&members),
                serve_phase: wave_serve_phase(&members),
                fulfills: members
                    .iter()
                    .map(|node| node.replay_boundary.clone())
                    .collect(),
            },
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

fn warmup_proof(
    node: &ShapeEnvelopeNode,
    region: &CompileRegion,
    all_ranks: &[u16],
) -> WarmupCoverageProof {
    WarmupCoverageProof {
        proof_id: format!("warmup:{}:{}", node.name, region.name),
        node_name: node.name.clone(),
        region_name: region.name.clone(),
        plane: node.plane,
        required_artifacts: vec![region.name.clone()],
        rank_scope: all_ranks.to_vec(),
        blocking: node.plane == CoveragePlane::Correctness,
        expected_states: coverage_states_for_plane(node.plane)
            .into_iter()
            .map(|state| format!("{state:?}"))
            .collect(),
        contradiction_triggers: vec![
            WarmupContradiction {
                trigger: "shape_escape".to_owned(),
                invalidates_node: node.name.clone(),
                next_action: "expand the warmup envelope or downgrade the serve guarantee"
                    .to_owned(),
            },
            WarmupContradiction {
                trigger: "artifact_identity_change".to_owned(),
                invalidates_node: region.name.clone(),
                next_action: "re-materialize the artifact and re-run warmup".to_owned(),
            },
        ],
        serve_phase: if node.plane == CoveragePlane::Correctness {
            ServePhase::PreServeBlocking
        } else {
            ServePhase::DeferredPerformance
        },
    }
}

fn artifact_fanout_strategy(requirement: &ArtifactRequirement) -> FanoutStrategy {
    match requirement.rank_disposition {
        RankDisposition::Shared => FanoutStrategy::BroadcastFromLeader,
        RankDisposition::RankLocal => FanoutStrategy::RebuildPerRank,
    }
}

fn warmup_dependency_nodes(
    topology: &ExecutionTopology,
    artifact_requirements: &[ArtifactRequirement],
    required_artifacts: &[String],
) -> Result<Vec<String>, CanonicalError> {
    required_artifacts
        .iter()
        .flat_map(|artifact| {
            artifact_requirements
                .iter()
                .filter(|requirement| requirement.scope == *artifact)
                .map(|requirement| {
                    if requirement.rank_disposition == RankDisposition::Shared
                        && total_rank_count(topology) > 1
                    {
                        fanout_node_handle(requirement)
                    } else {
                        artifact_node_handle(requirement)
                    }
                })
        })
        .collect()
}

fn node_discipline(rank_disposition: RankDisposition, queue: QueueKind) -> QueueDiscipline {
    match (rank_disposition, queue) {
        (RankDisposition::Shared, QueueKind::ArtifactIo) => QueueDiscipline::LeaderThenBroadcast,
        (RankDisposition::Shared, _) => QueueDiscipline::Serial,
        (RankDisposition::RankLocal, _) => QueueDiscipline::ParallelPerRank,
    }
}

fn runtime_roi(requirement: &ArtifactRequirement) -> RuntimeRoi {
    let compile_ms = requirement.expected_compile_ms.unwrap_or(0);
    let transfer_ms = requirement.expected_transfer_ms.unwrap_or(0);
    let rebuild_ms = compile_ms.saturating_mul(2);
    let preferred_strategy =
        if requirement.rank_disposition == RankDisposition::Shared && transfer_ms < rebuild_ms {
            FanoutStrategy::BroadcastFromLeader
        } else if requirement.rank_disposition == RankDisposition::Shared {
            FanoutStrategy::RebuildPerRank
        } else {
            FanoutStrategy::RebuildPerRank
        };

    RuntimeRoi {
        artifact_scope: requirement.scope.clone(),
        compile_ms,
        transfer_ms,
        rebuild_ms,
        preferred_strategy,
    }
}

fn wave_discipline(members: &[&MaterializationNode]) -> QueueDiscipline {
    if members
        .iter()
        .any(|node| node.execution_contract.discipline == QueueDiscipline::LeaderThenBroadcast)
    {
        QueueDiscipline::LeaderThenBroadcast
    } else if members
        .iter()
        .all(|node| node.execution_contract.discipline == QueueDiscipline::ParallelPerRank)
    {
        QueueDiscipline::ParallelPerRank
    } else {
        QueueDiscipline::Serial
    }
}

fn wave_serve_phase(members: &[&MaterializationNode]) -> ServePhase {
    if members
        .iter()
        .any(|node| node.execution_contract.serve_phase == ServePhase::PreServeBlocking)
    {
        ServePhase::PreServeBlocking
    } else if members
        .iter()
        .any(|node| node.execution_contract.serve_phase == ServePhase::EarlyServeReady)
    {
        ServePhase::EarlyServeReady
    } else {
        ServePhase::DeferredPerformance
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
                packaging_strategy: PackagingStrategy::PreferPrebuiltThenAot,
                runtime_jit_policy: sock_core::RuntimeJitPolicy {
                    disposition: RuntimeJitDisposition::Forbidden,
                    max_residual_node_count: 0,
                },
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
            optimization_policy: sock_core::OptimizationPolicy {
                level: sock_core::OptimizationLevel::O2,
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

    #[test]
    fn materialization_graph_uses_unique_artifact_nodes() {
        let planner = Planner::new(host());
        let outcome = planner.resolve(request()).expect("plan");
        let mut names = std::collections::BTreeSet::new();

        for node in &outcome.plan.materialization_graph.nodes {
            assert!(
                names.insert(node.name.as_str()),
                "duplicate materialization node {}",
                node.name
            );
        }
    }

    #[test]
    fn prebuilt_only_policy_fails_closed_without_witness() {
        let mut host = host();
        host.flashinfer_prebuilt_available = false;
        let planner = Planner::new(host);
        let mut request = request();
        request.backend_policy.preferred_families = vec![BackendFamily::FlashInfer];

        let err = planner
            .resolve(request)
            .expect_err("prebuilt-only path should fail without witness");
        assert!(
            matches!(err, PlanError::Validation(message) if message.contains("no admissible backend remained"))
        );
    }

    #[test]
    fn bounded_jit_policy_is_explicitly_supported() {
        let planner = Planner::new(host());
        let mut request = request();
        request.backend_policy.packaging_strategy = PackagingStrategy::PreferPrebuiltThenAotThenJit;
        request.backend_policy.runtime_jit_policy.disposition = RuntimeJitDisposition::ShapeBounded;
        request.backend_policy.correctness_target.level = GuaranteeLevel::WarmupBounded;
        request
            .backend_policy
            .runtime_jit_policy
            .max_residual_node_count = 1;

        let outcome = planner.resolve(request).expect("bounded jit plan");
        assert!(
            outcome
                .plan
                .shape_envelope
                .nodes
                .iter()
                .any(|node| node.intent == RangeIntent::UncoveredResidual)
        );
    }
}
