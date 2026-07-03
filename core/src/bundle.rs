use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use hex::ToHex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    ArtifactClosure, ArtifactManifestEntry, BackendDecisionDocument, CanonicalError, CanonicalHash,
    DiagnosticsDocument, MaterializationExecutionReport, OptimizationExplainDocument,
    ReplayProofDocument, ResolvedBuildPlan, RewriteTraceDocument, SocPlanDocument,
    VerificationReport, VllmEntrypointDocument, VllmIntegrationDocument, canonical_json,
    render_backend_decision, render_replay_bundle_explain,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactManifestDocument {
    pub schema_version: crate::SchemaVersion,
    pub plan_identity: CanonicalHash,
    pub artifacts: Vec<ArtifactManifestEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayBundleMetadata {
    pub schema_version: crate::SchemaVersion,
    pub plan_identity: CanonicalHash,
    pub file_digests: BTreeMap<String, String>,
    pub replay_entrypoint: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayBundle {
    pub build_plan: ResolvedBuildPlan,
    pub artifact_closure: ArtifactClosure,
    pub verification_report: VerificationReport,
    pub diagnostics: DiagnosticsDocument,
    pub rewrite_trace: RewriteTraceDocument,
    pub optimization_explain: OptimizationExplainDocument,
    pub backend_decision: BackendDecisionDocument,
    pub materialization_report: MaterializationExecutionReport,
    pub replay_proof: ReplayProofDocument,
    pub vllm_integration: VllmIntegrationDocument,
    pub soc_plan: SocPlanDocument,
    pub vllm_entrypoints: VllmEntrypointDocument,
}

#[derive(Debug, Error)]
pub enum ReplayBundleError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("canonical error: {0}")]
    Canonical(#[from] CanonicalError),
    #[error("digest mismatch for {file}")]
    DigestMismatch { file: String },
    #[error("identity mismatch in {document}")]
    IdentityMismatch { document: String },
    #[error("verification report does not match the loaded build plan")]
    VerificationMismatch,
    #[error("backend decision does not match the loaded build plan")]
    BackendDecisionMismatch,
    #[error("replay proof does not match the loaded build plan and materialization report")]
    ReplayProofMismatch,
}

impl ArtifactManifestDocument {
    #[must_use]
    pub fn from_closure(closure: &ArtifactClosure) -> Self {
        let mut artifacts = closure.artifacts.clone();
        artifacts.sort();
        Self {
            schema_version: crate::SchemaVersion::current(),
            plan_identity: closure.plan_identity.clone(),
            artifacts,
        }
    }
}

impl ReplayBundle {
    #[must_use]
    pub fn replay_script(&self) -> String {
        "#!/usr/bin/env sh\nset -eu\nDIR=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\nexec cargo run --quiet --bin sock -- replay --bundle \"$DIR\"\n".to_owned()
    }

    pub fn write_to(&self, dir: &Path) -> Result<ReplayBundleMetadata, ReplayBundleError> {
        fs::create_dir_all(dir)?;
        let artifact_manifest = ArtifactManifestDocument::from_closure(&self.artifact_closure);
        let files = [
            ("buildplan.json", canonical_json(&self.build_plan)?),
            (
                "artifact_manifest.json",
                canonical_json(&artifact_manifest)?,
            ),
            (
                "verification_report.json",
                canonical_json(&self.verification_report)?,
            ),
            ("diagnostics.json", canonical_json(&self.diagnostics)?),
            ("rewrite_trace.json", canonical_json(&self.rewrite_trace)?),
            (
                "optimization_explain.json",
                canonical_json(&self.optimization_explain)?,
            ),
            (
                "backend_decision.json",
                canonical_json(&self.backend_decision)?,
            ),
            (
                "materialization_report.json",
                canonical_json(&self.materialization_report)?,
            ),
            ("replay_proof.json", canonical_json(&self.replay_proof)?),
            (
                "vllm_integration.json",
                canonical_json(&self.vllm_integration)?,
            ),
            ("soc_plan.json", canonical_json(&self.soc_plan)?),
            (
                "vllm_entrypoints.json",
                canonical_json(&self.vllm_entrypoints)?,
            ),
        ];
        let mut file_digests = BTreeMap::new();
        for (name, content) in files {
            fs::write(dir.join(name), content.as_bytes())?;
            file_digests.insert(name.to_owned(), digest(content.as_bytes()));
        }
        let explain_text = render_replay_bundle_explain(
            &self.build_plan,
            &self.optimization_explain,
            &self.verification_report,
            &self.diagnostics,
            &self.materialization_report,
            &self.replay_proof,
        );
        let explain_text = format!(
            "{explain_text}{}",
            render_backend_decision(&self.backend_decision)
        );
        fs::write(dir.join("explain.txt"), explain_text.as_bytes())?;
        file_digests.insert("explain.txt".to_owned(), digest(explain_text.as_bytes()));
        let replay_script = self.replay_script();
        fs::write(dir.join("replay.sh"), replay_script.as_bytes())?;
        file_digests.insert("replay.sh".to_owned(), digest(replay_script.as_bytes()));
        let metadata = ReplayBundleMetadata {
            schema_version: crate::SchemaVersion::current(),
            plan_identity: self.build_plan.structural_identity.plan_identity.clone(),
            file_digests,
            replay_entrypoint: "./replay.sh".to_owned(),
        };
        fs::write(
            dir.join("bundle_metadata.json"),
            canonical_json(&metadata)?.as_bytes(),
        )?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = fs::metadata(dir.join("replay.sh"))?.permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(dir.join("replay.sh"), permissions)?;
        }
        Ok(metadata)
    }

    pub fn load_from(dir: &Path) -> Result<Self, ReplayBundleError> {
        let metadata: ReplayBundleMetadata =
            serde_json::from_str(&fs::read_to_string(dir.join("bundle_metadata.json"))?)?;
        for (file, expected) in &metadata.file_digests {
            let actual = digest(fs::read(dir.join(file))?.as_slice());
            if &actual != expected {
                return Err(ReplayBundleError::DigestMismatch { file: file.clone() });
            }
        }

        let build_plan: ResolvedBuildPlan =
            serde_json::from_str(&fs::read_to_string(dir.join("buildplan.json"))?)?;
        let artifact_manifest: ArtifactManifestDocument =
            serde_json::from_str(&fs::read_to_string(dir.join("artifact_manifest.json"))?)?;
        let verification_report: VerificationReport =
            serde_json::from_str(&fs::read_to_string(dir.join("verification_report.json"))?)?;
        let diagnostics: DiagnosticsDocument =
            serde_json::from_str(&fs::read_to_string(dir.join("diagnostics.json"))?)?;
        let rewrite_trace: RewriteTraceDocument =
            serde_json::from_str(&fs::read_to_string(dir.join("rewrite_trace.json"))?)?;
        let optimization_explain: OptimizationExplainDocument =
            serde_json::from_str(&fs::read_to_string(dir.join("optimization_explain.json"))?)?;
        let backend_decision: BackendDecisionDocument =
            serde_json::from_str(&fs::read_to_string(dir.join("backend_decision.json"))?)?;
        let materialization_report: MaterializationExecutionReport = serde_json::from_str(
            &fs::read_to_string(dir.join("materialization_report.json"))?,
        )?;
        let replay_proof: ReplayProofDocument =
            serde_json::from_str(&fs::read_to_string(dir.join("replay_proof.json"))?)?;
        let vllm_integration: VllmIntegrationDocument =
            serde_json::from_str(&fs::read_to_string(dir.join("vllm_integration.json"))?)?;
        let soc_plan: SocPlanDocument =
            serde_json::from_str(&fs::read_to_string(dir.join("soc_plan.json"))?)?;
        let vllm_entrypoints: VllmEntrypointDocument =
            serde_json::from_str(&fs::read_to_string(dir.join("vllm_entrypoints.json"))?)?;
        let plan_identity = build_plan.structural_identity.plan_identity.clone();

        if metadata.plan_identity != plan_identity {
            return Err(ReplayBundleError::IdentityMismatch {
                document: "bundle_metadata.json".to_owned(),
            });
        }
        if artifact_manifest.plan_identity != plan_identity {
            return Err(ReplayBundleError::IdentityMismatch {
                document: "artifact_manifest.json".to_owned(),
            });
        }
        if diagnostics.plan_identity != plan_identity {
            return Err(ReplayBundleError::IdentityMismatch {
                document: "diagnostics.json".to_owned(),
            });
        }
        if rewrite_trace.plan_identity != plan_identity {
            return Err(ReplayBundleError::IdentityMismatch {
                document: "rewrite_trace.json".to_owned(),
            });
        }
        if optimization_explain.plan_identity != plan_identity {
            return Err(ReplayBundleError::IdentityMismatch {
                document: "optimization_explain.json".to_owned(),
            });
        }
        if backend_decision.plan_identity != plan_identity {
            return Err(ReplayBundleError::IdentityMismatch {
                document: "backend_decision.json".to_owned(),
            });
        }
        if materialization_report.plan_identity != plan_identity {
            return Err(ReplayBundleError::IdentityMismatch {
                document: "materialization_report.json".to_owned(),
            });
        }
        if replay_proof.plan_identity != plan_identity {
            return Err(ReplayBundleError::IdentityMismatch {
                document: "replay_proof.json".to_owned(),
            });
        }
        if vllm_integration.plan_identity != plan_identity {
            return Err(ReplayBundleError::IdentityMismatch {
                document: "vllm_integration.json".to_owned(),
            });
        }
        if soc_plan.plan_identity != plan_identity {
            return Err(ReplayBundleError::IdentityMismatch {
                document: "soc_plan.json".to_owned(),
            });
        }
        if vllm_entrypoints.plan_identity != plan_identity {
            return Err(ReplayBundleError::IdentityMismatch {
                document: "vllm_entrypoints.json".to_owned(),
            });
        }
        if build_plan.validate() != verification_report {
            return Err(ReplayBundleError::VerificationMismatch);
        }
        if BackendDecisionDocument::from_plan(&build_plan) != backend_decision {
            return Err(ReplayBundleError::BackendDecisionMismatch);
        }
        if ReplayProofDocument::from_plan_and_materialization(&build_plan, &materialization_report)?
            != replay_proof
        {
            return Err(ReplayBundleError::ReplayProofMismatch);
        }

        Ok(Self {
            build_plan,
            artifact_closure: ArtifactClosure {
                plan_identity: artifact_manifest.plan_identity.clone(),
                artifacts: {
                    let mut artifacts = artifact_manifest.artifacts;
                    artifacts.sort();
                    artifacts
                },
            },
            verification_report,
            diagnostics,
            rewrite_trace,
            optimization_explain,
            backend_decision,
            materialization_report,
            replay_proof,
            vllm_integration,
            soc_plan,
            vllm_entrypoints,
        })
    }
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes).encode_hex::<String>()
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use crate::{
        AbiFingerprint, AcceleratorVendor, ArtifactAcquisition, ArtifactClass, ArtifactPortability,
        ArtifactRequirement, BackendCandidate, BackendExtensionFingerprint, BackendFamily,
        BackendPolicy, BackendSelection, CachePolicy, CapabilityWitness, CompileRegion,
        ConfigEntry, ConfigLayer, CoveragePlane, CoverageState, CoverageWitness,
        DiagnosticsDocument, EngineSource, ExecutionTopology, FailureMode, FanoutStrategy,
        GuaranteeDimension, GuaranteeEnvelope, GuaranteeEvidence, GuaranteeLevel, GuaranteeTarget,
        MaterializationDisposition, MaterializationExecutionReport, MaterializationGraph,
        MaterializationNode, MaterializationNodeKind, MaterializationNodeRecord,
        MaterializationWave, MaterializationWaveRecord, MaterializedArtifactRecord, ModelRef,
        NodeExecutionContract, ObservedReadinessLevel, OptimizationEnvelope, OptimizationLevel,
        OptimizationPolicy, PortabilityFingerprint, QueueDiscipline, QueueKind, RangeIntent,
        RankDisposition, RawRequest, ReadinessObservation, ReplayProofDocument,
        RequestedEnvironment, RewritePassContract, RewritePhase, RewriteTraceDocument,
        RuntimeJitEvidence, RuntimeJitObservation, RuntimeJitObservationStatus, RuntimeRoi,
        SchemaVersion, ServePhase, ShapeEnvelope, ShapeEnvelopeNode, ShapePoint, ShapePolicy,
        ShapeRange, SourceAnchor, SourceEvidence, StartupClosureOutcome, StructuralIdentity,
        TargetEngine, WarmupContradiction, WarmupCoverageProof, WarmupObligation, WarmupPolicy,
        WaveEstimate, WaveExecutionContract, canonical_hash,
    };

    use super::{ReplayBundle, ReplayBundleError, ReplayBundleMetadata, digest};

    fn sample_bundle() -> ReplayBundle {
        let normalized_request = RawRequest {
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
                operating_system: crate::OperatingSystem::Linux,
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
                packaging_strategy: crate::PackagingStrategy::PreferPrebuiltThenAot,
                runtime_jit_policy: crate::RuntimeJitPolicy {
                    disposition: crate::RuntimeJitDisposition::Forbidden,
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
                max_warmup_steps: 4,
                verify_cuda_graph_capture: true,
            },
            optimization_policy: OptimizationPolicy {
                level: OptimizationLevel::O2,
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
        .normalize()
        .expect("normalize request");

        let optimization_envelope = OptimizationEnvelope::from_level(OptimizationLevel::O2);
        let backend_registry = crate::BackendCapabilityRegistry {
            entries: vec![crate::BackendCapability {
                family: BackendFamily::FlashInfer,
                supported_operating_systems: vec![crate::OperatingSystem::Linux],
                supported_accelerator_vendors: vec![AcceleratorVendor::Nvidia],
                allowed_acquisitions: vec![ArtifactAcquisition::VendorPrebuilt],
                required_witnesses: vec!["flashinfer.prebuilt".to_owned()],
                legal_portability: vec![ArtifactPortability::GpuArchitectureFamilyPortable],
                provenance: vec![crate::CapabilityProvenance {
                    source: "fixture".to_owned(),
                    detail: "flashinfer fixture".to_owned(),
                }],
            }],
        };
        let backend_proof = crate::BackendAdmissibilityProof {
            verdict: crate::AdmissibilityVerdict::Admissible,
            family: BackendFamily::FlashInfer,
            acquisition: ArtifactAcquisition::VendorPrebuilt,
            packaging_strategy: crate::PackagingStrategy::PreferPrebuiltThenAot,
            required_witnesses: vec!["flashinfer.prebuilt".to_owned()],
            satisfied_witnesses: vec!["flashinfer.prebuilt".to_owned()],
            rejected_reasons: Vec::new(),
            provenance: vec![crate::CapabilityProvenance {
                source: "fixture".to_owned(),
                detail: "flashinfer admissible".to_owned(),
            }],
        };

        let selected_backends = BackendSelection {
            primary: BackendCandidate {
                family: BackendFamily::FlashInfer,
                acquisition: ArtifactAcquisition::VendorPrebuilt,
                reason: "prebuilt flashinfer available".to_owned(),
                admissibility: backend_proof.clone(),
            },
            secondary: vec![BackendCandidate {
                family: BackendFamily::Triton,
                acquisition: ArtifactAcquisition::LocalAotBuild,
                reason: "triton remains admissible".to_owned(),
                admissibility: crate::BackendAdmissibilityProof {
                    verdict: crate::AdmissibilityVerdict::Admissible,
                    family: BackendFamily::Triton,
                    acquisition: ArtifactAcquisition::LocalAotBuild,
                    packaging_strategy: crate::PackagingStrategy::PreferPrebuiltThenAot,
                    required_witnesses: Vec::new(),
                    satisfied_witnesses: Vec::new(),
                    rejected_reasons: Vec::new(),
                    provenance: vec![crate::CapabilityProvenance {
                        source: "fixture".to_owned(),
                        detail: "triton admissible".to_owned(),
                    }],
                },
            }],
        };
        let compile_regions = vec![
            CompileRegion {
                name: "prefill_attention".to_owned(),
                kind: crate::adapter::CompileRegionKind::PrefillMicrograph,
                family: BackendFamily::FlashInfer,
                reusable: true,
                regional_compile_candidate: true,
                boundaries: vec!["prefill-boundary".to_owned()],
                rationale: "prefill region".to_owned(),
                invalidation_domain: "prefill_attention".to_owned(),
                shape_planes: vec![CoveragePlane::Correctness, CoveragePlane::Performance],
                stable_identity: canonical_hash(&"fixture-prefill-stable")
                    .expect("fixture prefill stable identity"),
                equivalence_identity: canonical_hash(&"fixture-prefill-equivalence")
                    .expect("fixture prefill equivalence identity"),
                cache_namespace: "compile-cache".to_owned(),
                cache_sharing: crate::RegionCacheSharing::ContentAddressed,
                portability: ArtifactPortability::GpuArchitectureFamilyPortable,
                rank_disposition: RankDisposition::Shared,
                topology_sensitive: false,
                portability_scope: "gpu_architecture_family".to_owned(),
                topology_scope: "cross_rank_and_cross_process".to_owned(),
                warmup_scope: "prefill_attention".to_owned(),
                closure_verification_criteria: vec![
                    "prefill warmup proof present".to_owned(),
                    "shape-envelope bounded specialization verified".to_owned(),
                ],
                evidence: SourceEvidence {
                    summary: "fixture".to_owned(),
                    anchors: Vec::new(),
                },
            },
            CompileRegion {
                name: "decode_attention".to_owned(),
                kind: crate::adapter::CompileRegionKind::DecodeMicrograph,
                family: BackendFamily::FlashInfer,
                reusable: true,
                regional_compile_candidate: true,
                boundaries: vec!["decode-boundary".to_owned()],
                rationale: "decode region".to_owned(),
                invalidation_domain: "decode_attention".to_owned(),
                shape_planes: vec![
                    CoveragePlane::Correctness,
                    CoveragePlane::Performance,
                    CoveragePlane::CudaGraph,
                ],
                stable_identity: canonical_hash(&"fixture-decode-stable")
                    .expect("fixture decode stable identity"),
                equivalence_identity: canonical_hash(&"fixture-decode-equivalence")
                    .expect("fixture decode equivalence identity"),
                cache_namespace: "cuda-graph-cache".to_owned(),
                cache_sharing: crate::RegionCacheSharing::NamespaceLocal,
                portability: ArtifactPortability::TopologyScoped,
                rank_disposition: RankDisposition::RankLocal,
                topology_sensitive: true,
                portability_scope: "exact_runtime_topology".to_owned(),
                topology_scope: "rank_local".to_owned(),
                warmup_scope: "decode_attention".to_owned(),
                closure_verification_criteria: vec![
                    "decode warmup proof present".to_owned(),
                    "cuda graph capture verified".to_owned(),
                ],
                evidence: SourceEvidence {
                    summary: "fixture".to_owned(),
                    anchors: Vec::new(),
                },
            },
        ];
        let shape_envelope = ShapeEnvelope {
            nodes: vec![
                ShapeEnvelopeNode {
                    name: "correctness-range".to_owned(),
                    plane: CoveragePlane::Correctness,
                    intent: RangeIntent::SymbolicRange,
                    range: ShapeRange {
                        min_batch_size: 1,
                        max_batch_size: 8,
                        min_sequence_length: 1,
                        max_sequence_length: 4096,
                    },
                    exact_shape: None,
                    required_backends: vec![BackendFamily::FlashInfer],
                },
                ShapeEnvelopeNode {
                    name: "performance-range".to_owned(),
                    plane: CoveragePlane::Performance,
                    intent: RangeIntent::FallbackRange,
                    range: ShapeRange {
                        min_batch_size: 1,
                        max_batch_size: 4,
                        min_sequence_length: 1,
                        max_sequence_length: 2048,
                    },
                    exact_shape: None,
                    required_backends: vec![BackendFamily::FlashInfer],
                },
            ],
        };
        let abi_identity = canonical_hash(&AbiFingerprint {
            operating_system: normalized_request.environment.operating_system,
            accelerator_vendor: normalized_request.environment.accelerator_vendor,
            gpu_arches: normalized_request.environment.gpu_arches.clone(),
            cuda_version: normalized_request.environment.cuda_version.clone(),
            driver_version: normalized_request.environment.driver_version.clone(),
            python_abi: normalized_request.environment.python_abi.clone(),
            libc_abi: normalized_request.environment.libc_abi.clone(),
            topology: normalized_request.topology.clone(),
        })
        .expect("abi identity");
        let shape_envelope_identity =
            canonical_hash(&shape_envelope).expect("shape envelope identity");
        let artifact_admissibility = crate::ArtifactAdmissibilityProof {
            proof_identity: canonical_hash(&"fixture-artifact-proof").expect("proof identity"),
            artifact_scope: "prefill_attention".to_owned(),
            class: ArtifactClass::CompiledGraph,
            backend: BackendFamily::FlashInfer,
            acquisition: ArtifactAcquisition::VendorPrebuilt,
            portability: ArtifactPortability::GpuArchitectureFamilyPortable,
            target_abi_identity: abi_identity.clone(),
            target_shape_envelope_identity: shape_envelope_identity.clone(),
            target_topology: normalized_request.topology.clone(),
            required_witnesses: vec!["flashinfer.prebuilt".to_owned()],
            satisfied_witnesses: vec!["flashinfer.prebuilt".to_owned()],
            fail_closed: true,
            rationale: vec!["fixture admissibility".to_owned()],
        };
        let artifact_requirements = vec![ArtifactRequirement {
            class: ArtifactClass::CompiledGraph,
            backend: BackendFamily::FlashInfer,
            acquisition: ArtifactAcquisition::VendorPrebuilt,
            scope: "prefill_attention".to_owned(),
            portability: ArtifactPortability::GpuArchitectureFamilyPortable,
            rank_disposition: RankDisposition::Shared,
            expected_bytes: Some(24_000_000),
            expected_compile_ms: Some(2_500),
            expected_transfer_ms: Some(220),
            admissibility: artifact_admissibility.clone(),
        }];
        let warmup_obligations = vec![
            WarmupObligation {
                node_name: "correctness-range".to_owned(),
                region_name: "prefill_attention".to_owned(),
                step_count: 2,
                plane: CoveragePlane::Correctness,
                blocking: true,
                required_artifacts: vec!["prefill_attention".to_owned()],
                rank_scope: vec![0, 1],
                requires_capture: false,
                requires_autotune: true,
                proof: WarmupCoverageProof {
                    proof_id: "warmup:correctness-range:prefill_attention".to_owned(),
                    node_name: "correctness-range".to_owned(),
                    region_name: "prefill_attention".to_owned(),
                    plane: CoveragePlane::Correctness,
                    required_artifacts: vec!["prefill_attention".to_owned()],
                    rank_scope: vec![0, 1],
                    blocking: true,
                    expected_states: vec!["Compiled".to_owned(), "Executed".to_owned()],
                    contradiction_triggers: vec![WarmupContradiction {
                        trigger: "shape_escape".to_owned(),
                        invalidates_node: "correctness-range".to_owned(),
                        next_action: "expand warmup".to_owned(),
                    }],
                    serve_phase: ServePhase::PreServeBlocking,
                },
            },
            WarmupObligation {
                node_name: "performance-range".to_owned(),
                region_name: "decode_attention".to_owned(),
                step_count: 1,
                plane: CoveragePlane::Performance,
                blocking: false,
                required_artifacts: vec!["prefill_attention".to_owned()],
                rank_scope: vec![0, 1],
                requires_capture: false,
                requires_autotune: true,
                proof: WarmupCoverageProof {
                    proof_id: "warmup:performance-range:decode_attention".to_owned(),
                    node_name: "performance-range".to_owned(),
                    region_name: "decode_attention".to_owned(),
                    plane: CoveragePlane::Performance,
                    required_artifacts: vec!["prefill_attention".to_owned()],
                    rank_scope: vec![0, 1],
                    blocking: false,
                    expected_states: vec![
                        "Compiled".to_owned(),
                        "Executed".to_owned(),
                        "VerifiedNoNewCompile".to_owned(),
                    ],
                    contradiction_triggers: vec![WarmupContradiction {
                        trigger: "shape_escape".to_owned(),
                        invalidates_node: "performance-range".to_owned(),
                        next_action: "expand warmup".to_owned(),
                    }],
                    serve_phase: ServePhase::DeferredPerformance,
                },
            },
        ];
        let materialization_graph = MaterializationGraph {
            nodes: vec![MaterializationNode {
                name: "import:prefill_attention".to_owned(),
                wave: 0,
                kind: MaterializationNodeKind::Materialize,
                queue: QueueKind::Compile,
                plane: CoveragePlane::Correctness,
                dependency_nodes: Vec::new(),
                consumes: Vec::new(),
                produces: vec!["prefill_attention".to_owned()],
                rank_scope: vec![0],
                invalidation_domain: "prefill_attention".to_owned(),
                replay_boundary: "artifact:prefill_attention".to_owned(),
                expected_compile_ms: Some(2500),
                expected_bytes_written: Some(24_000_000),
                expected_transfer_ms: Some(0),
                residual_jit_risk_removed: 1,
                execution_contract: NodeExecutionContract {
                    queue: QueueKind::Compile,
                    discipline: QueueDiscipline::Serial,
                    serve_phase: ServePhase::EarlyServeReady,
                    fanout_strategy: FanoutStrategy::BroadcastFromLeader,
                    dependency_barrier: Vec::new(),
                },
            }],
            waves: vec![MaterializationWave {
                name: "wave-0".to_owned(),
                queue: QueueKind::Compile,
                node_names: vec!["import:prefill_attention".to_owned()],
                estimate: WaveEstimate {
                    expected_compile_ms: Some(2500),
                    expected_bytes_written: Some(24_000_000),
                    expected_transfer_ms: Some(0),
                    fanout_count: 1,
                    residual_jit_risk_removed: 1,
                },
                hazard_repairs: Vec::new(),
                execution_contract: WaveExecutionContract {
                    wave_name: "wave-0".to_owned(),
                    queue: QueueKind::Compile,
                    discipline: QueueDiscipline::Serial,
                    serve_phase: ServePhase::EarlyServeReady,
                    fulfills: vec!["artifact:prefill_attention".to_owned()],
                },
            }],
            leader_assignments: Vec::new(),
            early_serve_frontier: vec!["import:prefill_attention".to_owned()],
            late_bindings: vec![("cache_root".to_owned(), "host://cache/sock".to_owned())],
            runtime_roi: vec![RuntimeRoi {
                artifact_scope: "prefill_attention".to_owned(),
                compile_ms: 2500,
                transfer_ms: 0,
                rebuild_ms: 5000,
                preferred_strategy: FanoutStrategy::BroadcastFromLeader,
            }],
        };
        let artifact_manifest = vec![crate::ArtifactManifestEntry {
            identity: "flashinfer:CompiledGraph:prefill_attention".to_owned(),
            class: ArtifactClass::CompiledGraph,
            backend: BackendFamily::FlashInfer,
            scope: "prefill_attention".to_owned(),
            admissibility: artifact_admissibility,
        }];
        let guarantee_evidence = GuaranteeEvidence {
            capability_witnesses: vec![CapabilityWitness {
                key: "flashinfer.prebuilt".to_owned(),
                value: "available".to_owned(),
                provenance: "host-discovery".to_owned(),
            }],
            artifact_manifest: artifact_manifest.clone(),
            warmup_obligations: warmup_obligations.clone(),
            coverage_witnesses: vec![CoverageWitness {
                plane: CoveragePlane::Correctness,
                node_name: "correctness-range".to_owned(),
                evidence: "planned:correctness-range".to_owned(),
                coverage_states: vec![CoverageState::Compiled, CoverageState::Executed],
                artifact_scopes: vec!["prefill_attention".to_owned()],
                uncovered_residuals: Vec::new(),
            }],
            runtime_jit_evidence: Vec::<RuntimeJitEvidence>::new(),
        };
        let guarantee_envelope = GuaranteeEnvelope {
            requested_correctness: GuaranteeTarget {
                level: GuaranteeLevel::ShapeBoundedAot,
                failure_mode: FailureMode::FailClosed,
            },
            requested_performance: GuaranteeTarget {
                level: GuaranteeLevel::WarmupBounded,
                failure_mode: FailureMode::FailClosed,
            },
            achieved_correctness: GuaranteeLevel::ShapeBoundedAot,
            achieved_performance: GuaranteeLevel::StrictNoSurpriseJit,
            covered_dimensions: vec![
                GuaranteeDimension::Environment,
                GuaranteeDimension::Kernel,
                GuaranteeDimension::Shape,
                GuaranteeDimension::Runtime,
                GuaranteeDimension::Topology,
            ],
            covered_shapes: vec!["correctness-range".to_owned()],
            residual_risks: Vec::new(),
        };
        let rewrite_trace = vec![crate::PassTrace {
            contract: RewritePassContract::new(
                "artifact-emission-shaping",
                RewritePhase::EmitArtifacts,
                vec!["guarantee-envelope", "artifact-requirements"],
                vec!["artifact-closure"],
                vec!["closure-manifest-derived-from-plan"],
            ),
            before_identity: normalized_request.identity.to_string(),
            after_identity: "after".to_owned(),
            matched_rules: vec!["closure-manifest-derived-from-plan".to_owned()],
            repairs: Vec::new(),
            invalidated_assumptions: Vec::new(),
            validated_invariants: vec!["closure-manifest-derived-from-plan".to_owned()],
            violations: Vec::new(),
        }];
        let capability_identity =
            canonical_hash(&guarantee_evidence.capability_witnesses).expect("capability identity");
        let backend_registry_identity =
            canonical_hash(&backend_registry).expect("backend registry identity");
        let backend_extension_identity = canonical_hash(&BackendExtensionFingerprint::from_plan(
            &selected_backends,
            &compile_regions,
        ))
        .expect("backend extension identity");
        let portability_identity = canonical_hash(&PortabilityFingerprint::from_plan(
            normalized_request.cache_policy.namespace.clone(),
            normalized_request.cache_policy.allow_cross_machine_reuse,
            normalized_request.topology.clone(),
            &artifact_requirements,
        ))
        .expect("portability identity");
        let backend_decision = crate::BackendDecisionPlan {
            build_profile_identity: canonical_hash(&normalized_request.optimization_policy)
                .expect("build profile identity"),
            entries: vec![
                crate::BackendDecisionEntry {
                    family: BackendFamily::FlashInfer,
                    technically_available: true,
                    selected_for_deployment: true,
                    reachable_from_model_family: true,
                    reachable_from_materialization_plan: true,
                    runtime_reachable: true,
                    build_technically_possible: true,
                    chosen_acquisition: Some(ArtifactAcquisition::VendorPrebuilt),
                    required_witnesses: vec!["flashinfer.prebuilt".to_owned()],
                    satisfied_witnesses: vec!["flashinfer.prebuilt".to_owned()],
                    accepted_reasons: vec![
                        "admissible backend proof available".to_owned(),
                        "selected by deployment profile".to_owned(),
                    ],
                    rejected_reasons: Vec::new(),
                    reachable_compile_regions: vec![
                        "prefill_attention".to_owned(),
                        "decode_attention".to_owned(),
                    ],
                    reachable_artifact_scopes: vec!["prefill_attention".to_owned()],
                    reachable_warmup_scopes: vec![
                        "warmup:correctness-range:prefill_attention".to_owned(),
                        "warmup:performance-range:decode_attention".to_owned(),
                    ],
                    pass_through_optimizations: vec![
                        "native autotune cache reuse".to_owned(),
                        "vendored sparse-MLA warmup".to_owned(),
                    ],
                },
                crate::BackendDecisionEntry {
                    family: BackendFamily::Triton,
                    technically_available: false,
                    selected_for_deployment: true,
                    reachable_from_model_family: false,
                    reachable_from_materialization_plan: false,
                    runtime_reachable: false,
                    build_technically_possible: false,
                    chosen_acquisition: Some(ArtifactAcquisition::LocalAotBuild),
                    required_witnesses: Vec::new(),
                    satisfied_witnesses: Vec::new(),
                    accepted_reasons: vec!["selected as secondary backend".to_owned()],
                    rejected_reasons: vec![
                        "backend registry has no technical capability entry".to_owned(),
                    ],
                    reachable_compile_regions: Vec::new(),
                    reachable_artifact_scopes: Vec::new(),
                    reachable_warmup_scopes: Vec::new(),
                    pass_through_optimizations: vec![
                        "piecewise compile cache".to_owned(),
                        "vendored Triton warmup".to_owned(),
                    ],
                },
            ],
            extension_manifests: vec![
                crate::BackendExtensionManifest {
                    extension_key: "flashinfer".to_owned(),
                    binary_name: "flashinfer_extension.so".to_owned(),
                    backend_family: BackendFamily::FlashInfer,
                    model_repositories: vec!["meta-llama/Llama-3.1-8B-Instruct".to_owned()],
                    build_technically_possible: true,
                    runtime_reachable: true,
                    reachable_compile_regions: vec![
                        "prefill_attention".to_owned(),
                        "decode_attention".to_owned(),
                    ],
                    reachable_artifact_scopes: vec!["prefill_attention".to_owned()],
                    artifact_classes: vec!["compiled-graph".to_owned()],
                },
                crate::BackendExtensionManifest {
                    extension_key: "triton".to_owned(),
                    binary_name: "triton_kernel_pack.so".to_owned(),
                    backend_family: BackendFamily::Triton,
                    model_repositories: vec!["meta-llama/Llama-3.1-8B-Instruct".to_owned()],
                    build_technically_possible: false,
                    runtime_reachable: false,
                    reachable_compile_regions: Vec::new(),
                    reachable_artifact_scopes: Vec::new(),
                    artifact_classes: Vec::new(),
                },
            ],
        };
        let backend_decision_identity =
            canonical_hash(&backend_decision).expect("backend decision identity");
        let structural_identity = StructuralIdentity {
            request_identity: normalized_request.identity.clone(),
            optimization_identity: canonical_hash(&normalized_request.optimization_policy)
                .expect("optimization identity"),
            backend_decision_identity: backend_decision_identity.clone(),
            backend_registry_identity,
            shape_envelope_identity: shape_envelope_identity,
            compile_region_identity: canonical_hash(&compile_regions).expect("region identity"),
            capability_identity: capability_identity.clone(),
            abi_identity: abi_identity.clone(),
            backend_extension_identity: backend_extension_identity.clone(),
            portability_identity: portability_identity.clone(),
            artifact_identity: canonical_hash(&artifact_requirements).expect("artifact identity"),
            evidence_identity: canonical_hash(&guarantee_evidence).expect("evidence identity"),
            plan_identity: canonical_hash(&(
                &selected_backends,
                &backend_decision_identity,
                &compile_regions,
                &shape_envelope,
                &artifact_requirements,
                &materialization_graph,
                &guarantee_envelope,
                &normalized_request.identity,
                &capability_identity,
                &abi_identity,
                &backend_extension_identity,
                &portability_identity,
            ))
            .expect("plan identity"),
        };
        let plan = crate::ResolvedBuildPlan {
            normalized_request,
            requested_readiness: None,
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
            rewrite_trace: rewrite_trace.clone(),
            backend_decision,
            structural_identity,
        };
        let verification_report = plan.validate();
        let diagnostics = DiagnosticsDocument {
            schema_version: SchemaVersion::current(),
            plan_identity: plan.structural_identity.plan_identity.clone(),
            diagnostics: vec![crate::StructuredDiagnostic {
                code: "verified_bundle".to_owned(),
                category: crate::DiagnosticCategory::OperatorDx,
                severity: crate::diagnostics::DiagnosticSeverity::Info,
                evidence: Vec::new(),
                confidence: crate::DiagnosticConfidence::High,
                likely_root_cause: "fixture".to_owned(),
                next_action: "none".to_owned(),
                auto_fix_allowed: false,
            }],
        };
        let rewrite_trace = RewriteTraceDocument {
            schema_version: SchemaVersion::current(),
            plan_identity: plan.structural_identity.plan_identity.clone(),
            passes: rewrite_trace,
        };
        let optimization_explain = crate::OptimizationExplainDocument::from_plan(&plan);
        let materialization_report = MaterializationExecutionReport {
            schema_version: SchemaVersion::current(),
            plan_identity: plan.structural_identity.plan_identity.clone(),
            artifact_root: "artifacts".to_owned(),
            cache_root: "/tmp/cache".to_owned(),
            node_root: "materialization/nodes".to_owned(),
            wave_root: "materialization/waves".to_owned(),
            artifact_count: 1,
            executed_artifact_count: 1,
            reused_artifact_count: 0,
            wall_clock_ms: 2500,
            total_bytes_written: 24_000_000,
            total_compile_ms: 2500,
            total_transfer_ms: 0,
            total_rebuild_ms: 2500,
            unique_artifact_count: 1,
            duplicate_artifact_count: 0,
            unique_artifact_bytes: 24_000_000,
            duplicate_artifact_bytes: 0,
            artifact_deserialization_ms: 0,
            duplicate_rank_local_compile_count: 0,
            duplicate_rank_local_load_count: 0,
            closure_expansion: crate::ClosureExpansionRecord {
                requested_regions: vec!["prefill_attention".to_owned()],
                requested_artifact_scopes: vec!["prefill_attention".to_owned()],
                requested_backend_families: vec!["flashinfer".to_owned()],
                requested_cache_namespaces: vec!["compile-cache".to_owned()],
                requested_warmup_scopes: vec!["prefill_attention".to_owned()],
                expanded_regions: vec!["prefill_attention".to_owned()],
                expanded_artifact_scopes: vec!["prefill_attention".to_owned()],
                expanded_warmup_scopes: vec![
                    "warmup:correctness-range:prefill_attention".to_owned(),
                    "warmup:performance-range:decode_attention".to_owned(),
                ],
                deterministically_closed: true,
            },
            closure_outcome: StartupClosureOutcome::FullCompileClosure,
            readiness: ReadinessObservation {
                requested_readiness: ObservedReadinessLevel::Correctness,
                achieved_readiness: ObservedReadinessLevel::Correctness,
                blocking_warmups_complete: true,
                early_serve_frontier_complete: true,
                deferred_warmups_complete: true,
            },
            runtime_jit_observations: vec![RuntimeJitObservation {
                surface_name: "fixture".to_owned(),
                status: RuntimeJitObservationStatus::Bounded,
                observed_artifacts: vec!["prefill_attention".to_owned()],
                observed_warmup_proofs: vec![
                    "warmup:correctness-range:prefill_attention".to_owned(),
                ],
                contradiction_reasons: Vec::new(),
            }],
            verify_replay_compile_free: true,
            verify_replay_status: crate::ValidationStatus::Passed,
            artifacts: vec![MaterializedArtifactRecord {
                storage_key: "fixture-storage-key".to_owned(),
                manifest_identity: "flashinfer:CompiledGraph:prefill_attention".to_owned(),
                scope: "prefill_attention".to_owned(),
                class: ArtifactClass::CompiledGraph,
                backend: BackendFamily::FlashInfer,
                region_stable_identity: Some(
                    canonical_hash(&"fixture-prefill-stable")
                        .expect("fixture prefill stable identity"),
                ),
                region_equivalence_identity: Some(
                    canonical_hash(&"fixture-prefill-equivalence")
                        .expect("fixture prefill equivalence identity"),
                ),
                cache_sharing: Some(crate::RegionCacheSharing::ContentAddressed),
                cache_namespace: "compile-cache".to_owned(),
                invalidation_domain: "prefill_attention".to_owned(),
                acquisition: ArtifactAcquisition::VendorPrebuilt,
                rank_disposition: RankDisposition::Shared,
                preferred_fanout_strategy: FanoutStrategy::BroadcastFromLeader,
                disposition: MaterializationDisposition::Executed,
                relative_path: "artifacts/fixture-storage-key/artifact.json".to_owned(),
                cache_relative_path: "compile-cache/fixture-storage-key/artifact.json".to_owned(),
                bytes_written: 24_000_000,
                deserialization_ms: 0,
                rank_count: 2,
                compile_ms: 2500,
                transfer_ms: 0,
                rebuild_ms: 2500,
                source_anchors: vec![SourceAnchor {
                    file: "fixture.py".to_owned(),
                    line: 1,
                }],
            }],
            nodes: vec![MaterializationNodeRecord {
                node_name: "warmup:correctness-range:prefill_attention".to_owned(),
                wave: 3,
                kind: crate::MaterializationNodeKind::Warmup,
                queue: QueueKind::Warmup,
                disposition: MaterializationDisposition::Executed,
                dependency_nodes: vec!["artifact:compiled-graph:fixture".to_owned()],
                outputs: vec!["coverage:correctness-range:prefill_attention".to_owned()],
                relative_path: "materialization/nodes/warmup-correctness.json".to_owned(),
                duration_ms: 10,
                bytes_written: 0,
            }],
            waves: vec![MaterializationWaveRecord {
                wave_name: "wave-3".to_owned(),
                queue: QueueKind::Warmup,
                discipline: QueueDiscipline::ParallelPerRank,
                scheduling_mode: crate::MaterializationSchedulingMode::Parallel,
                max_parallelism: 2,
                node_names: vec!["warmup:correctness-range:prefill_attention".to_owned()],
                relative_path: "materialization/waves/wave-3.json".to_owned(),
                duration_ms: 10,
                bytes_written: 0,
            }],
        };
        let replay_proof =
            ReplayProofDocument::from_plan_and_materialization(&plan, &materialization_report)
                .expect("replay proof");
        let vllm_integration = crate::VllmIntegrationDocument {
            schema_version: SchemaVersion::current(),
            plan_identity: plan.structural_identity.plan_identity.clone(),
            engine_root: "/tmp/vllm".to_owned(),
            engine_revision: "deadbeef".to_owned(),
            surfaces: vec![crate::VllmIntegrationSurface {
                id: "compile-region:prefill_attention".to_owned(),
                scope_kind: crate::IntegrationScopeKind::CompileRegion,
                scope_name: "prefill_attention".to_owned(),
                backend: Some(BackendFamily::FlashInfer),
                cache_namespace: Some("compile-cache".to_owned()),
                warmup_scope: Some("prefill_attention".to_owned()),
                rationale: "fixture surface".to_owned(),
                preserved_inputs: vec!["CompilationConfig".to_owned()],
                preserved_abstractions: vec!["Graph and region boundaries".to_owned()],
                isolation: crate::VllmIsolationContract {
                    disposition: crate::VllmIsolationDisposition::ContextBound,
                    subset_build_valid: true,
                    direct_entrypoint_invocable: false,
                    required_context: vec!["Worker context".to_owned()],
                    blockers: Vec::new(),
                    evidence: SourceEvidence {
                        summary: "fixture".to_owned(),
                        anchors: Vec::new(),
                    },
                },
                primary: crate::VllmCallableTarget {
                    module: "vllm.fixture".to_owned(),
                    callable: "compile_prefill".to_owned(),
                    summary: "fixture callable".to_owned(),
                    evidence: SourceEvidence {
                        summary: "fixture".to_owned(),
                        anchors: Vec::new(),
                    },
                },
                auxiliary: Vec::new(),
            }],
            replay_roots: vec![crate::VllmReplayRoot {
                id: "replay-root:compile-region:prefill_attention".to_owned(),
                root_kind: crate::VllmReplayRootKind::CompileRegion,
                surface_id: "compile-region:prefill_attention".to_owned(),
                scope_name: "prefill_attention".to_owned(),
                root_key: plan.structural_identity.plan_identity.clone(),
                cache_namespace: Some("compile-cache".to_owned()),
                warmup_scope: Some("prefill_attention".to_owned()),
                replay_boundary: "compile-region:prefill_attention".to_owned(),
                manifest_paths: vec![
                    "compile_replay_manifest.json".to_owned(),
                    "graph_artifact_store.json".to_owned(),
                    "warmup_materialization_manifest.json".to_owned(),
                ],
            }],
        };
        let vllm_entrypoints = crate::VllmEntrypointDocument {
            schema_version: SchemaVersion::current(),
            plan_identity: plan.structural_identity.plan_identity.clone(),
            engine_root: "/tmp/vllm".to_owned(),
            engine_revision: "deadbeef".to_owned(),
            entrypoints: vec![crate::VllmEntrypoint {
                id: "entrypoint:prefill_attention".to_owned(),
                surface_id: "compile-region:prefill_attention".to_owned(),
                scope_name: "prefill_attention".to_owned(),
                isolation: crate::VllmIsolationContract {
                    disposition: crate::VllmIsolationDisposition::ContextBound,
                    subset_build_valid: true,
                    direct_entrypoint_invocable: false,
                    required_context: vec!["Worker context".to_owned()],
                    blockers: Vec::new(),
                    evidence: SourceEvidence {
                        summary: "fixture".to_owned(),
                        anchors: Vec::new(),
                    },
                },
                context_kind: crate::VllmContextKind::Worker,
                call_strategy: crate::VllmCallStrategy::ModuleFunctionWithContext,
                callable: crate::VllmCallableTarget {
                    module: "vllm.fixture".to_owned(),
                    callable: "compile_prefill".to_owned(),
                    summary: "fixture callable".to_owned(),
                    evidence: SourceEvidence {
                        summary: "fixture".to_owned(),
                        anchors: Vec::new(),
                    },
                },
                args: std::collections::BTreeMap::new(),
                required_env: Vec::new(),
                preserved_inputs: vec!["CompilationConfig".to_owned()],
                preserved_abstractions: vec!["Graph and region boundaries".to_owned()],
                summary: "fixture entrypoint".to_owned(),
                manifest_path: "vllm-entrypoints/surfaces/prefill_attention.json".to_owned(),
                wrapper_path: "vllm-entrypoints/prefill_attention.sh".to_owned(),
            }],
        };
        let soc_plan = crate::SocPlanDocument {
            schema_version: SchemaVersion::current(),
            plan_identity: plan.structural_identity.plan_identity.clone(),
            derivation_strategy: "derived_from_resolved_build_plan_and_vllm_integration".to_owned(),
            selectors: crate::SocSelectorSnapshot {
                requested_regions: vec!["prefill_attention".to_owned()],
                requested_artifact_scopes: Vec::new(),
                requested_backend_families: vec!["flashinfer".to_owned()],
                requested_topology_scopes: vec!["shared".to_owned()],
                requested_cache_namespaces: vec!["compile-cache".to_owned()],
                requested_warmup_scopes: vec!["prefill_attention".to_owned()],
                requested_readiness: "correctness".to_owned(),
            },
            namespaces: vec![crate::SocNamespacePlan {
                namespace: "compile-cache".to_owned(),
                scope_kind: crate::IntegrationScopeKind::CompileRegion,
                materialization_mode: crate::SocMaterializationMode::EagerBlocking,
                subset_build_valid: true,
                direct_entrypoint_invocable: false,
                artifact_scopes: vec!["prefill_attention".to_owned()],
                artifact_classes: vec!["compiled-graph".to_owned(), "triton-binary".to_owned()],
                required_artifacts: vec![
                    "compiled-graph:prefill_attention".to_owned(),
                    "triton-binary:prefill_attention".to_owned(),
                ],
                warmup_scopes: vec!["prefill_attention".to_owned()],
                warmup_proof_ids: vec!["warmup:correctness-range:prefill_attention".to_owned()],
                replay_root_ids: vec!["replay-root:compile-region:prefill_attention".to_owned()],
                source_surface_ids: vec!["compile-region:prefill_attention".to_owned()],
                source_callables: vec!["vllm.fixture::compile_prefill".to_owned()],
                rationale: "fixture surface".to_owned(),
            }],
            replay_root_ids: vec!["replay-root:compile-region:prefill_attention".to_owned()],
            shared_abstractions: vec!["Graph and region boundaries".to_owned()],
        };

        ReplayBundle {
            build_plan: plan.clone(),
            artifact_closure: crate::ArtifactClosure {
                plan_identity: plan.structural_identity.plan_identity.clone(),
                artifacts: artifact_manifest,
            },
            verification_report,
            diagnostics,
            rewrite_trace,
            optimization_explain,
            backend_decision: crate::BackendDecisionDocument::from_plan(&plan),
            materialization_report,
            replay_proof,
            vllm_integration,
            soc_plan,
            vllm_entrypoints,
        }
    }

    #[test]
    fn replay_bundle_round_trips() {
        let dir = tempdir().expect("tempdir");
        let bundle = sample_bundle();
        bundle.write_to(dir.path()).expect("write bundle");
        let loaded = ReplayBundle::load_from(dir.path()).expect("load bundle");
        assert_eq!(loaded.build_plan, bundle.build_plan);
        assert_eq!(loaded.artifact_closure, bundle.artifact_closure);
        assert_eq!(loaded.verification_report, bundle.verification_report);
        assert_eq!(loaded.diagnostics, bundle.diagnostics);
        assert_eq!(loaded.rewrite_trace, bundle.rewrite_trace);
        assert_eq!(loaded.backend_decision, bundle.backend_decision);
        assert_eq!(loaded.materialization_report, bundle.materialization_report);
        assert_eq!(loaded.replay_proof, bundle.replay_proof);
        assert_eq!(loaded.vllm_integration, bundle.vllm_integration);
        assert_eq!(loaded.soc_plan, bundle.soc_plan);
        assert_eq!(loaded.vllm_entrypoints, bundle.vllm_entrypoints);
    }

    #[test]
    fn replay_bundle_rejects_digest_tampering() {
        let dir = tempdir().expect("tempdir");
        let bundle = sample_bundle();
        bundle.write_to(dir.path()).expect("write bundle");
        fs::write(dir.path().join("diagnostics.json"), "{}").expect("tamper diagnostics");
        let err = ReplayBundle::load_from(dir.path()).expect_err("tamper should fail");
        assert!(matches!(err, ReplayBundleError::DigestMismatch { .. }));
    }

    #[test]
    fn replay_bundle_rejects_identity_mismatch() {
        let dir = tempdir().expect("tempdir");
        let bundle = sample_bundle();
        let metadata = bundle.write_to(dir.path()).expect("write bundle");
        let mut diagnostics: DiagnosticsDocument = serde_json::from_str(
            &fs::read_to_string(dir.path().join("diagnostics.json")).expect("read diagnostics"),
        )
        .expect("parse diagnostics");
        diagnostics.plan_identity = canonical_hash(&"other-plan").expect("mismatch identity");
        fs::write(
            dir.path().join("diagnostics.json"),
            serde_json::to_string_pretty(&diagnostics).expect("serialize diagnostics"),
        )
        .expect("write diagnostics");

        let mut digests = metadata.file_digests.clone();
        digests.insert(
            "diagnostics.json".to_owned(),
            digest(
                fs::read(dir.path().join("diagnostics.json"))
                    .expect("read diagnostics for digest")
                    .as_slice(),
            ),
        );
        let metadata = ReplayBundleMetadata {
            schema_version: metadata.schema_version,
            plan_identity: metadata.plan_identity,
            file_digests: digests,
            replay_entrypoint: metadata.replay_entrypoint,
        };
        fs::write(
            dir.path().join("bundle_metadata.json"),
            crate::canonical_json(&metadata).expect("serialize metadata"),
        )
        .expect("write metadata");

        let err = ReplayBundle::load_from(dir.path()).expect_err("identity mismatch should fail");
        assert!(matches!(err, ReplayBundleError::IdentityMismatch { .. }));
    }

    #[test]
    fn replay_bundle_rejects_invalid_reuse_after_digest_refresh() {
        let dir = tempdir().expect("tempdir");
        let bundle = sample_bundle();
        let metadata = bundle.write_to(dir.path()).expect("write bundle");
        let mut build_plan: crate::ResolvedBuildPlan = serde_json::from_str(
            &fs::read_to_string(dir.path().join("buildplan.json")).expect("read buildplan"),
        )
        .expect("parse buildplan");
        build_plan.guarantee_evidence.artifact_manifest[0].scope = "wrong_scope".to_owned();
        fs::write(
            dir.path().join("buildplan.json"),
            serde_json::to_string_pretty(&build_plan).expect("serialize buildplan"),
        )
        .expect("write buildplan");

        let mut digests = metadata.file_digests.clone();
        digests.insert(
            "buildplan.json".to_owned(),
            digest(
                fs::read(dir.path().join("buildplan.json"))
                    .expect("read buildplan for digest")
                    .as_slice(),
            ),
        );
        let metadata = ReplayBundleMetadata {
            schema_version: metadata.schema_version,
            plan_identity: metadata.plan_identity,
            file_digests: digests,
            replay_entrypoint: metadata.replay_entrypoint,
        };
        fs::write(
            dir.path().join("bundle_metadata.json"),
            crate::canonical_json(&metadata).expect("serialize metadata"),
        )
        .expect("write metadata");

        let err = ReplayBundle::load_from(dir.path()).expect_err("invalid reuse should fail");
        assert!(matches!(err, ReplayBundleError::VerificationMismatch));
    }

    #[test]
    fn replay_bundle_rejects_backend_surface_widening_after_digest_refresh() {
        let dir = tempdir().expect("tempdir");
        let bundle = sample_bundle();
        let metadata = bundle.write_to(dir.path()).expect("write bundle");
        let mut build_plan: crate::ResolvedBuildPlan = serde_json::from_str(
            &fs::read_to_string(dir.path().join("buildplan.json")).expect("read buildplan"),
        )
        .expect("parse buildplan");
        build_plan.compile_regions[0].family = crate::BackendFamily::AotInductor;
        fs::write(
            dir.path().join("buildplan.json"),
            serde_json::to_string_pretty(&build_plan).expect("serialize buildplan"),
        )
        .expect("write buildplan");

        let mut digests = metadata.file_digests.clone();
        digests.insert(
            "buildplan.json".to_owned(),
            digest(
                fs::read(dir.path().join("buildplan.json"))
                    .expect("read buildplan for digest")
                    .as_slice(),
            ),
        );
        let metadata = ReplayBundleMetadata {
            schema_version: metadata.schema_version,
            plan_identity: metadata.plan_identity,
            file_digests: digests,
            replay_entrypoint: metadata.replay_entrypoint,
        };
        fs::write(
            dir.path().join("bundle_metadata.json"),
            crate::canonical_json(&metadata).expect("serialize metadata"),
        )
        .expect("write metadata");

        let err = ReplayBundle::load_from(dir.path()).expect_err("backend widening should fail");
        assert!(matches!(err, ReplayBundleError::VerificationMismatch));
    }

    #[test]
    fn replay_bundle_rejects_backend_decision_drift_after_digest_refresh() {
        let dir = tempdir().expect("tempdir");
        let bundle = sample_bundle();
        let metadata = bundle.write_to(dir.path()).expect("write bundle");
        let mut backend_decision: crate::BackendDecisionDocument = serde_json::from_str(
            &fs::read_to_string(dir.path().join("backend_decision.json"))
                .expect("read backend decision"),
        )
        .expect("parse backend decision");
        backend_decision.entries[0].runtime_reachable = false;
        fs::write(
            dir.path().join("backend_decision.json"),
            serde_json::to_string_pretty(&backend_decision).expect("serialize backend decision"),
        )
        .expect("write backend decision");

        let mut digests = metadata.file_digests.clone();
        digests.insert(
            "backend_decision.json".to_owned(),
            digest(
                fs::read(dir.path().join("backend_decision.json"))
                    .expect("read backend decision for digest")
                    .as_slice(),
            ),
        );
        let metadata = ReplayBundleMetadata {
            schema_version: metadata.schema_version,
            plan_identity: metadata.plan_identity,
            file_digests: digests,
            replay_entrypoint: metadata.replay_entrypoint,
        };
        fs::write(
            dir.path().join("bundle_metadata.json"),
            crate::canonical_json(&metadata).expect("serialize metadata"),
        )
        .expect("write metadata");

        let err = ReplayBundle::load_from(dir.path()).expect_err("backend drift should fail");
        assert!(matches!(err, ReplayBundleError::BackendDecisionMismatch));
    }

    #[test]
    fn replay_bundle_rejects_replay_proof_mismatch_after_digest_refresh() {
        let dir = tempdir().expect("tempdir");
        let bundle = sample_bundle();
        let metadata = bundle.write_to(dir.path()).expect("write bundle");
        let mut replay_proof: ReplayProofDocument = serde_json::from_str(
            &fs::read_to_string(dir.path().join("replay_proof.json")).expect("read replay proof"),
        )
        .expect("parse replay proof");
        replay_proof.realization_mode = crate::ArtifactRealizationMode::Mixed;
        fs::write(
            dir.path().join("replay_proof.json"),
            serde_json::to_string_pretty(&replay_proof).expect("serialize replay proof"),
        )
        .expect("write replay proof");

        let mut digests = metadata.file_digests.clone();
        digests.insert(
            "replay_proof.json".to_owned(),
            digest(
                fs::read(dir.path().join("replay_proof.json"))
                    .expect("read replay proof for digest")
                    .as_slice(),
            ),
        );
        let metadata = ReplayBundleMetadata {
            schema_version: metadata.schema_version,
            plan_identity: metadata.plan_identity,
            file_digests: digests,
            replay_entrypoint: metadata.replay_entrypoint,
        };
        fs::write(
            dir.path().join("bundle_metadata.json"),
            crate::canonical_json(&metadata).expect("serialize metadata"),
        )
        .expect("write metadata");

        let err = ReplayBundle::load_from(dir.path()).expect_err("replay proof mismatch");
        assert!(matches!(err, ReplayBundleError::ReplayProofMismatch));
    }
}
