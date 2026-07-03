use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use hex::ToHex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    ArtifactClosure, ArtifactManifestEntry, CanonicalError, CanonicalHash, DiagnosticsDocument,
    ResolvedBuildPlan, RewriteTraceDocument, VerificationReport, canonical_json,
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
        ];
        let mut file_digests = BTreeMap::new();
        for (name, content) in files {
            fs::write(dir.join(name), content.as_bytes())?;
            file_digests.insert(name.to_owned(), digest(content.as_bytes()));
        }
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
        if build_plan.validate() != verification_report {
            return Err(ReplayBundleError::VerificationMismatch);
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
        DiagnosticsDocument, EngineSource, ExecutionTopology, FailureMode, GuaranteeDimension,
        GuaranteeEnvelope, GuaranteeEvidence, GuaranteeLevel, GuaranteeTarget,
        MaterializationGraph, MaterializationNode, MaterializationNodeKind, MaterializationWave,
        ModelRef, PortabilityFingerprint, QueueKind, RangeIntent, RankDisposition, RawRequest,
        RequestedEnvironment, RewritePassContract, RewritePhase, RewriteTraceDocument,
        SchemaVersion, ShapeEnvelope, ShapeEnvelopeNode, ShapePoint, ShapePolicy, ShapeRange,
        SourceEvidence, StructuralIdentity, TargetEngine, WarmupObligation, WarmupPolicy,
        WaveEstimate, canonical_hash,
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
            }],
            leader_assignments: Vec::new(),
            early_serve_frontier: vec!["verify:correctness".to_owned()],
            late_bindings: vec![("cache_root".to_owned(), "host://cache/sock".to_owned())],
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
        let structural_identity = StructuralIdentity {
            request_identity: normalized_request.identity.clone(),
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

        ReplayBundle {
            build_plan: plan.clone(),
            artifact_closure: crate::ArtifactClosure {
                plan_identity: plan.structural_identity.plan_identity.clone(),
                artifacts: artifact_manifest,
            },
            verification_report,
            diagnostics,
            rewrite_trace,
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
}
