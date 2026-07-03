use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sock_core::{
    ArtifactAcquisition, ArtifactClass, ArtifactRequirement, CanonicalError,
    ClosureExpansionRecord, FanoutStrategy, MaterializationDisposition,
    MaterializationExecutionReport, MaterializationNode, MaterializationNodeKind,
    MaterializationNodeRecord, MaterializationSchedulingMode, MaterializationWave,
    MaterializationWaveRecord, MaterializedArtifactRecord, ObservedReadinessLevel, QueueDiscipline,
    RankDisposition, ReadinessObservation, RuntimeJitObservation, RuntimeJitObservationStatus,
    SchemaVersion, SourceAnchor, StartupClosureOutcome, WarmupObligation, canonical_hash,
    canonical_json,
};
use thiserror::Error;

use crate::{BuildReadiness, BuildScope, PlanningOutcome, vllm};

#[derive(Debug, Error)]
pub enum MaterializationError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("canonical error: {0}")]
    Canonical(#[from] CanonicalError),
    #[error("materialization dependency missing for node {node}")]
    MissingDependency { node: String },
    #[error("materialization node {node} does not map to a known artifact requirement")]
    UnknownArtifactNode { node: String },
}

#[derive(Debug, Default)]
pub struct MaterializationExecutor;

#[derive(Debug, Clone)]
struct ArtifactDeserialization {
    document: MaterializedArtifactDocument,
    duration_ms: u64,
}

#[derive(Debug, Clone)]
pub struct StorageRoots {
    pub bundle_root: std::path::PathBuf,
    pub cache_root: std::path::PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct MaterializedArtifactDocument {
    schema_version: SchemaVersion,
    storage_key: String,
    manifest_identity: String,
    scope: String,
    class: sock_core::ArtifactClass,
    backend: sock_core::BackendFamily,
    cache_namespace: String,
    invalidation_domain: String,
    acquisition: ArtifactAcquisition,
    rank_disposition: RankDisposition,
    engine_root: String,
    engine_revision: String,
    source_anchors: Vec<SourceAnchor>,
    admissibility_summary: String,
    observed_compile_ms: u64,
    observed_transfer_ms: u64,
    observed_rebuild_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct FanoutReceipt {
    schema_version: SchemaVersion,
    plan_identity: sock_core::CanonicalHash,
    node_name: String,
    dependency_nodes: Vec<String>,
    ranks: Vec<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct WarmupReceipt {
    schema_version: SchemaVersion,
    plan_identity: sock_core::CanonicalHash,
    node_name: String,
    dependency_nodes: Vec<String>,
    required_artifacts: Vec<String>,
    proof: sock_core::WarmupCoverageProof,
}

#[derive(Debug, Clone)]
struct NodeExecutionResult {
    record: MaterializationNodeRecord,
    artifact: Option<MaterializedArtifactRecord>,
    compile_ms: u64,
    transfer_ms: u64,
}

impl MaterializationExecutor {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    pub fn execute(
        &self,
        outcome: &PlanningOutcome,
        scope: &BuildScope,
        roots: &StorageRoots,
    ) -> Result<MaterializationExecutionReport, MaterializationError> {
        let started = Instant::now();
        let artifact_root = roots.bundle_root.join("artifacts");
        let cache_root = roots.cache_root.clone();
        let node_root = roots.bundle_root.join("materialization").join("nodes");
        let wave_root = roots.bundle_root.join("materialization").join("waves");
        let warmup_root = roots.bundle_root.join("warmup");
        fs::create_dir_all(&artifact_root)?;
        fs::create_dir_all(&cache_root)?;
        fs::create_dir_all(&node_root)?;
        fs::create_dir_all(&wave_root)?;
        fs::create_dir_all(&warmup_root)?;

        let requirement_index = outcome
            .plan
            .artifact_requirements
            .iter()
            .map(|requirement| Ok((artifact_handle(requirement)?, requirement.clone())))
            .collect::<Result<BTreeMap<_, _>, MaterializationError>>()?;
        let warmup_index = outcome
            .plan
            .warmup_obligations
            .iter()
            .map(|obligation| {
                (
                    format!("warmup:{}:{}", obligation.node_name, obligation.region_name),
                    obligation.clone(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let node_index = outcome
            .plan
            .materialization_graph
            .nodes
            .iter()
            .map(|node| (node.name.clone(), node.clone()))
            .collect::<BTreeMap<_, _>>();

        let mut completed_nodes = BTreeSet::new();
        let mut artifacts = BTreeMap::new();
        let mut node_records = Vec::new();
        let mut wave_records = Vec::new();
        let mut total_bytes_written = 0_u64;
        let mut total_compile_ms = 0_u64;
        let mut total_transfer_ms = 0_u64;
        let mut total_rebuild_ms = 0_u64;
        let mut artifact_deserialization_ms = 0_u64;

        for wave in &outcome.plan.materialization_graph.waves {
            let wave_started = Instant::now();
            let wave_results = self.execute_wave(
                outcome,
                wave,
                &node_index,
                &requirement_index,
                &warmup_index,
                &artifact_root,
                &cache_root,
                &node_root,
                &wave_root,
                &warmup_root,
                &completed_nodes,
            )?;
            let mut wave_bytes = 0_u64;
            for result in wave_results {
                wave_bytes = wave_bytes.saturating_add(result.record.bytes_written);
                total_bytes_written =
                    total_bytes_written.saturating_add(result.record.bytes_written);
                total_compile_ms = total_compile_ms.saturating_add(result.compile_ms);
                total_transfer_ms = total_transfer_ms.saturating_add(result.transfer_ms);
                if let Some(artifact) = result.artifact {
                    total_rebuild_ms = total_rebuild_ms.saturating_add(artifact.rebuild_ms);
                    artifact_deserialization_ms = artifact_deserialization_ms
                        .saturating_add(artifact_deserialization_ms_for(&artifact));
                    artifacts.insert(artifact.storage_key.clone(), artifact);
                }
                completed_nodes.insert(result.record.node_name.clone());
                node_records.push(result.record);
            }

            let wave_record = MaterializationWaveRecord {
                wave_name: wave.name.clone(),
                queue: wave.queue,
                discipline: wave.execution_contract.discipline,
                scheduling_mode: scheduling_mode(wave.execution_contract.discipline),
                max_parallelism: wave_max_parallelism(wave),
                node_names: wave.node_names.clone(),
                relative_path: self.write_receipt(
                    &wave_root,
                    "materialization/waves",
                    &wave.name,
                    &serde_json::json!({
                        "schema_version": SchemaVersion::current(),
                        "plan_identity": outcome.plan.structural_identity.plan_identity,
                        "node_names": wave.node_names,
                        "fulfills": wave.execution_contract.fulfills,
                        "queue": wave.queue,
                        "discipline": wave.execution_contract.discipline,
                        "scheduling_mode": scheduling_mode(wave.execution_contract.discipline),
                        "max_parallelism": wave_max_parallelism(wave),
                    }),
                )?,
                duration_ms: elapsed_ms(wave_started.elapsed()),
                bytes_written: wave_bytes,
            };
            wave_records.push(wave_record);
        }

        let mut artifact_records = artifacts.into_values().collect::<Vec<_>>();
        artifact_records.sort_by(|left, right| left.storage_key.cmp(&right.storage_key));
        node_records.sort_by(|left, right| left.node_name.cmp(&right.node_name));
        wave_records.sort_by(|left, right| left.wave_name.cmp(&right.wave_name));
        let readiness = observed_readiness(scope, outcome, &completed_nodes);
        let runtime_jit_observations =
            observed_runtime_jit(outcome, &artifact_records, &completed_nodes, &warmup_index);
        let unique_artifact_count = artifact_records.len() as u32;
        let duplicate_artifact_count = artifact_records
            .iter()
            .map(duplicate_instances_for_artifact)
            .sum::<u32>();
        let duplicate_artifact_bytes = artifact_records
            .iter()
            .map(duplicate_bytes_for_artifact)
            .sum::<u64>();
        let unique_artifact_bytes = artifact_records
            .iter()
            .map(|artifact| artifact.bytes_written)
            .sum::<u64>();
        let closure_expansion = closure_expansion(scope, outcome);
        let verify_replay_compile_free = verify_replay_compile_free(outcome);
        let closure_outcome = closure_outcome(
            &closure_expansion,
            &runtime_jit_observations,
            verify_replay_compile_free,
        );

        let report = MaterializationExecutionReport {
            schema_version: SchemaVersion::current(),
            plan_identity: outcome.plan.structural_identity.plan_identity.clone(),
            artifact_root: "artifacts".to_owned(),
            cache_root: roots.cache_root.display().to_string(),
            node_root: "materialization/nodes".to_owned(),
            wave_root: "materialization/waves".to_owned(),
            artifact_count: artifact_records.len() as u32,
            executed_artifact_count: artifact_records
                .iter()
                .filter(|artifact| artifact.disposition == MaterializationDisposition::Executed)
                .count() as u32,
            reused_artifact_count: artifact_records
                .iter()
                .filter(|artifact| artifact.disposition == MaterializationDisposition::Reused)
                .count() as u32,
            wall_clock_ms: elapsed_ms(started.elapsed()),
            total_bytes_written,
            total_compile_ms,
            total_transfer_ms,
            total_rebuild_ms,
            unique_artifact_count,
            duplicate_artifact_count,
            unique_artifact_bytes,
            duplicate_artifact_bytes,
            artifact_deserialization_ms,
            duplicate_rank_local_compile_count: artifact_records
                .iter()
                .map(duplicate_instances_for_artifact)
                .sum(),
            duplicate_rank_local_load_count: artifact_records
                .iter()
                .filter(|artifact| artifact.disposition == MaterializationDisposition::Reused)
                .map(duplicate_instances_for_artifact)
                .sum(),
            closure_expansion,
            closure_outcome,
            readiness,
            runtime_jit_observations,
            verify_replay_compile_free,
            verify_replay_status: outcome.verification.status.clone(),
            artifacts: artifact_records,
            nodes: node_records,
            waves: wave_records,
        };
        fs::write(
            roots.bundle_root.join("materialization_report.json"),
            canonical_json(&report)?.as_bytes(),
        )?;
        Ok(report)
    }

    fn materialize_artifact(
        &self,
        outcome: &PlanningOutcome,
        requirement: &ArtifactRequirement,
        artifact_root: &Path,
        cache_root: &Path,
    ) -> Result<MaterializedArtifactRecord, MaterializationError> {
        let started = Instant::now();
        let storage_key = canonical_hash(requirement)?.to_string();
        let manifest_identity = outcome
            .closure
            .artifacts
            .iter()
            .find(|artifact| {
                artifact.scope == requirement.scope && artifact.class == requirement.class
            })
            .map(|artifact| artifact.identity.clone())
            .expect("closure artifact exists");
        let cache_namespace = cache_namespace_for_scope(outcome, &requirement.scope);
        let invalidation_domain = invalidation_domain_for_scope(outcome, &requirement.scope);
        let relative_path = format!("artifacts/{storage_key}/artifact.json");
        let cache_relative_path = format!("{}/{storage_key}/artifact.json", slug(&cache_namespace));
        let absolute_path = artifact_root.join(&storage_key).join("artifact.json");
        let cache_path = cache_root
            .join(slug(&cache_namespace))
            .join(&storage_key)
            .join("artifact.json");
        fs::create_dir_all(absolute_path.parent().expect("artifact parent"))?;
        fs::create_dir_all(cache_path.parent().expect("cache parent"))?;
        let document = MaterializedArtifactDocument {
            schema_version: SchemaVersion::current(),
            storage_key: storage_key.clone(),
            manifest_identity: manifest_identity.clone(),
            scope: requirement.scope.clone(),
            class: requirement.class,
            backend: requirement.backend,
            cache_namespace: cache_namespace.clone(),
            invalidation_domain: invalidation_domain.clone(),
            acquisition: requirement.acquisition,
            rank_disposition: requirement.rank_disposition,
            engine_root: vllm::root().display().to_string(),
            engine_revision: vllm::revision().to_owned(),
            source_anchors: source_anchors_for_scope(outcome, &requirement.scope),
            admissibility_summary: requirement.admissibility.rationale.join("; "),
            observed_compile_ms: 0,
            observed_transfer_ms: 0,
            observed_rebuild_ms: 0,
        };
        let elapsed_ms_now = observed_elapsed_ms(started.elapsed());
        let is_transfer = matches!(
            requirement.acquisition,
            ArtifactAcquisition::VendorPrebuilt | ArtifactAcquisition::UpstreamCacheBundle
        );
        let observed_compile_ms = if is_transfer { 0 } else { elapsed_ms_now };
        let observed_transfer_ms = if is_transfer { elapsed_ms_now } else { 0 };
        let observed_rebuild_ms = observed_compile_ms.saturating_add(observed_transfer_ms);
        let (disposition, cache_bytes_written, rebuild_ms, deserialization_ms) =
            if cache_path.exists() {
                let existing = load_existing_document(&cache_path)?;
                if same_artifact_document_identity(&existing.document, &document) {
                    (
                        MaterializationDisposition::Reused,
                        0,
                        existing.document.observed_rebuild_ms.max(
                            existing
                                .document
                                .observed_compile_ms
                                .saturating_add(existing.document.observed_transfer_ms),
                        ),
                        existing.duration_ms,
                    )
                } else {
                    evict_invalidated_siblings(
                        cache_root,
                        &cache_namespace,
                        &invalidation_domain,
                        requirement.class,
                        &storage_key,
                    )?;
                    let mut document = document.clone();
                    document.observed_compile_ms = observed_compile_ms;
                    document.observed_transfer_ms = observed_transfer_ms;
                    document.observed_rebuild_ms = observed_rebuild_ms;
                    let content = canonical_json(&document)?;
                    write_bytes_atomically(&cache_path, content.as_bytes())?;
                    (
                        MaterializationDisposition::Executed,
                        content.len() as u64,
                        observed_rebuild_ms,
                        0,
                    )
                }
            } else {
                evict_invalidated_siblings(
                    cache_root,
                    &cache_namespace,
                    &invalidation_domain,
                    requirement.class,
                    &storage_key,
                )?;
                let mut document = document.clone();
                document.observed_compile_ms = observed_compile_ms;
                document.observed_transfer_ms = observed_transfer_ms;
                document.observed_rebuild_ms = observed_rebuild_ms;
                let content = canonical_json(&document)?;
                write_bytes_atomically(&cache_path, content.as_bytes())?;
                (
                    MaterializationDisposition::Executed,
                    content.len() as u64,
                    observed_rebuild_ms,
                    0,
                )
            };
        copy_atomically(&cache_path, &absolute_path)?;
        let bundle_bytes_written = if disposition == MaterializationDisposition::Reused {
            file_size(&absolute_path)?
        } else {
            cache_bytes_written.max(file_size(&absolute_path)?)
        };
        Ok(MaterializedArtifactRecord {
            storage_key,
            manifest_identity,
            scope: requirement.scope.clone(),
            class: requirement.class,
            backend: requirement.backend,
            cache_namespace,
            invalidation_domain,
            acquisition: requirement.acquisition,
            rank_disposition: requirement.rank_disposition,
            preferred_fanout_strategy: observed_preferred_strategy(requirement, rebuild_ms),
            disposition,
            relative_path,
            cache_relative_path,
            bytes_written: bundle_bytes_written,
            deserialization_ms,
            rank_count: 1,
            compile_ms: if disposition == MaterializationDisposition::Executed {
                observed_compile_ms
            } else {
                0
            },
            transfer_ms: if disposition == MaterializationDisposition::Executed {
                observed_transfer_ms
            } else {
                0
            },
            rebuild_ms,
            source_anchors: source_anchors_for_scope(outcome, &requirement.scope),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_wave(
        &self,
        outcome: &PlanningOutcome,
        wave: &MaterializationWave,
        node_index: &BTreeMap<String, MaterializationNode>,
        requirement_index: &BTreeMap<String, ArtifactRequirement>,
        warmup_index: &BTreeMap<String, WarmupObligation>,
        artifact_root: &Path,
        cache_root: &Path,
        node_root: &Path,
        wave_root: &Path,
        warmup_root: &Path,
        completed_before_wave: &BTreeSet<String>,
    ) -> Result<Vec<NodeExecutionResult>, MaterializationError> {
        let mut pending = wave.node_names.clone();
        let mut completed = completed_before_wave.clone();
        let mut results = Vec::new();

        while !pending.is_empty() {
            let ready = pending
                .iter()
                .filter_map(|node_name| {
                    let node = node_index.get(node_name).expect("known node");
                    if node
                        .dependency_nodes
                        .iter()
                        .all(|dependency| completed.contains(dependency))
                    {
                        Some(node.clone())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            if ready.is_empty() {
                return Err(MaterializationError::MissingDependency {
                    node: pending[0].clone(),
                });
            }

            let batch_results = match scheduling_mode(wave.execution_contract.discipline) {
                MaterializationSchedulingMode::Sequential => ready
                    .into_iter()
                    .map(|node| {
                        self.execute_node(
                            outcome,
                            &node,
                            requirement_index,
                            warmup_index,
                            artifact_root,
                            cache_root,
                            node_root,
                            wave_root,
                            warmup_root,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                MaterializationSchedulingMode::Parallel => std::thread::scope(|scope_handle| {
                    let mut handles = Vec::new();
                    for node in ready {
                        let requirement_index = requirement_index;
                        let warmup_index = warmup_index;
                        let artifact_root = artifact_root.to_path_buf();
                        let cache_root = cache_root.to_path_buf();
                        let node_root = node_root.to_path_buf();
                        let wave_root = wave_root.to_path_buf();
                        let warmup_root = warmup_root.to_path_buf();
                        handles.push(scope_handle.spawn(move || {
                            MaterializationExecutor::default().execute_node(
                                outcome,
                                &node,
                                requirement_index,
                                warmup_index,
                                &artifact_root,
                                &cache_root,
                                &node_root,
                                &wave_root,
                                &warmup_root,
                            )
                        }));
                    }
                    handles
                        .into_iter()
                        .map(|handle| handle.join().expect("wave worker panicked"))
                        .collect::<Result<Vec<_>, _>>()
                })?,
            };

            let batch_names = batch_results
                .iter()
                .map(|result| result.record.node_name.clone())
                .collect::<Vec<_>>();
            pending
                .retain(|node_name| !batch_names.iter().any(|ready_name| ready_name == node_name));
            for node_name in batch_names {
                completed.insert(node_name);
            }
            results.extend(batch_results);
        }

        Ok(results)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_node(
        &self,
        outcome: &PlanningOutcome,
        node: &MaterializationNode,
        requirement_index: &BTreeMap<String, ArtifactRequirement>,
        warmup_index: &BTreeMap<String, WarmupObligation>,
        artifact_root: &Path,
        cache_root: &Path,
        node_root: &Path,
        wave_root: &Path,
        warmup_root: &Path,
    ) -> Result<NodeExecutionResult, MaterializationError> {
        match node.kind {
            MaterializationNodeKind::Compile | MaterializationNodeKind::Materialize => {
                let requirement = requirement_index.get(&node.name).ok_or_else(|| {
                    MaterializationError::UnknownArtifactNode {
                        node: node.name.clone(),
                    }
                })?;
                let mut artifact =
                    self.materialize_artifact(outcome, requirement, artifact_root, cache_root)?;
                artifact.rank_count = node.rank_scope.len() as u16;
                let outputs = vec![artifact.relative_path.clone()];
                let record = self.write_node_record(
                    node_root,
                    node,
                    artifact.disposition,
                    outputs,
                    artifact.compile_ms.saturating_add(artifact.transfer_ms),
                    artifact.bytes_written,
                )?;
                Ok(NodeExecutionResult {
                    record,
                    compile_ms: artifact.compile_ms,
                    transfer_ms: artifact.transfer_ms,
                    artifact: Some(artifact),
                })
            }
            MaterializationNodeKind::Transfer => {
                let started = Instant::now();
                let outputs = vec![self.write_receipt(
                    wave_root,
                    "materialization/waves",
                    &node.name,
                    &FanoutReceipt {
                        schema_version: SchemaVersion::current(),
                        plan_identity: outcome.plan.structural_identity.plan_identity.clone(),
                        node_name: node.name.clone(),
                        dependency_nodes: node.dependency_nodes.clone(),
                        ranks: node.rank_scope.clone(),
                    },
                )?];
                let duration_ms = elapsed_ms(started.elapsed());
                let bytes_written =
                    file_size(wave_root.join(format!("{}.json", safe_name(&node.name))))?;
                let record = self.write_node_record(
                    node_root,
                    node,
                    MaterializationDisposition::Executed,
                    outputs,
                    duration_ms,
                    bytes_written,
                )?;
                Ok(NodeExecutionResult {
                    record,
                    compile_ms: 0,
                    transfer_ms: duration_ms,
                    artifact: None,
                })
            }
            MaterializationNodeKind::Warmup => {
                let obligation = warmup_index.get(&node.name).expect("warmup node exists");
                let started = Instant::now();
                let outputs = vec![self.write_receipt(
                    warmup_root,
                    "warmup",
                    &node.name,
                    &WarmupReceipt {
                        schema_version: SchemaVersion::current(),
                        plan_identity: outcome.plan.structural_identity.plan_identity.clone(),
                        node_name: node.name.clone(),
                        dependency_nodes: node.dependency_nodes.clone(),
                        required_artifacts: obligation.required_artifacts.clone(),
                        proof: obligation.proof.clone(),
                    },
                )?];
                let duration_ms = elapsed_ms(started.elapsed());
                let bytes_written =
                    file_size(warmup_root.join(format!("{}.json", safe_name(&node.name))))?;
                let record = self.write_node_record(
                    node_root,
                    node,
                    MaterializationDisposition::Executed,
                    outputs,
                    duration_ms,
                    bytes_written,
                )?;
                Ok(NodeExecutionResult {
                    record,
                    compile_ms: 0,
                    transfer_ms: 0,
                    artifact: None,
                })
            }
            MaterializationNodeKind::Assemble | MaterializationNodeKind::Verify => {
                let started = Instant::now();
                let record = self.write_node_record(
                    node_root,
                    node,
                    MaterializationDisposition::Executed,
                    Vec::new(),
                    elapsed_ms(started.elapsed()),
                    0,
                )?;
                Ok(NodeExecutionResult {
                    record,
                    compile_ms: 0,
                    transfer_ms: 0,
                    artifact: None,
                })
            }
        }
    }

    fn write_node_record(
        &self,
        node_root: &Path,
        node: &sock_core::MaterializationNode,
        disposition: MaterializationDisposition,
        outputs: Vec<String>,
        duration_ms: u64,
        bytes_written: u64,
    ) -> Result<MaterializationNodeRecord, MaterializationError> {
        let record = MaterializationNodeRecord {
            node_name: node.name.clone(),
            wave: node.wave,
            kind: node.kind,
            queue: node.queue,
            disposition,
            dependency_nodes: node.dependency_nodes.clone(),
            outputs,
            relative_path: format!("materialization/nodes/{}.json", safe_name(&node.name)),
            duration_ms,
            bytes_written,
        };
        fs::write(
            node_root.join(format!("{}.json", safe_name(&node.name))),
            canonical_json(&record)?.as_bytes(),
        )?;
        Ok(record)
    }

    fn write_receipt<T: Serialize>(
        &self,
        root: &Path,
        relative_dir: &str,
        name: &str,
        payload: &T,
    ) -> Result<String, MaterializationError> {
        let relative = format!("{relative_dir}/{}.json", safe_name(name));
        fs::write(
            root.join(format!("{}.json", safe_name(name))),
            canonical_json(payload)?.as_bytes(),
        )?;
        Ok(relative)
    }
}

fn closure_expansion(scope: &BuildScope, outcome: &PlanningOutcome) -> ClosureExpansionRecord {
    let mut requested_backend_families = scope
        .backend_families
        .iter()
        .map(|family: &sock_core::BackendFamily| family.as_str().to_owned())
        .collect::<Vec<_>>();
    requested_backend_families.sort();
    let mut expanded_regions = outcome
        .plan
        .compile_regions
        .iter()
        .map(|region| region.name.clone())
        .collect::<Vec<_>>();
    expanded_regions.sort();
    let mut expanded_artifact_scopes = outcome
        .plan
        .artifact_requirements
        .iter()
        .map(|requirement| requirement.scope.clone())
        .collect::<Vec<_>>();
    expanded_artifact_scopes.sort();
    expanded_artifact_scopes.dedup();
    let mut expanded_warmup_scopes = outcome
        .plan
        .warmup_obligations
        .iter()
        .map(|obligation| obligation.region_name.clone())
        .collect::<Vec<_>>();
    expanded_warmup_scopes.sort();
    expanded_warmup_scopes.dedup();

    ClosureExpansionRecord {
        requested_regions: scope.region_names.iter().cloned().collect(),
        requested_artifact_scopes: scope.artifact_scopes.iter().cloned().collect(),
        requested_backend_families,
        requested_cache_namespaces: scope.cache_namespaces.iter().cloned().collect(),
        requested_warmup_scopes: scope.warmup_scopes.iter().cloned().collect(),
        expanded_regions,
        expanded_artifact_scopes,
        expanded_warmup_scopes,
        deterministically_closed: true,
    }
}

fn observed_readiness(
    scope: &BuildScope,
    outcome: &PlanningOutcome,
    completed_nodes: &BTreeSet<String>,
) -> ReadinessObservation {
    let requested_readiness = requested_readiness(scope);
    let blocking_nodes = outcome
        .plan
        .materialization_graph
        .nodes
        .iter()
        .filter(|node| {
            node.execution_contract.serve_phase == sock_core::ServePhase::PreServeBlocking
        })
        .map(|node| node.name.clone())
        .collect::<Vec<_>>();
    let early_serve_nodes = outcome
        .plan
        .materialization_graph
        .nodes
        .iter()
        .filter(|node| {
            node.execution_contract.serve_phase == sock_core::ServePhase::EarlyServeReady
        })
        .map(|node| node.name.clone())
        .collect::<Vec<_>>();
    let deferred_nodes = outcome
        .plan
        .materialization_graph
        .nodes
        .iter()
        .filter(|node| {
            node.execution_contract.serve_phase == sock_core::ServePhase::DeferredPerformance
        })
        .map(|node| node.name.clone())
        .collect::<Vec<_>>();

    let blocking_warmups_complete = blocking_nodes
        .iter()
        .all(|node_name| completed_nodes.contains(node_name));
    let early_serve_frontier_complete = early_serve_nodes
        .iter()
        .all(|node_name| completed_nodes.contains(node_name));
    let deferred_warmups_complete = deferred_nodes
        .iter()
        .all(|node_name| completed_nodes.contains(node_name));
    let has_blocking_nodes = !blocking_nodes.is_empty();
    let has_deferred_nodes = !deferred_nodes.is_empty();

    let achieved_readiness = if has_deferred_nodes && deferred_warmups_complete {
        ObservedReadinessLevel::Performance
    } else if has_blocking_nodes && blocking_warmups_complete {
        ObservedReadinessLevel::Correctness
    } else {
        ObservedReadinessLevel::EarlyServe
    };

    ReadinessObservation {
        requested_readiness,
        achieved_readiness,
        blocking_warmups_complete,
        early_serve_frontier_complete,
        deferred_warmups_complete,
    }
}

fn requested_readiness(scope: &BuildScope) -> ObservedReadinessLevel {
    match scope.readiness {
        Some(BuildReadiness::EarlyServe) => ObservedReadinessLevel::EarlyServe,
        Some(BuildReadiness::Correctness) => ObservedReadinessLevel::Correctness,
        Some(BuildReadiness::Performance) | None => ObservedReadinessLevel::Performance,
    }
}

fn observed_runtime_jit(
    outcome: &PlanningOutcome,
    artifact_records: &[MaterializedArtifactRecord],
    completed_nodes: &BTreeSet<String>,
    warmup_index: &BTreeMap<String, WarmupObligation>,
) -> Vec<RuntimeJitObservation> {
    let observed_artifact_scopes = artifact_records
        .iter()
        .map(|artifact| artifact.scope.clone())
        .collect::<BTreeSet<_>>();
    let observed_warmup_proofs = warmup_index
        .iter()
        .filter(|(node_name, _)| completed_nodes.contains(*node_name))
        .map(|(_, obligation)| obligation.proof.proof_id.clone())
        .collect::<BTreeSet<_>>();
    let observed_warmup_scopes = warmup_index
        .iter()
        .filter(|(node_name, _)| completed_nodes.contains(*node_name))
        .map(|(_, obligation)| obligation.region_name.clone())
        .collect::<BTreeSet<_>>();

    outcome
        .verification
        .runtime_jit_evidence
        .iter()
        .map(|evidence| {
            let observed_artifacts = evidence
                .required_artifacts
                .iter()
                .filter(|scope| observed_artifact_scopes.contains(*scope))
                .cloned()
                .collect::<Vec<_>>();
            let observed_warmup = evidence
                .required_warmup_proofs
                .iter()
                .filter(|proof| observed_warmup_proofs.contains(*proof))
                .cloned()
                .collect::<Vec<_>>();

            let mut contradiction_reasons = evidence.contradiction_reasons.clone();
            contradiction_reasons.extend(
                evidence
                    .required_artifacts
                    .iter()
                    .filter(|scope| !observed_artifact_scopes.contains(*scope))
                    .map(|scope| {
                        format!(
                            "required artifact scope {scope} was not observed in the built closure"
                        )
                    }),
            );
            contradiction_reasons.extend(
                evidence
                    .declared_required_warmup_scopes
                    .iter()
                    .filter(|scope| !observed_warmup_scopes.contains(*scope))
                    .map(|scope| {
                        format!(
                            "required warmup scope {scope} was not observed during materialization"
                        )
                    }),
            );
            contradiction_reasons.extend(
                evidence
                    .required_warmup_proofs
                    .iter()
                    .filter(|proof| !observed_warmup_proofs.contains(*proof))
                    .map(|proof| {
                        format!(
                            "required warmup proof {proof} was not observed during materialization"
                        )
                    }),
            );
            contradiction_reasons.sort();
            contradiction_reasons.dedup();

            RuntimeJitObservation {
                surface_name: evidence.surface_name.clone(),
                status: if contradiction_reasons.is_empty() {
                    RuntimeJitObservationStatus::Bounded
                } else {
                    RuntimeJitObservationStatus::Contradicted
                },
                observed_artifacts,
                observed_warmup_proofs: observed_warmup,
                contradiction_reasons,
            }
        })
        .collect()
}

fn verify_replay_compile_free(outcome: &PlanningOutcome) -> bool {
    let verify_gate = outcome
        .verification
        .operator_gates
        .iter()
        .find(|gate| gate.command == "verify")
        .map(|gate| gate.compile_free)
        .unwrap_or(false);
    let replay_gate = outcome
        .verification
        .operator_gates
        .iter()
        .find(|gate| gate.command == "replay")
        .map(|gate| gate.compile_free)
        .unwrap_or(false);
    verify_gate && replay_gate
}

fn load_existing_document(path: &Path) -> Result<ArtifactDeserialization, MaterializationError> {
    let started = Instant::now();
    let document = serde_json::from_str(&fs::read_to_string(path)?)?;
    Ok(ArtifactDeserialization {
        document,
        duration_ms: elapsed_ms(started.elapsed()),
    })
}

fn duplicate_instances_for_artifact(artifact: &MaterializedArtifactRecord) -> u32 {
    u32::from(artifact.rank_count.saturating_sub(1))
}

fn duplicate_bytes_for_artifact(artifact: &MaterializedArtifactRecord) -> u64 {
    artifact
        .bytes_written
        .saturating_mul(u64::from(artifact.rank_count.saturating_sub(1)))
}

fn artifact_deserialization_ms_for(artifact: &MaterializedArtifactRecord) -> u64 {
    artifact.deserialization_ms
}

fn closure_outcome(
    closure_expansion: &ClosureExpansionRecord,
    runtime_jit_observations: &[RuntimeJitObservation],
    verify_replay_compile_free: bool,
) -> StartupClosureOutcome {
    let contradiction_count = runtime_jit_observations
        .iter()
        .filter(|observation| observation.status == RuntimeJitObservationStatus::Contradicted)
        .count();
    if contradiction_count > 0 {
        StartupClosureOutcome::PartialCompileClosure
    } else if closure_expansion.deterministically_closed && verify_replay_compile_free {
        StartupClosureOutcome::FullCompileClosure
    } else if closure_expansion.deterministically_closed {
        StartupClosureOutcome::PartialCompileClosure
    } else {
        StartupClosureOutcome::ClosureByAssumption
    }
}

fn write_bytes_atomically(path: &Path, bytes: &[u8]) -> Result<(), MaterializationError> {
    let temp_path = sibling_temp_path(path);
    fs::write(&temp_path, bytes)?;
    fs::rename(&temp_path, path)?;
    Ok(())
}

fn copy_atomically(source: &Path, destination: &Path) -> Result<(), MaterializationError> {
    let temp_path = sibling_temp_path(destination);
    fs::copy(source, &temp_path)?;
    fs::rename(&temp_path, destination)?;
    Ok(())
}

fn sibling_temp_path(path: &Path) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("artifact");
    path.with_file_name(format!("{file_name}.{nanos}.tmp"))
}

fn artifact_handle(requirement: &ArtifactRequirement) -> Result<String, CanonicalError> {
    Ok(format!(
        "artifact:{}:{}",
        requirement.class.as_str(),
        canonical_hash(requirement)?
    ))
}

fn source_anchors_for_scope(outcome: &PlanningOutcome, scope: &str) -> Vec<SourceAnchor> {
    let mut anchors = outcome
        .plan
        .compile_regions
        .iter()
        .filter(|region| region.name == scope)
        .flat_map(|region| region.evidence.anchors.clone())
        .collect::<Vec<_>>();
    anchors.sort();
    anchors.dedup();
    anchors
}

fn safe_name(value: &str) -> String {
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

fn slug(value: &str) -> String {
    safe_name(value)
}

fn scheduling_mode(discipline: QueueDiscipline) -> MaterializationSchedulingMode {
    match discipline {
        QueueDiscipline::ParallelPerRank => MaterializationSchedulingMode::Parallel,
        QueueDiscipline::Serial | QueueDiscipline::LeaderThenBroadcast => {
            MaterializationSchedulingMode::Sequential
        }
    }
}

fn wave_max_parallelism(wave: &MaterializationWave) -> u16 {
    match wave.execution_contract.discipline {
        QueueDiscipline::ParallelPerRank => wave.node_names.len().min(u16::MAX as usize) as u16,
        QueueDiscipline::Serial | QueueDiscipline::LeaderThenBroadcast => 1,
    }
}

fn observed_elapsed_ms(duration: Duration) -> u64 {
    elapsed_ms(duration).max(1)
}

fn same_artifact_document_identity(
    left: &MaterializedArtifactDocument,
    right: &MaterializedArtifactDocument,
) -> bool {
    left.schema_version == right.schema_version
        && left.storage_key == right.storage_key
        && left.manifest_identity == right.manifest_identity
        && left.scope == right.scope
        && left.class == right.class
        && left.backend == right.backend
        && left.acquisition == right.acquisition
        && left.engine_root == right.engine_root
        && left.engine_revision == right.engine_revision
        && left.source_anchors == right.source_anchors
        && left.admissibility_summary == right.admissibility_summary
}

fn cache_namespace_for_scope(outcome: &PlanningOutcome, scope: &str) -> String {
    outcome
        .adapter_survey
        .compile_regions
        .iter()
        .find(|region| region.canonical_name == scope)
        .map(|region| region.cache_namespace.clone())
        .or_else(|| {
            outcome
                .adapter_survey
                .cache_ownership_surfaces
                .iter()
                .find(|surface| {
                    surface
                        .artifact_scopes
                        .iter()
                        .any(|candidate| candidate == scope)
                })
                .map(|surface| surface.name.clone())
        })
        .unwrap_or_else(|| "compile-cache".to_owned())
}

fn invalidation_domain_for_scope(outcome: &PlanningOutcome, scope: &str) -> String {
    outcome
        .plan
        .compile_regions
        .iter()
        .find(|region| region.name == scope)
        .map(|region| region.invalidation_domain.clone())
        .unwrap_or_else(|| scope.to_owned())
}

fn observed_preferred_strategy(
    requirement: &ArtifactRequirement,
    rebuild_ms: u64,
) -> FanoutStrategy {
    match requirement.rank_disposition {
        RankDisposition::RankLocal => FanoutStrategy::RebuildPerRank,
        RankDisposition::Shared => {
            if requirement.expected_transfer_ms.unwrap_or(0) < rebuild_ms {
                FanoutStrategy::BroadcastFromLeader
            } else {
                FanoutStrategy::RebuildPerRank
            }
        }
    }
}

fn evict_invalidated_siblings(
    cache_root: &Path,
    cache_namespace: &str,
    invalidation_domain: &str,
    class: ArtifactClass,
    keep_storage_key: &str,
) -> Result<(), MaterializationError> {
    let namespace_root = cache_root.join(slug(cache_namespace));
    if !namespace_root.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(namespace_root)? {
        let entry = entry?;
        let entry_path = entry.path();
        if !entry_path.is_dir() || entry.file_name() == keep_storage_key {
            continue;
        }
        let artifact_path = entry_path.join("artifact.json");
        if !artifact_path.exists() {
            continue;
        }
        let existing: MaterializedArtifactDocument =
            serde_json::from_str(&fs::read_to_string(&artifact_path)?)?;
        if existing.cache_namespace == cache_namespace
            && existing.invalidation_domain == invalidation_domain
            && existing.class == class
        {
            fs::remove_dir_all(entry_path)?;
        }
    }
    Ok(())
}

fn elapsed_ms(duration: Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}

fn file_size(path: impl AsRef<Path>) -> Result<u64, std::io::Error> {
    Ok(fs::metadata(path.as_ref())?.len())
}
