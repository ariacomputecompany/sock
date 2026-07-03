use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sock_core::{
    ArtifactAcquisition, ArtifactRequirement, CanonicalError, ClosureExpansionRecord,
    MaterializationDisposition, MaterializationExecutionReport, MaterializationNode,
    MaterializationNodeKind, MaterializationNodeRecord, MaterializationSchedulingMode,
    MaterializationWave, MaterializationWaveRecord, MaterializedArtifactRecord, QueueDiscipline,
    SchemaVersion, SourceAnchor, WarmupObligation, canonical_hash, canonical_json,
};
use thiserror::Error;

use crate::{BuildScope, PlanningOutcome, vllm};

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct MaterializedArtifactDocument {
    schema_version: SchemaVersion,
    storage_key: String,
    manifest_identity: String,
    scope: String,
    class: sock_core::ArtifactClass,
    backend: sock_core::BackendFamily,
    acquisition: ArtifactAcquisition,
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
        out_dir: &Path,
    ) -> Result<MaterializationExecutionReport, MaterializationError> {
        let artifact_root = out_dir.join("artifacts");
        let node_root = out_dir.join("materialization").join("nodes");
        let wave_root = out_dir.join("materialization").join("waves");
        let warmup_root = out_dir.join("warmup");
        fs::create_dir_all(&artifact_root)?;
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

        for wave in &outcome.plan.materialization_graph.waves {
            let wave_started = Instant::now();
            let wave_results = self.execute_wave(
                outcome,
                wave,
                &node_index,
                &requirement_index,
                &warmup_index,
                &artifact_root,
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

        let report = MaterializationExecutionReport {
            schema_version: SchemaVersion::current(),
            plan_identity: outcome.plan.structural_identity.plan_identity.clone(),
            artifact_root: "artifacts".to_owned(),
            node_root: "materialization/nodes".to_owned(),
            wave_root: "materialization/waves".to_owned(),
            artifact_count: artifact_records.len() as u32,
            reused_artifact_count: artifact_records
                .iter()
                .filter(|artifact| artifact.disposition == MaterializationDisposition::Reused)
                .count() as u32,
            total_bytes_written,
            total_compile_ms,
            total_transfer_ms,
            total_rebuild_ms,
            closure_expansion: closure_expansion(scope, outcome),
            artifacts: artifact_records,
            nodes: node_records,
            waves: wave_records,
        };
        fs::write(
            out_dir.join("materialization_report.json"),
            canonical_json(&report)?.as_bytes(),
        )?;
        Ok(report)
    }

    fn materialize_artifact(
        &self,
        outcome: &PlanningOutcome,
        requirement: &ArtifactRequirement,
        artifact_root: &Path,
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
        let relative_path = format!("artifacts/{storage_key}/artifact.json");
        let absolute_path = artifact_root.join(&storage_key).join("artifact.json");
        fs::create_dir_all(absolute_path.parent().expect("artifact parent"))?;
        let document = MaterializedArtifactDocument {
            schema_version: SchemaVersion::current(),
            storage_key: storage_key.clone(),
            manifest_identity: manifest_identity.clone(),
            scope: requirement.scope.clone(),
            class: requirement.class,
            backend: requirement.backend,
            acquisition: requirement.acquisition,
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
        let (disposition, bytes_written, rebuild_ms) = if absolute_path.exists() {
            let existing: MaterializedArtifactDocument =
                serde_json::from_str(&fs::read_to_string(&absolute_path)?)?;
            if same_artifact_document_identity(&existing, &document) {
                (
                    MaterializationDisposition::Reused,
                    0,
                    existing.observed_rebuild_ms.max(
                        existing
                            .observed_compile_ms
                            .saturating_add(existing.observed_transfer_ms),
                    ),
                )
            } else {
                let mut document = document.clone();
                document.observed_compile_ms = observed_compile_ms;
                document.observed_transfer_ms = observed_transfer_ms;
                document.observed_rebuild_ms = observed_rebuild_ms;
                let content = canonical_json(&document)?;
                fs::write(&absolute_path, content.as_bytes())?;
                (
                    MaterializationDisposition::Executed,
                    content.len() as u64,
                    observed_rebuild_ms,
                )
            }
        } else {
            let mut document = document.clone();
            document.observed_compile_ms = observed_compile_ms;
            document.observed_transfer_ms = observed_transfer_ms;
            document.observed_rebuild_ms = observed_rebuild_ms;
            let content = canonical_json(&document)?;
            fs::write(&absolute_path, content.as_bytes())?;
            (
                MaterializationDisposition::Executed,
                content.len() as u64,
                observed_rebuild_ms,
            )
        };
        Ok(MaterializedArtifactRecord {
            storage_key,
            manifest_identity,
            scope: requirement.scope.clone(),
            class: requirement.class,
            backend: requirement.backend,
            acquisition: requirement.acquisition,
            disposition,
            relative_path,
            bytes_written,
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
                let artifact = self.materialize_artifact(outcome, requirement, artifact_root)?;
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
                let record = self.write_node_record(
                    node_root,
                    node,
                    MaterializationDisposition::Executed,
                    Vec::new(),
                    0,
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

fn elapsed_ms(duration: Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}

fn file_size(path: impl AsRef<Path>) -> Result<u64, std::io::Error> {
    Ok(fs::metadata(path.as_ref())?.len())
}
