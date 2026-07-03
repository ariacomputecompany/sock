use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sock_core::{
    ArtifactAcquisition, ArtifactRequirement, CanonicalError, MaterializationDisposition,
    MaterializationExecutionReport, MaterializationNodeKind, MaterializationNodeRecord,
    MaterializationWaveRecord, MaterializedArtifactRecord, SchemaVersion, SourceAnchor,
    canonical_hash, canonical_json,
};
use thiserror::Error;

use crate::{PlanningOutcome, vllm};

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

impl MaterializationExecutor {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    pub fn execute(
        &self,
        outcome: &PlanningOutcome,
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
            .map(|requirement| Ok((artifact_handle(requirement)?, requirement)))
            .collect::<Result<BTreeMap<_, _>, MaterializationError>>()?;
        let warmup_index = outcome
            .plan
            .warmup_obligations
            .iter()
            .map(|obligation| {
                (
                    format!("warmup:{}:{}", obligation.node_name, obligation.region_name),
                    obligation,
                )
            })
            .collect::<BTreeMap<_, _>>();

        let mut completed_nodes = BTreeSet::new();
        let mut artifacts = BTreeMap::new();
        let mut node_records = Vec::new();
        let mut wave_records = Vec::new();
        let mut total_bytes_written = 0_u64;
        let mut total_compile_ms = 0_u64;
        let mut total_transfer_ms = 0_u64;

        for wave in &outcome.plan.materialization_graph.waves {
            let wave_started = Instant::now();
            let mut wave_bytes = 0_u64;
            for node_name in &wave.node_names {
                let node = outcome
                    .plan
                    .materialization_graph
                    .nodes
                    .iter()
                    .find(|candidate| &candidate.name == node_name)
                    .expect("wave references known node");
                if node
                    .dependency_nodes
                    .iter()
                    .any(|dependency| !completed_nodes.contains(dependency))
                {
                    return Err(MaterializationError::MissingDependency {
                        node: node.name.clone(),
                    });
                }

                let record = match node.kind {
                    MaterializationNodeKind::Compile | MaterializationNodeKind::Materialize => {
                        let requirement = requirement_index.get(&node.name).ok_or_else(|| {
                            MaterializationError::UnknownArtifactNode {
                                node: node.name.clone(),
                            }
                        })?;
                        let artifact =
                            self.materialize_artifact(outcome, requirement, &artifact_root)?;
                        total_compile_ms = total_compile_ms.saturating_add(artifact.compile_ms);
                        total_transfer_ms = total_transfer_ms.saturating_add(artifact.transfer_ms);
                        wave_bytes = wave_bytes.saturating_add(artifact.bytes_written);
                        let outputs = vec![artifact.relative_path.clone()];
                        let disposition = artifact.disposition;
                        let duration_ms = artifact.compile_ms;
                        let bytes_written = artifact.bytes_written;
                        artifacts.insert(artifact.storage_key.clone(), artifact);
                        self.write_node_record(
                            &node_root,
                            node,
                            disposition,
                            outputs,
                            duration_ms,
                            bytes_written,
                        )?
                    }
                    MaterializationNodeKind::Transfer => {
                        let outputs = vec![self.write_receipt(
                            &wave_root,
                            "materialization/waves",
                            &node.name,
                            &FanoutReceipt {
                                schema_version: SchemaVersion::current(),
                                plan_identity:
                                    outcome.plan.structural_identity.plan_identity.clone(),
                                node_name: node.name.clone(),
                                dependency_nodes: node.dependency_nodes.clone(),
                                ranks: node.rank_scope.clone(),
                            },
                        )?];
                        let bytes_written = file_size(out_dir.join(&outputs[0]))?;
                        wave_bytes = wave_bytes.saturating_add(bytes_written);
                        total_transfer_ms =
                            total_transfer_ms.saturating_add(elapsed_ms(wave_started.elapsed()));
                        self.write_node_record(
                            &node_root,
                            node,
                            MaterializationDisposition::Executed,
                            outputs,
                            elapsed_ms(wave_started.elapsed()),
                            bytes_written,
                        )?
                    }
                    MaterializationNodeKind::Warmup => {
                        let obligation = warmup_index.get(&node.name).expect("warmup node exists");
                        let outputs = vec![self.write_receipt(
                            &warmup_root,
                            "warmup",
                            &node.name,
                            &WarmupReceipt {
                                schema_version: SchemaVersion::current(),
                                plan_identity:
                                    outcome.plan.structural_identity.plan_identity.clone(),
                                node_name: node.name.clone(),
                                dependency_nodes: node.dependency_nodes.clone(),
                                required_artifacts: obligation.required_artifacts.clone(),
                                proof: obligation.proof.clone(),
                            },
                        )?];
                        let bytes_written = file_size(out_dir.join(&outputs[0]))?;
                        wave_bytes = wave_bytes.saturating_add(bytes_written);
                        self.write_node_record(
                            &node_root,
                            node,
                            MaterializationDisposition::Executed,
                            outputs,
                            elapsed_ms(wave_started.elapsed()),
                            bytes_written,
                        )?
                    }
                    MaterializationNodeKind::Assemble | MaterializationNodeKind::Verify => self
                        .write_node_record(
                            &node_root,
                            node,
                            MaterializationDisposition::Executed,
                            Vec::new(),
                            0,
                            0,
                        )?,
                };
                total_bytes_written = total_bytes_written.saturating_add(record.bytes_written);
                completed_nodes.insert(node.name.clone());
                node_records.push(record);
            }

            let wave_record = MaterializationWaveRecord {
                wave_name: wave.name.clone(),
                queue: wave.queue,
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
        };
        let content = canonical_json(&document)?;
        let (disposition, bytes_written) = if absolute_path.exists() {
            let existing: MaterializedArtifactDocument =
                serde_json::from_str(&fs::read_to_string(&absolute_path)?)?;
            if existing == document {
                (MaterializationDisposition::Reused, 0)
            } else {
                fs::write(&absolute_path, content.as_bytes())?;
                (MaterializationDisposition::Executed, content.len() as u64)
            }
        } else {
            fs::write(&absolute_path, content.as_bytes())?;
            (MaterializationDisposition::Executed, content.len() as u64)
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
            compile_ms: elapsed_ms(started.elapsed()),
            transfer_ms: 0,
            source_anchors: source_anchors_for_scope(outcome, &requirement.scope),
        })
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

fn elapsed_ms(duration: Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}

fn file_size(path: impl AsRef<Path>) -> Result<u64, std::io::Error> {
    Ok(fs::metadata(path.as_ref())?.len())
}
