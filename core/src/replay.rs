use serde::{Deserialize, Serialize};

use crate::{
    CanonicalError, CanonicalHash, FanoutStrategy, MaterializationExecutionReport, RankDisposition,
    ResolvedBuildPlan, SchemaVersion, ValidationStatus, canonical_hash,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactRealizationMode {
    RebuiltOnly,
    ReusedOnly,
    Mixed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheCompatibilityRecord {
    pub storage_key: String,
    pub manifest_identity: String,
    pub cache_namespace: String,
    pub scope: String,
    pub backend: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyReuseRecord {
    pub scope: String,
    pub rank_disposition: RankDisposition,
    pub rank_count: u16,
    pub preferred_fanout_strategy: FanoutStrategy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayProofDocument {
    pub schema_version: SchemaVersion,
    pub plan_identity: CanonicalHash,
    pub request_identity: CanonicalHash,
    pub compile_closure_verified: bool,
    pub warmup_closure_verified: bool,
    pub autotune_closure_verified: bool,
    pub cache_artifact_compatibility_verified: bool,
    pub topology_scoped_reuse_verified: bool,
    pub compile_closure_artifacts: Vec<String>,
    pub warmup_closure_proofs: Vec<String>,
    pub autotune_closure_proofs: Vec<String>,
    pub cache_compatibility: Vec<CacheCompatibilityRecord>,
    pub topology_reuse: Vec<TopologyReuseRecord>,
    pub result_artifact_identity: CanonicalHash,
    pub realization_identity: CanonicalHash,
    pub realization_mode: ArtifactRealizationMode,
    pub contradiction_contract: String,
}

impl ReplayProofDocument {
    pub fn from_plan_and_materialization(
        plan: &ResolvedBuildPlan,
        report: &MaterializationExecutionReport,
    ) -> Result<Self, CanonicalError> {
        let compile_closure_artifacts = plan
            .guarantee_evidence
            .artifact_manifest
            .iter()
            .map(|artifact| artifact.identity.clone())
            .collect::<Vec<_>>();
        let warmup_closure_proofs = plan
            .warmup_obligations
            .iter()
            .map(|obligation| obligation.proof.proof_id.clone())
            .collect::<Vec<_>>();
        let autotune_closure_proofs = plan
            .warmup_obligations
            .iter()
            .filter(|obligation| obligation.requires_autotune)
            .map(|obligation| obligation.proof.proof_id.clone())
            .collect::<Vec<_>>();
        let cache_compatibility = report
            .artifacts
            .iter()
            .map(|artifact| CacheCompatibilityRecord {
                storage_key: artifact.storage_key.clone(),
                manifest_identity: artifact.manifest_identity.clone(),
                cache_namespace: artifact.cache_namespace.clone(),
                scope: artifact.scope.clone(),
                backend: artifact.backend.as_str().to_owned(),
            })
            .collect::<Vec<_>>();
        let topology_reuse = report
            .artifacts
            .iter()
            .map(|artifact| TopologyReuseRecord {
                scope: artifact.scope.clone(),
                rank_disposition: artifact.rank_disposition,
                rank_count: artifact.rank_count,
                preferred_fanout_strategy: artifact.preferred_fanout_strategy,
            })
            .collect::<Vec<_>>();

        let result_artifact_identity = canonical_hash(
            &report
                .artifacts
                .iter()
                .map(|artifact| {
                    (
                        artifact.manifest_identity.clone(),
                        artifact.scope.clone(),
                        artifact.class.as_str().to_owned(),
                        artifact.backend.as_str().to_owned(),
                    )
                })
                .collect::<Vec<_>>(),
        )?;
        let realization_identity = canonical_hash(
            &report
                .artifacts
                .iter()
                .map(|artifact| {
                    (
                        artifact.storage_key.clone(),
                        artifact.manifest_identity.clone(),
                        artifact.disposition,
                        artifact.cache_namespace.clone(),
                        artifact.rank_disposition,
                    )
                })
                .collect::<Vec<_>>(),
        )?;

        let realization_mode = match (
            report.executed_artifact_count > 0,
            report.reused_artifact_count > 0,
        ) {
            (true, true) => ArtifactRealizationMode::Mixed,
            (true, false) => ArtifactRealizationMode::RebuiltOnly,
            (false, true) => ArtifactRealizationMode::ReusedOnly,
            (false, false) => ArtifactRealizationMode::RebuiltOnly,
        };
        let observed_warmup_nodes = report
            .nodes
            .iter()
            .map(|node| node.node_name.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        let warmup_closure_verified = plan.warmup_obligations.iter().all(|obligation| {
            observed_warmup_nodes.contains(
                format!("warmup:{}:{}", obligation.node_name, obligation.region_name).as_str(),
            )
        });
        let autotune_closure_verified = plan
            .warmup_obligations
            .iter()
            .filter(|obligation| obligation.requires_autotune)
            .all(|obligation| {
                observed_warmup_nodes.contains(
                    format!("warmup:{}:{}", obligation.node_name, obligation.region_name).as_str(),
                )
            });
        let cache_artifact_compatibility_verified = plan
            .guarantee_evidence
            .artifact_manifest
            .iter()
            .all(|expected| {
                report
                    .artifacts
                    .iter()
                    .any(|actual| actual.manifest_identity == expected.identity)
            });
        let total_ranks = plan.normalized_request.topology.tensor_parallelism
            * plan.normalized_request.topology.pipeline_parallelism
            * plan.normalized_request.topology.replicas;
        let topology_scoped_reuse_verified = report.artifacts.iter().all(|artifact| match artifact
            .rank_disposition
        {
            RankDisposition::Shared => artifact.rank_count == total_ranks,
            RankDisposition::RankLocal => artifact.rank_count >= 1,
        });

        Ok(Self {
            schema_version: SchemaVersion::current(),
            plan_identity: plan.structural_identity.plan_identity.clone(),
            request_identity: plan.normalized_request.identity.clone(),
            compile_closure_verified: report.closure_outcome
                == crate::StartupClosureOutcome::FullCompileClosure
                && report.verify_replay_status == ValidationStatus::Passed,
            warmup_closure_verified,
            autotune_closure_verified,
            cache_artifact_compatibility_verified,
            topology_scoped_reuse_verified,
            compile_closure_artifacts,
            warmup_closure_proofs,
            autotune_closure_proofs,
            cache_compatibility,
            topology_reuse,
            result_artifact_identity,
            realization_identity,
            realization_mode,
            contradiction_contract: "same_requested_plan_requires_same_result_artifact_identity"
                .to_owned(),
        })
    }
}

impl ArtifactRealizationMode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RebuiltOnly => "rebuilt_only",
            Self::ReusedOnly => "reused_only",
            Self::Mixed => "mixed",
        }
    }
}
