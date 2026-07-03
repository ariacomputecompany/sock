use serde::{Deserialize, Serialize};

use crate::SchemaVersion;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasurementCaseReport {
    pub label: String,
    pub requested_regions: Vec<String>,
    pub requested_artifact_scopes: Vec<String>,
    pub requested_backend_families: Vec<String>,
    pub requested_topology_scopes: Vec<String>,
    pub requested_cache_namespaces: Vec<String>,
    pub requested_warmup_scopes: Vec<String>,
    pub requested_readiness: String,
    pub plan_identity: String,
    pub artifact_count: u32,
    pub executed_artifact_count: u32,
    pub reused_artifact_count: u32,
    pub wall_clock_ms: u64,
    pub total_compile_ms: u64,
    pub total_transfer_ms: u64,
    pub total_rebuild_ms: u64,
    pub total_bytes_written: u64,
    pub runtime_jit_contradiction_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasurementComparisonReport {
    pub baseline_label: String,
    pub candidate_label: String,
    pub wall_clock_ms_delta: i64,
    pub wall_clock_reduction_bps: i64,
    pub executed_artifact_delta: i64,
    pub executed_artifact_reduction_bps: i64,
    pub bytes_written_delta: i64,
    pub bytes_written_reduction_bps: i64,
    pub reused_artifact_delta: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildMeasurementReport {
    pub schema_version: SchemaVersion,
    pub intent: String,
    pub broad_cold: MeasurementCaseReport,
    pub scoped_cold: MeasurementCaseReport,
    pub scoped_warm: MeasurementCaseReport,
    pub scoped_vs_broad: MeasurementComparisonReport,
    pub warm_vs_cold: MeasurementComparisonReport,
}

impl MeasurementComparisonReport {
    #[must_use]
    pub fn between(
        baseline_label: impl Into<String>,
        baseline: &MeasurementCaseReport,
        candidate_label: impl Into<String>,
        candidate: &MeasurementCaseReport,
    ) -> Self {
        Self {
            baseline_label: baseline_label.into(),
            candidate_label: candidate_label.into(),
            wall_clock_ms_delta: baseline.wall_clock_ms as i64 - candidate.wall_clock_ms as i64,
            wall_clock_reduction_bps: reduction_bps(
                baseline.wall_clock_ms,
                candidate.wall_clock_ms,
            ),
            executed_artifact_delta: baseline.executed_artifact_count as i64
                - candidate.executed_artifact_count as i64,
            executed_artifact_reduction_bps: reduction_bps(
                baseline.executed_artifact_count as u64,
                candidate.executed_artifact_count as u64,
            ),
            bytes_written_delta: baseline.total_bytes_written as i64
                - candidate.total_bytes_written as i64,
            bytes_written_reduction_bps: reduction_bps(
                baseline.total_bytes_written,
                candidate.total_bytes_written,
            ),
            reused_artifact_delta: candidate.reused_artifact_count as i64
                - baseline.reused_artifact_count as i64,
        }
    }
}

fn reduction_bps(baseline: u64, candidate: u64) -> i64 {
    if baseline == 0 {
        0
    } else {
        (((baseline as i128 - candidate as i128) * 10_000_i128) / baseline as i128) as i64
    }
}
