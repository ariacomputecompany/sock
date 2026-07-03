use serde::{Deserialize, Serialize};

use crate::SchemaVersion;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasurementPhaseTimings {
    pub configure_ms: u64,
    pub compile_ms: u64,
    pub link_assemble_ms: u64,
    pub packaging_ms: u64,
    pub warmup_materialization_ms: u64,
    pub verification_ms: u64,
}

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
    pub replay_plan_identity: String,
    pub artifact_count: u32,
    pub executed_artifact_count: u32,
    pub reused_artifact_count: u32,
    pub unique_artifact_count: u32,
    pub duplicate_artifact_count: u32,
    pub wall_clock_ms: u64,
    pub total_compile_ms: u64,
    pub total_transfer_ms: u64,
    pub total_rebuild_ms: u64,
    pub total_bytes_written: u64,
    pub unique_artifact_bytes: u64,
    pub duplicate_artifact_bytes: u64,
    pub artifact_deserialization_ms: u64,
    pub duplicate_rank_local_compile_count: u32,
    pub duplicate_rank_local_load_count: u32,
    pub runtime_jit_contradiction_count: u32,
    pub closure_outcome: String,
    pub phase_timings: MeasurementPhaseTimings,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasurementComparisonReport {
    pub baseline_label: String,
    pub candidate_label: String,
    pub baseline_plan_identity: String,
    pub candidate_plan_identity: String,
    pub wall_clock_ms_delta: i64,
    pub wall_clock_reduction_bps: i64,
    pub executed_artifact_delta: i64,
    pub executed_artifact_reduction_bps: i64,
    pub bytes_written_delta: i64,
    pub bytes_written_reduction_bps: i64,
    pub reused_artifact_delta: i64,
    pub configure_ms_delta: i64,
    pub compile_ms_delta: i64,
    pub link_assemble_ms_delta: i64,
    pub packaging_ms_delta: i64,
    pub warmup_materialization_ms_delta: i64,
    pub verification_ms_delta: i64,
    pub changed_phases: Vec<String>,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkTraceReference {
    pub scenario: String,
    pub trace_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkCaseArtifactPaths {
    pub label: String,
    pub bundle_root: String,
    pub buildplan_path: String,
    pub artifact_manifest_path: String,
    pub materialization_report_path: String,
    pub measurement_report_path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkMatrixEntry {
    pub label: String,
    pub benchmark_class: String,
    pub baseline_description: String,
    pub candidate_description: String,
    pub selected_backend_only: bool,
    pub measurement: BuildMeasurementReport,
    pub artifact_paths: Vec<BenchmarkCaseArtifactPaths>,
    pub cold_artifact_count_delta: i64,
    pub cold_unique_artifact_bytes_delta: i64,
    pub cold_duplicate_load_savings_bytes: i64,
    pub warm_duplicate_load_savings_bytes: i64,
    pub warm_start_latency_ms: u64,
    pub warm_start_reduction_bps: i64,
    pub trace_references: Vec<BenchmarkTraceReference>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkMatrixReport {
    pub schema_version: SchemaVersion,
    pub benchmark_program_version: u32,
    pub verification_manifest_path: String,
    pub benchmark_trace_scenario: String,
    pub benchmark_trace_path: String,
    pub entries: Vec<BenchmarkMatrixEntry>,
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
            baseline_plan_identity: baseline.plan_identity.clone(),
            candidate_plan_identity: candidate.plan_identity.clone(),
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
            configure_ms_delta: delta(
                baseline.phase_timings.configure_ms,
                candidate.phase_timings.configure_ms,
            ),
            compile_ms_delta: delta(
                baseline.phase_timings.compile_ms,
                candidate.phase_timings.compile_ms,
            ),
            link_assemble_ms_delta: delta(
                baseline.phase_timings.link_assemble_ms,
                candidate.phase_timings.link_assemble_ms,
            ),
            packaging_ms_delta: delta(
                baseline.phase_timings.packaging_ms,
                candidate.phase_timings.packaging_ms,
            ),
            warmup_materialization_ms_delta: delta(
                baseline.phase_timings.warmup_materialization_ms,
                candidate.phase_timings.warmup_materialization_ms,
            ),
            verification_ms_delta: delta(
                baseline.phase_timings.verification_ms,
                candidate.phase_timings.verification_ms,
            ),
            changed_phases: changed_phases(baseline, candidate),
        }
    }
}

fn delta(baseline: u64, candidate: u64) -> i64 {
    baseline as i64 - candidate as i64
}

fn reduction_bps(baseline: u64, candidate: u64) -> i64 {
    if baseline == 0 {
        0
    } else {
        (((baseline as i128 - candidate as i128) * 10_000_i128) / baseline as i128) as i64
    }
}

fn changed_phases(
    baseline: &MeasurementCaseReport,
    candidate: &MeasurementCaseReport,
) -> Vec<String> {
    let mut phases = Vec::new();
    if baseline.phase_timings.configure_ms != candidate.phase_timings.configure_ms {
        phases.push("configure".to_owned());
    }
    if baseline.phase_timings.compile_ms != candidate.phase_timings.compile_ms {
        phases.push("compile".to_owned());
    }
    if baseline.phase_timings.link_assemble_ms != candidate.phase_timings.link_assemble_ms {
        phases.push("link_assemble".to_owned());
    }
    if baseline.phase_timings.packaging_ms != candidate.phase_timings.packaging_ms {
        phases.push("packaging".to_owned());
    }
    if baseline.phase_timings.warmup_materialization_ms
        != candidate.phase_timings.warmup_materialization_ms
    {
        phases.push("warmup_materialization".to_owned());
    }
    if baseline.phase_timings.verification_ms != candidate.phase_timings.verification_ms {
        phases.push("verification".to_owned());
    }
    phases
}

#[cfg(test)]
mod tests {
    use crate::{MeasurementCaseReport, MeasurementComparisonReport, MeasurementPhaseTimings};

    fn sample_case(
        label: &str,
        plan_identity: &str,
        phase_timings: MeasurementPhaseTimings,
    ) -> MeasurementCaseReport {
        MeasurementCaseReport {
            label: label.to_owned(),
            requested_regions: Vec::new(),
            requested_artifact_scopes: Vec::new(),
            requested_backend_families: Vec::new(),
            requested_topology_scopes: Vec::new(),
            requested_cache_namespaces: Vec::new(),
            requested_warmup_scopes: Vec::new(),
            requested_readiness: "default".to_owned(),
            plan_identity: plan_identity.to_owned(),
            replay_plan_identity: plan_identity.to_owned(),
            artifact_count: 1,
            executed_artifact_count: 1,
            reused_artifact_count: 0,
            unique_artifact_count: 1,
            duplicate_artifact_count: 0,
            wall_clock_ms: 10,
            total_compile_ms: phase_timings.compile_ms,
            total_transfer_ms: 0,
            total_rebuild_ms: phase_timings.compile_ms,
            total_bytes_written: 10,
            unique_artifact_bytes: 10,
            duplicate_artifact_bytes: 0,
            artifact_deserialization_ms: 0,
            duplicate_rank_local_compile_count: 0,
            duplicate_rank_local_load_count: 0,
            runtime_jit_contradiction_count: 0,
            closure_outcome: "full_compile_closure".to_owned(),
            phase_timings,
        }
    }

    #[test]
    fn comparison_report_lists_changed_phases() {
        let baseline = sample_case(
            "baseline",
            "plan-a",
            MeasurementPhaseTimings {
                configure_ms: 5,
                compile_ms: 7,
                link_assemble_ms: 11,
                packaging_ms: 13,
                warmup_materialization_ms: 17,
                verification_ms: 19,
            },
        );
        let candidate = sample_case(
            "candidate",
            "plan-b",
            MeasurementPhaseTimings {
                configure_ms: 5,
                compile_ms: 3,
                link_assemble_ms: 11,
                packaging_ms: 9,
                warmup_materialization_ms: 17,
                verification_ms: 23,
            },
        );
        let comparison =
            MeasurementComparisonReport::between("baseline", &baseline, "candidate", &candidate);
        assert_eq!(comparison.baseline_plan_identity, "plan-a");
        assert_eq!(comparison.candidate_plan_identity, "plan-b");
        assert_eq!(
            comparison.changed_phases,
            vec![
                "compile".to_owned(),
                "packaging".to_owned(),
                "verification".to_owned()
            ]
        );
    }
}
