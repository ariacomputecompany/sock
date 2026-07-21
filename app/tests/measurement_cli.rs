use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;
use tempfile::tempdir;

#[test]
fn measure_prefill_path_proves_scoped_and_warm_reuse_reduction() {
    let dir = tempdir().expect("tempdir");

    let mut cmd = Command::cargo_bin("sock").expect("sock binary");
    cmd.env("SOCK_TEST_HOST_PROFILE", "nvidia-sm90")
        .args([
            "measure",
            "prefill-path",
            "--out",
            dir.path().to_str().expect("utf8 path"),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("measurement intent=prefill_path"))
        .stdout(predicate::str::contains("scoped_wall_clock_reduction_bps="))
        .stdout(predicate::str::contains("warm_reused="));

    for file in [
        "measurement_report.json",
        "broad-cold/materialization_report.json",
        "scoped-cold/materialization_report.json",
        "scoped-warm/materialization_report.json",
    ] {
        assert!(dir.path().join(file).exists(), "missing {file}");
    }

    let report: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("measurement_report.json"))
            .expect("read measurement report"),
    )
    .expect("parse measurement report");

    assert_eq!(report["intent"], Value::String("prefill_path".to_owned()));
    assert!(
        report["scoped_vs_broad"]["executed_artifact_delta"]
            .as_i64()
            .expect("executed artifact delta")
            > 0
    );
    assert!(
        report["scoped_vs_broad"]["bytes_written_delta"]
            .as_i64()
            .expect("bytes delta")
            > 0
    );
    assert!(
        report["warm_vs_cold"]["reused_artifact_delta"]
            .as_i64()
            .expect("reused delta")
            > 0
    );
    assert!(
        report["scoped_warm"]["reused_artifact_count"]
            .as_u64()
            .expect("warm reused count")
            > 0
    );
}
