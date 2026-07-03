use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;
use tempfile::tempdir;

#[test]
fn plan_summary_is_stable() {
    Command::cargo_bin("sock")
        .expect("sock binary")
        .arg("plan")
        .assert()
        .success()
        .stdout(predicate::str::contains("engine vllm"))
        .stdout(predicate::str::contains(
            "model meta-llama/Llama-3.1-8B-Instruct@main",
        ))
        .stdout(predicate::str::contains("backend FlashInfer"));
}

#[test]
fn explain_includes_trace_and_diagnostics() {
    Command::cargo_bin("sock")
        .expect("sock binary")
        .arg("explain")
        .assert()
        .success()
        .stdout(predicate::str::contains("rewrite trace:"))
        .stdout(predicate::str::contains("diagnostics:"))
        .stdout(predicate::str::contains("verified_bundle"));
}

#[test]
fn build_verify_and_replay_bundle_round_trip() {
    let dir = tempdir().expect("tempdir");

    Command::cargo_bin("sock")
        .expect("sock binary")
        .args(["build", "--out"])
        .arg(dir.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("bundle="))
        .stdout(predicate::str::contains("replay_entrypoint=./replay.sh"));

    for file in [
        "artifact_manifest.json",
        "buildplan.json",
        "bundle_metadata.json",
        "diagnostics.json",
        "materialization_report.json",
        "replay.sh",
        "rewrite_trace.json",
        "verification_report.json",
    ] {
        assert!(dir.path().join(file).exists(), "missing {file}");
    }

    let materialization: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("materialization_report.json"))
            .expect("read materialization report"),
    )
    .expect("parse materialization report");
    assert!(
        materialization["artifact_count"]
            .as_u64()
            .expect("artifact count")
            > 0
    );

    Command::cargo_bin("sock")
        .expect("sock binary")
        .args(["verify", "--bundle"])
        .arg(dir.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("verification Passed"))
        .stdout(predicate::str::contains("runtime-jit evidence:"))
        .stdout(predicate::str::contains(
            "verify compile_free=true forbidden_queues=Compile,Assemble,ArtifactIo,Warmup",
        ));

    Command::cargo_bin("sock")
        .expect("sock binary")
        .args(["replay", "--bundle"])
        .arg(dir.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("plan "))
        .stdout(predicate::str::contains("verification Passed"))
        .stdout(predicate::str::contains("runtime-jit evidence:"))
        .stdout(predicate::str::contains(
            "replay compile_free=true forbidden_queues=Compile,Assemble,ArtifactIo,Warmup",
        ))
        .stdout(predicate::str::contains("[info] verified_bundle"));
}

#[test]
fn tampered_bundle_is_rejected() {
    let dir = tempdir().expect("tempdir");

    Command::cargo_bin("sock")
        .expect("sock binary")
        .args(["build", "--out"])
        .arg(dir.path())
        .assert()
        .success();

    std::fs::write(dir.path().join("diagnostics.json"), "{}").expect("tamper diagnostics");

    Command::cargo_bin("sock")
        .expect("sock binary")
        .args(["verify", "--bundle"])
        .arg(dir.path())
        .assert()
        .failure()
        .stderr(predicate::str::contains("digest mismatch"));
}

#[test]
fn repeated_build_reuses_materialized_artifacts() {
    let dir = tempdir().expect("tempdir");

    Command::cargo_bin("sock")
        .expect("sock binary")
        .args(["build", "--out"])
        .arg(dir.path())
        .assert()
        .success();

    Command::cargo_bin("sock")
        .expect("sock binary")
        .args(["build", "--out"])
        .arg(dir.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("reused="));

    let materialization: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("materialization_report.json"))
            .expect("read materialization report"),
    )
    .expect("parse materialization report");
    assert!(
        materialization["reused_artifact_count"]
            .as_u64()
            .expect("reused artifact count")
            > 0
    );
}

#[test]
fn scoped_prefill_build_emits_minimal_closure() {
    let dir = tempdir().expect("tempdir");

    Command::cargo_bin("sock")
        .expect("sock binary")
        .args([
            "build",
            "--out",
            dir.path().to_str().expect("utf8 path"),
            "--region",
            "prefill_attention",
            "--readiness",
            "correctness",
        ])
        .assert()
        .success();

    let plan: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("buildplan.json")).expect("read buildplan"),
    )
    .expect("parse buildplan");
    let compile_regions = plan["compile_regions"]
        .as_array()
        .expect("compile regions array");
    assert_eq!(compile_regions.len(), 1);
    assert_eq!(compile_regions[0]["name"], "prefill_attention");

    let artifact_manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("artifact_manifest.json"))
            .expect("read artifact manifest"),
    )
    .expect("parse artifact manifest");
    let artifacts = artifact_manifest["artifacts"]
        .as_array()
        .expect("artifacts array");
    assert!(
        artifacts
            .iter()
            .all(|artifact| artifact["scope"] == "prefill_attention")
    );

    let warmup_obligations = plan["warmup_obligations"]
        .as_array()
        .expect("warmup obligations array");
    assert!(
        warmup_obligations
            .iter()
            .all(|obligation| obligation["region_name"] == "prefill_attention")
    );
    assert!(
        warmup_obligations
            .iter()
            .all(|obligation| obligation["blocking"] == true)
    );

    let materialization: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("materialization_report.json"))
            .expect("read materialization report"),
    )
    .expect("parse materialization report");
    let materialized_artifacts = materialization["artifacts"]
        .as_array()
        .expect("materialized artifacts array");
    assert!(
        materialized_artifacts
            .iter()
            .all(|artifact| artifact["scope"] == "prefill_attention")
    );
    assert!(materialized_artifacts.iter().all(|artifact| {
        let relative_path = artifact["relative_path"].as_str().expect("relative path");
        dir.path().join(relative_path).exists()
    }));
}

#[test]
fn backend_family_scope_selects_decode_closure() {
    let dir = tempdir().expect("tempdir");

    Command::cargo_bin("sock")
        .expect("sock binary")
        .args([
            "build",
            "--out",
            dir.path().to_str().expect("utf8 path"),
            "--backend-family",
            "cuda-graphs",
            "--readiness",
            "performance",
        ])
        .assert()
        .success();

    let plan: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("buildplan.json")).expect("read buildplan"),
    )
    .expect("parse buildplan");
    let compile_regions = plan["compile_regions"]
        .as_array()
        .expect("compile regions array");
    assert_eq!(compile_regions.len(), 1);
    assert_eq!(compile_regions[0]["name"], "decode_attention");
}

#[test]
fn cache_namespace_scope_selects_flashinfer_kv_update_closure() {
    let dir = tempdir().expect("tempdir");

    Command::cargo_bin("sock")
        .expect("sock binary")
        .args([
            "build",
            "--out",
            dir.path().to_str().expect("utf8 path"),
            "--cache-namespace",
            "flashinfer-autotune-cache",
            "--readiness",
            "correctness",
        ])
        .assert()
        .success();

    let plan: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("buildplan.json")).expect("read buildplan"),
    )
    .expect("parse buildplan");
    let compile_regions = plan["compile_regions"]
        .as_array()
        .expect("compile regions array");
    assert_eq!(compile_regions.len(), 1);
    assert_eq!(compile_regions[0]["name"], "kv_cache_update");
}

#[test]
fn warmup_scope_selects_prefill_closure() {
    let dir = tempdir().expect("tempdir");

    Command::cargo_bin("sock")
        .expect("sock binary")
        .args([
            "build",
            "--out",
            dir.path().to_str().expect("utf8 path"),
            "--warmup-scope",
            "prefill_attention",
            "--readiness",
            "correctness",
        ])
        .assert()
        .success();

    let plan: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("buildplan.json")).expect("read buildplan"),
    )
    .expect("parse buildplan");
    let compile_regions = plan["compile_regions"]
        .as_array()
        .expect("compile regions array");
    assert_eq!(compile_regions.len(), 1);
    assert_eq!(compile_regions[0]["name"], "prefill_attention");
}
