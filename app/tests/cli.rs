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
        "vllm_integration.json",
        "vllm_entrypoints.json",
    ] {
        assert!(dir.path().join(file).exists(), "missing {file}");
    }
    assert!(
        dir.path()
            .join("vllm-entrypoints")
            .join("invoke_vllm_surface.py")
            .exists()
    );

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
    assert_eq!(
        materialization["closure_expansion"]["deterministically_closed"],
        Value::Bool(true)
    );
    assert_eq!(
        materialization["verify_replay_compile_free"],
        Value::Bool(true)
    );
    assert!(
        materialization["waves"]
            .as_array()
            .expect("waves array")
            .iter()
            .all(|wave| wave.get("discipline").is_some())
    );
    assert!(
        materialization["waves"]
            .as_array()
            .expect("waves array")
            .iter()
            .all(|wave| wave.get("scheduling_mode").is_some())
    );
    assert_eq!(
        materialization["readiness"]["achieved_readiness"],
        Value::String("performance".to_owned())
    );
    assert!(
        materialization["runtime_jit_observations"]
            .as_array()
            .expect("runtime jit observations")
            .iter()
            .all(|observation| observation.get("status").is_some())
    );

    let integration: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("vllm_integration.json"))
            .expect("read vllm integration"),
    )
    .expect("parse vllm integration");
    assert!(
        integration["surfaces"]
            .as_array()
            .expect("integration surfaces")
            .iter()
            .any(|surface| surface["id"] == "compile-region:prefill_attention")
    );

    let entrypoints: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("vllm_entrypoints.json"))
            .expect("read vllm entrypoints"),
    )
    .expect("parse vllm entrypoints");
    assert!(
        entrypoints["entrypoints"]
            .as_array()
            .expect("entrypoints array")
            .iter()
            .any(|entrypoint| entrypoint["scope_name"] == "prefill_attention")
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
    assert!(
        materialization["total_rebuild_ms"]
            .as_u64()
            .expect("total rebuild ms")
            > 0
    );
    let reused_artifact = materialization["artifacts"]
        .as_array()
        .expect("artifacts array")
        .iter()
        .find(|artifact| artifact["disposition"] == "reused")
        .expect("reused artifact");
    assert_eq!(reused_artifact["compile_ms"], Value::from(0));
    assert_eq!(reused_artifact["transfer_ms"], Value::from(0));
    assert!(
        reused_artifact["rebuild_ms"]
            .as_u64()
            .expect("artifact rebuild ms")
            > 0
    );
}

#[test]
fn build_reports_split_cache_ownership_surfaces() {
    let dir = tempdir().expect("tempdir");

    Command::cargo_bin("sock")
        .expect("sock binary")
        .args(["build", "--out"])
        .arg(dir.path())
        .assert()
        .success();

    let materialization: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("materialization_report.json"))
            .expect("read materialization report"),
    )
    .expect("parse materialization report");
    let cache_namespaces = materialization["artifacts"]
        .as_array()
        .expect("artifacts array")
        .iter()
        .filter_map(|artifact| artifact["cache_namespace"].as_str())
        .collect::<std::collections::BTreeSet<_>>();
    assert!(cache_namespaces.contains("compile-cache"));
    assert!(cache_namespaces.contains("flashinfer-autotune-cache"));
    assert!(cache_namespaces.contains("cuda-graph-cache"));
    assert!(
        materialization["cache_root"]
            .as_str()
            .expect("cache root")
            .ends_with("/.sock-cache")
    );
}

#[test]
fn shared_cache_root_reuses_artifacts_across_bundle_roots() {
    let first_dir = tempdir().expect("tempdir");
    let second_dir = tempdir().expect("tempdir");
    let cache_dir = tempdir().expect("cache tempdir");

    Command::cargo_bin("sock")
        .expect("sock binary")
        .args([
            "build",
            "--out",
            first_dir.path().to_str().expect("utf8 path"),
            "--cache-root",
            cache_dir.path().to_str().expect("utf8 path"),
            "--region",
            "prefill_attention",
            "--readiness",
            "correctness",
        ])
        .assert()
        .success();

    Command::cargo_bin("sock")
        .expect("sock binary")
        .args([
            "build",
            "--out",
            second_dir.path().to_str().expect("utf8 path"),
            "--cache-root",
            cache_dir.path().to_str().expect("utf8 path"),
            "--region",
            "prefill_attention",
            "--readiness",
            "correctness",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("reused="));

    let materialization: Value = serde_json::from_str(
        &std::fs::read_to_string(second_dir.path().join("materialization_report.json"))
            .expect("read materialization report"),
    )
    .expect("parse materialization report");
    assert!(
        materialization["reused_artifact_count"]
            .as_u64()
            .expect("reused artifact count")
            > 0
    );
    for artifact in materialization["artifacts"]
        .as_array()
        .expect("artifacts array")
    {
        let relative_path = artifact["cache_relative_path"]
            .as_str()
            .expect("cache relative path");
        assert!(cache_dir.path().join(relative_path).exists());
    }
}

#[test]
fn invalidation_evicts_only_affected_cache_closure() {
    let bundle_dir = tempdir().expect("tempdir");
    let cache_dir = tempdir().expect("cache tempdir");

    Command::cargo_bin("sock")
        .expect("sock binary")
        .args([
            "build",
            "--out",
            bundle_dir.path().to_str().expect("utf8 path"),
            "--cache-root",
            cache_dir.path().to_str().expect("utf8 path"),
            "--region",
            "prefill_attention",
            "--readiness",
            "correctness",
        ])
        .assert()
        .success();

    let materialization: Value = serde_json::from_str(
        &std::fs::read_to_string(bundle_dir.path().join("materialization_report.json"))
            .expect("read materialization report"),
    )
    .expect("parse materialization report");
    let artifact = materialization["artifacts"]
        .as_array()
        .expect("artifacts array")
        .iter()
        .find(|artifact| artifact["scope"] == "prefill_attention")
        .expect("prefill artifact");
    let cache_relative_path = artifact["cache_relative_path"]
        .as_str()
        .expect("cache relative path");
    let cache_path = cache_dir.path().join(cache_relative_path);
    let cache_doc: Value =
        serde_json::from_str(&std::fs::read_to_string(&cache_path).expect("read cached artifact"))
            .expect("parse cached artifact");
    let namespace_dir = cache_path
        .parent()
        .and_then(|path| path.parent())
        .expect("namespace dir")
        .to_path_buf();

    let mut stale_doc = cache_doc.clone();
    stale_doc["storage_key"] = Value::String("stale-prefill-sibling".to_owned());
    stale_doc["manifest_identity"] = Value::String("stale-prefill-sibling".to_owned());
    let stale_dir = namespace_dir.join("stale-prefill-sibling");
    std::fs::create_dir_all(&stale_dir).expect("create stale dir");
    std::fs::write(
        stale_dir.join("artifact.json"),
        serde_json::to_vec(&stale_doc).expect("serialize stale doc"),
    )
    .expect("write stale doc");

    let mut unrelated_doc = cache_doc.clone();
    unrelated_doc["storage_key"] = Value::String("unrelated-sibling".to_owned());
    unrelated_doc["manifest_identity"] = Value::String("unrelated-sibling".to_owned());
    unrelated_doc["invalidation_domain"] = Value::String("other_domain".to_owned());
    let unrelated_dir = namespace_dir.join("unrelated-sibling");
    std::fs::create_dir_all(&unrelated_dir).expect("create unrelated dir");
    std::fs::write(
        unrelated_dir.join("artifact.json"),
        serde_json::to_vec(&unrelated_doc).expect("serialize unrelated doc"),
    )
    .expect("write unrelated doc");

    let mut corrupted_primary = cache_doc.clone();
    corrupted_primary["manifest_identity"] = Value::String("corrupted-primary".to_owned());
    std::fs::write(
        &cache_path,
        serde_json::to_vec(&corrupted_primary).expect("serialize corrupted doc"),
    )
    .expect("write corrupted primary");

    Command::cargo_bin("sock")
        .expect("sock binary")
        .args([
            "build",
            "--out",
            bundle_dir.path().to_str().expect("utf8 path"),
            "--cache-root",
            cache_dir.path().to_str().expect("utf8 path"),
            "--region",
            "prefill_attention",
            "--readiness",
            "correctness",
        ])
        .assert()
        .success();

    assert!(!stale_dir.exists(), "stale sibling should be evicted");
    assert!(
        unrelated_dir.exists(),
        "unrelated invalidation domain should remain"
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
    assert_eq!(
        materialization["closure_expansion"]["requested_regions"],
        Value::Array(vec![Value::String("prefill_attention".to_owned())])
    );
    assert_eq!(
        materialization["closure_expansion"]["expanded_regions"],
        Value::Array(vec![Value::String("prefill_attention".to_owned())])
    );
    assert_eq!(
        materialization["closure_expansion"]["expanded_warmup_scopes"],
        Value::Array(vec![Value::String("prefill_attention".to_owned())])
    );
    assert_eq!(
        materialization["closure_expansion"]["deterministically_closed"],
        Value::Bool(true)
    );
    assert_eq!(
        materialization["readiness"]["achieved_readiness"],
        Value::String("correctness".to_owned())
    );
    assert!(
        materialization["waves"]
            .as_array()
            .expect("waves array")
            .iter()
            .any(|wave| wave["scheduling_mode"] == "parallel")
    );

    let integration: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("vllm_integration.json"))
            .expect("read vllm integration"),
    )
    .expect("parse vllm integration");
    let surfaces = integration["surfaces"]
        .as_array()
        .expect("integration surfaces");
    assert!(
        surfaces
            .iter()
            .any(|surface| surface["id"] == "compile-region:prefill_attention")
    );
    let prefill_surface = surfaces
        .iter()
        .find(|surface| surface["id"] == "compile-region:prefill_attention")
        .expect("prefill integration surface");
    assert_eq!(
        prefill_surface["isolation"]["subset_build_valid"],
        Value::Bool(true)
    );
    assert!(
        !surfaces
            .iter()
            .any(|surface| surface["id"] == "compile-region:decode_attention")
    );

    let entrypoints: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("vllm_entrypoints.json"))
            .expect("read vllm entrypoints"),
    )
    .expect("parse vllm entrypoints");
    let build_entrypoints = entrypoints["entrypoints"]
        .as_array()
        .expect("entrypoints array");
    assert!(
        build_entrypoints
            .iter()
            .any(|entrypoint| entrypoint["scope_name"] == "prefill_attention")
    );
    assert!(
        !build_entrypoints
            .iter()
            .any(|entrypoint| entrypoint["scope_name"] == "decode_attention")
    );
    let wrapper_path = build_entrypoints
        .iter()
        .find(|entrypoint| entrypoint["scope_name"] == "prefill_attention")
        .and_then(|entrypoint| entrypoint["wrapper_path"].as_str())
        .expect("prefill wrapper path");
    assert!(dir.path().join(wrapper_path).exists());
}

#[test]
fn early_serve_build_skips_warmup_and_records_runtime_jit_contradictions() {
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
            "early-serve",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("readiness=EarlyServe"));

    let plan: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("buildplan.json")).expect("read buildplan"),
    )
    .expect("parse buildplan");
    assert_eq!(
        plan["warmup_obligations"]
            .as_array()
            .expect("warmup obligations array")
            .len(),
        0
    );

    let materialization: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("materialization_report.json"))
            .expect("read materialization report"),
    )
    .expect("parse materialization report");
    assert_eq!(
        materialization["readiness"]["requested_readiness"],
        Value::String("early_serve".to_owned())
    );
    assert_eq!(
        materialization["readiness"]["achieved_readiness"],
        Value::String("early_serve".to_owned())
    );
    assert_eq!(
        materialization["readiness"]["blocking_warmups_complete"],
        Value::Bool(true)
    );
    assert_eq!(
        materialization["readiness"]["deferred_warmups_complete"],
        Value::Bool(true)
    );
    assert!(
        materialization["runtime_jit_observations"]
            .as_array()
            .expect("runtime jit observations")
            .iter()
            .any(|observation| observation["status"] == "contradicted")
    );
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
        .failure()
        .stderr(predicate::str::contains(
            "scoped subset build is not semantically valid for compile-region:decode_attention",
        ))
        .stderr(predicate::str::contains("mixed-batch dummy runs"));
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
        .failure()
        .stderr(predicate::str::contains(
            "scoped subset build is not semantically valid for compile-region:kv_cache_update",
        ))
        .stderr(predicate::str::contains("mixed prefill/decode warmup"));
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
