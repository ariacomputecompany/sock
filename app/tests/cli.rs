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
        .stdout(predicate::str::contains("request contract:"))
        .stdout(predicate::str::contains("expanded closure:"))
        .stdout(predicate::str::contains("estimated work:"))
        .stdout(predicate::str::contains("optimization: level=O2"))
        .stdout(predicate::str::contains("vllm native contract:"))
        .stdout(predicate::str::contains("soc integration:"))
        .stdout(predicate::str::contains("replay root key:"))
        .stdout(predicate::str::contains("rooted vllm replay surfaces:"))
        .stdout(predicate::str::contains("optimization envelope:"))
        .stdout(predicate::str::contains("rewrite trace:"))
        .stdout(predicate::str::contains("diagnostics:"))
        .stdout(predicate::str::contains("verified_bundle"));
}

#[test]
fn o0_explain_reduces_cuda_graph_and_performance_scope() {
    let output = Command::cargo_bin("sock")
        .expect("sock binary")
        .args(["explain", "--format", "json", "-O", "o0"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let explain: Value = serde_json::from_slice(&output).expect("parse explain json");

    assert_eq!(
        explain["plan"]["optimization_envelope"]["level"],
        Value::String("o0".to_owned())
    );
    assert_eq!(
        explain["optimization_explain"]["profile_name"],
        Value::String("minimal_dev".to_owned())
    );
    assert_eq!(
        explain["optimization_explain"]["graph_actions"][0]["effect"],
        Value::String("cuda_graphs=disabled".to_owned())
    );
    assert!(
        explain["plan"]["shape_envelope"]["nodes"]
            .as_array()
            .expect("shape envelope nodes")
            .iter()
            .all(|node| node["plane"] != "CudaGraph" && node["plane"] != "Performance")
    );
}

#[test]
fn prepare_prefill_path_uses_common_intent_contract() {
    let dir = tempdir().expect("tempdir");

    Command::cargo_bin("sock")
        .expect("sock binary")
        .args([
            "prepare",
            "prefill-path",
            "--out",
            dir.path().to_str().expect("utf8 path"),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("readiness=Correctness"))
        .stdout(predicate::str::contains("intent=prefill_path"))
        .stdout(predicate::str::contains("requested selectors:"))
        .stdout(predicate::str::contains("regions=prefill_attention"));

    let plan: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("buildplan.json")).expect("read buildplan"),
    )
    .expect("parse buildplan");
    let compile_regions = plan["compile_regions"]
        .as_array()
        .expect("compile regions array");
    assert_eq!(compile_regions.len(), 1);
    assert_eq!(compile_regions[0]["name"], "prefill_attention");
    assert_eq!(compile_regions[0]["cache_namespace"], "compile-cache");
    assert_eq!(compile_regions[0]["cache_sharing"], "content_addressed");
    assert_eq!(
        compile_regions[0]["portability_scope"],
        "gpu_architecture_family"
    );
    assert_eq!(
        compile_regions[0]["topology_scope"],
        "cross_rank_and_cross_process"
    );
    assert_eq!(compile_regions[0]["warmup_scope"], "prefill_attention");
    assert!(compile_regions[0]["stable_identity"].is_string());
    assert!(compile_regions[0]["equivalence_identity"].is_string());
    assert!(
        compile_regions[0]["closure_verification_criteria"]
            .as_array()
            .expect("closure verification criteria")
            .len()
            >= 2
    );
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
        "backend_decision.json",
        "buildplan.json",
        "bundle_metadata.json",
        "diagnostics.json",
        "explain.txt",
        "materialization_report.json",
        "optimization_explain.json",
        "replay_proof.json",
        "replay.sh",
        "rewrite_trace.json",
        "soc_plan.json",
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
    assert!(
        integration["replay_roots"]
            .as_array()
            .expect("integration replay roots")
            .iter()
            .any(|root| root["surface_id"] == "compile-region:prefill_attention")
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

    let replay_proof: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("replay_proof.json")).expect("read replay proof"),
    )
    .expect("parse replay proof");
    assert_eq!(
        replay_proof["contradiction_contract"],
        Value::String("same_requested_plan_requires_same_result_artifact_identity".to_owned())
    );
    assert!(
        replay_proof["realization_identity"]
            .as_str()
            .expect("realization identity")
            .len()
            > 10
    );
    let explain_text =
        std::fs::read_to_string(dir.path().join("explain.txt")).expect("read explain text");
    assert!(explain_text.contains("identity lattice:"));
    assert!(explain_text.contains("replay proof:"));
    assert!(explain_text.contains("realization_mode="));
    assert!(explain_text.contains("backend decision:"));

    let backend_decision: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("backend_decision.json"))
            .expect("read backend decision"),
    )
    .expect("parse backend decision");
    assert!(
        backend_decision["entries"]
            .as_array()
            .expect("backend decision entries")
            .iter()
            .any(
                |entry| entry["family"] == "FlashInfer" && entry["selected_for_deployment"] == true
            )
    );
    assert!(
        backend_decision["extension_manifests"]
            .as_array()
            .expect("extension manifests")
            .iter()
            .any(|manifest| manifest["binary_name"] == "flashinfer_extension.so")
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
        .stdout(predicate::str::contains("identity lattice:"))
        .stdout(predicate::str::contains("replay proof:"))
        .stdout(predicate::str::contains("backend decision:"))
        .stdout(predicate::str::contains("realization_mode="))
        .stdout(predicate::str::contains("vllm replay roots key="))
        .stdout(predicate::str::contains("soc plan key="))
        .stdout(predicate::str::contains("verification Passed"))
        .stdout(predicate::str::contains("runtime-jit evidence:"))
        .stdout(predicate::str::contains(
            "replay compile_free=true forbidden_queues=Compile,Assemble,ArtifactIo,Warmup",
        ))
        .stdout(predicate::str::contains("[info] verified_bundle"));
}

#[test]
fn soc_plan_maps_cache_namespaces_to_artifacts_and_warmups() {
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

    let soc: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("soc_plan.json")).expect("read soc plan"),
    )
    .expect("parse soc plan");
    assert_eq!(
        soc["derivation_strategy"],
        Value::String("derived_from_resolved_build_plan_and_vllm_integration".to_owned())
    );
    assert_eq!(
        soc["selectors"]["requested_regions"],
        Value::Array(vec![Value::String("prefill_attention".to_owned())])
    );
    let namespaces = soc["namespaces"].as_array().expect("namespaces array");
    assert!(namespaces.iter().any(|namespace| {
        namespace["namespace"] == "compile-cache"
            && namespace["materialization_mode"] == "eager_blocking"
            && namespace["source_surface_ids"]
                .as_array()
                .expect("source surfaces")
                .iter()
                .any(|surface| surface == "compile-region:prefill_attention")
    }));
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

    let replay_proof: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("replay_proof.json")).expect("read replay proof"),
    )
    .expect("parse replay proof");
    assert_eq!(
        replay_proof["realization_mode"],
        Value::String("reused_only".to_owned())
    );
    assert_eq!(
        replay_proof["contradiction_contract"],
        Value::String("same_requested_plan_requires_same_result_artifact_identity".to_owned())
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
        artifact["cache_sharing"] == "content_addressed"
            && artifact["region_stable_identity"].is_string()
            && artifact["region_equivalence_identity"].is_string()
    }));
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

#[test]
fn measure_reports_phase_and_duplication_telemetry() {
    let dir = tempdir().expect("tempdir");

    Command::cargo_bin("sock")
        .expect("sock binary")
        .args([
            "measure",
            "prefill-path",
            "--out",
            dir.path().to_str().expect("utf8 path"),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("measurement intent=prefill_path"));

    let report: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("measurement_report.json"))
            .expect("read measurement report"),
    )
    .expect("parse measurement report");
    for case_name in ["broad_cold", "scoped_cold", "scoped_warm"] {
        let case = &report[case_name];
        assert!(case.get("plan_identity").is_some(), "missing plan_identity");
        assert!(
            case.get("replay_plan_identity").is_some(),
            "missing replay_plan_identity"
        );
        assert!(
            case.get("phase_timings").is_some(),
            "missing phase timings object"
        );
        assert!(case["phase_timings"].get("configure_ms").is_some());
        assert!(case["phase_timings"].get("compile_ms").is_some());
        assert!(case["phase_timings"].get("link_assemble_ms").is_some());
        assert!(case["phase_timings"].get("packaging_ms").is_some());
        assert!(
            case["phase_timings"]
                .get("warmup_materialization_ms")
                .is_some()
        );
        assert!(case["phase_timings"].get("verification_ms").is_some());
        assert!(case.get("unique_artifact_count").is_some());
        assert!(case.get("duplicate_artifact_count").is_some());
        assert!(case.get("artifact_deserialization_ms").is_some());
        assert!(case.get("duplicate_rank_local_compile_count").is_some());
        assert!(case.get("duplicate_rank_local_load_count").is_some());
        assert!(case.get("closure_outcome").is_some());
    }
    assert!(
        report["scoped_vs_broad"]
            .get("baseline_plan_identity")
            .is_some()
    );
    assert!(
        report["scoped_vs_broad"]
            .get("candidate_plan_identity")
            .is_some()
    );
    assert!(report["scoped_vs_broad"].get("changed_phases").is_some());
}

#[test]
fn benchmark_matrix_is_versioned_and_tied_to_manifests() {
    let dir = tempdir().expect("tempdir");

    Command::cargo_bin("sock")
        .expect("sock binary")
        .args([
            "benchmark",
            "--out",
            dir.path().to_str().expect("utf8 path"),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("benchmark_matrix entries=4"))
        .stdout(predicate::str::contains(
            "benchmark_trace_scenario=tests/benchmark.matrix.fozzy.json",
        ));

    let report: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("benchmark_matrix.json"))
            .expect("read benchmark matrix"),
    )
    .expect("parse benchmark matrix");
    assert_eq!(report["benchmark_program_version"], 1);
    assert_eq!(
        report["verification_manifest_path"],
        Value::String("fozzy/verification_program.json".to_owned())
    );
    assert_eq!(
        report["benchmark_trace_scenario"],
        Value::String("tests/benchmark.matrix.fozzy.json".to_owned())
    );
    assert_eq!(
        report["entries"].as_array().expect("entries array").len(),
        4
    );
    assert!(
        report["entries"]
            .as_array()
            .expect("entries array")
            .iter()
            .any(
                |entry| entry["label"] == "selected_backend_flashinfer_prefill"
                    && entry["selected_backend_only"] == Value::Bool(true)
            )
    );
    assert!(
        report["entries"]
            .as_array()
            .expect("entries array")
            .iter()
            .all(|entry| entry.get("artifact_paths").is_some())
    );
    assert!(
        report["entries"]
            .as_array()
            .expect("entries array")
            .iter()
            .all(|entry| entry.get("trace_references").is_some())
    );
}
