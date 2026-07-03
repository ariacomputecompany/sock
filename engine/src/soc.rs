use std::collections::{BTreeMap, BTreeSet};

use sock_core::{
    IntegrationScopeKind, SocMaterializationMode, SocNamespacePlan, SocPlanDocument,
    SocSelectorSnapshot, VllmIntegrationDocument,
};

use crate::{BuildReadiness, BuildScope, BuildTopologyScope, PlanningOutcome};

pub fn build_soc_plan_document(
    outcome: &PlanningOutcome,
    scope: &BuildScope,
    integration: &VllmIntegrationDocument,
) -> SocPlanDocument {
    let mut namespaces = BTreeMap::<String, SocNamespacePlan>::new();

    for surface in &integration.surfaces {
        let Some(namespace_name) = surface.cache_namespace.as_deref() else {
            continue;
        };
        let namespace = namespaces
            .entry(namespace_name.to_owned())
            .or_insert_with(|| SocNamespacePlan {
                namespace: namespace_name.to_owned(),
                scope_kind: surface.scope_kind,
                materialization_mode: SocMaterializationMode::Lazy,
                subset_build_valid: true,
                direct_entrypoint_invocable: true,
                artifact_scopes: Vec::new(),
                artifact_classes: Vec::new(),
                required_artifacts: Vec::new(),
                warmup_scopes: Vec::new(),
                warmup_proof_ids: Vec::new(),
                replay_root_ids: Vec::new(),
                source_surface_ids: Vec::new(),
                source_callables: Vec::new(),
                rationale: String::new(),
            });
        if surface.scope_kind == IntegrationScopeKind::CacheSurface {
            namespace.scope_kind = IntegrationScopeKind::CacheSurface;
        }
        namespace.subset_build_valid &= surface.isolation.subset_build_valid;
        namespace.direct_entrypoint_invocable &= surface.isolation.direct_entrypoint_invocable;
        namespace.source_surface_ids.push(surface.id.clone());
        namespace.source_callables.push(format!(
            "{}::{}",
            surface.primary.module, surface.primary.callable
        ));
        if !namespace.rationale.is_empty() {
            namespace.rationale.push(' ');
        }
        namespace.rationale.push_str(&surface.rationale);
        if let Some(warmup_scope) = &surface.warmup_scope {
            namespace.warmup_scopes.push(warmup_scope.clone());
        }
    }

    for requirement in &outcome.plan.artifact_requirements {
        let namespace_name = cache_namespace_for_scope(outcome, &requirement.scope);
        if let Some(namespace) = namespaces.get_mut(&namespace_name) {
            namespace.artifact_scopes.push(requirement.scope.clone());
            namespace
                .artifact_classes
                .push(requirement.class.as_str().to_owned());
            namespace.required_artifacts.push(format!(
                "{}:{}",
                requirement.class.as_str(),
                requirement.scope
            ));
        }
    }

    for obligation in &outcome.plan.warmup_obligations {
        let namespace_name = cache_namespace_for_scope(outcome, &obligation.region_name);
        if let Some(namespace) = namespaces.get_mut(&namespace_name) {
            namespace.warmup_scopes.push(obligation.region_name.clone());
            namespace
                .warmup_proof_ids
                .push(obligation.proof.proof_id.clone());
            namespace.materialization_mode =
                match (namespace.materialization_mode, obligation.blocking) {
                    (_, true) => SocMaterializationMode::EagerBlocking,
                    (SocMaterializationMode::Lazy, false) => SocMaterializationMode::EagerDeferred,
                    (existing, false) => existing,
                };
        }
    }

    for root in &integration.replay_roots {
        if let Some(namespace_name) = &root.cache_namespace {
            if let Some(namespace) = namespaces.get_mut(namespace_name) {
                namespace.replay_root_ids.push(root.id.clone());
            }
        }
    }

    let mut namespace_plans = namespaces.into_values().collect::<Vec<_>>();
    for namespace in &mut namespace_plans {
        dedup_and_sort(&mut namespace.artifact_scopes);
        dedup_and_sort(&mut namespace.artifact_classes);
        dedup_and_sort(&mut namespace.required_artifacts);
        dedup_and_sort(&mut namespace.warmup_scopes);
        dedup_and_sort(&mut namespace.warmup_proof_ids);
        dedup_and_sort(&mut namespace.replay_root_ids);
        dedup_and_sort(&mut namespace.source_surface_ids);
        dedup_and_sort(&mut namespace.source_callables);
    }
    namespace_plans.sort_by(|left, right| left.namespace.cmp(&right.namespace));

    let mut replay_root_ids = integration
        .replay_roots
        .iter()
        .map(|root| root.id.clone())
        .collect::<Vec<_>>();
    replay_root_ids.sort();

    let mut shared_abstractions = integration
        .surfaces
        .iter()
        .flat_map(|surface| surface.preserved_abstractions.clone())
        .collect::<Vec<_>>();
    dedup_and_sort(&mut shared_abstractions);

    SocPlanDocument {
        schema_version: sock_core::SchemaVersion::current(),
        plan_identity: outcome.plan.structural_identity.plan_identity.clone(),
        derivation_strategy: "derived_from_resolved_build_plan_and_vllm_integration".to_owned(),
        selectors: selector_snapshot(scope),
        namespaces: namespace_plans,
        replay_root_ids,
        shared_abstractions,
    }
}

fn selector_snapshot(scope: &BuildScope) -> SocSelectorSnapshot {
    let mut requested_backend_families = scope
        .backend_families
        .iter()
        .map(|family| family.as_str().to_owned())
        .collect::<Vec<_>>();
    requested_backend_families.sort();
    let mut requested_topology_scopes = scope
        .topology_scopes
        .iter()
        .map(|scope| match scope {
            BuildTopologyScope::Shared => "shared".to_owned(),
            BuildTopologyScope::RankLocal => "rank_local".to_owned(),
        })
        .collect::<Vec<_>>();
    requested_topology_scopes.sort();
    SocSelectorSnapshot {
        requested_regions: scope.region_names.iter().cloned().collect(),
        requested_artifact_scopes: scope.artifact_scopes.iter().cloned().collect(),
        requested_backend_families,
        requested_topology_scopes,
        requested_cache_namespaces: scope.cache_namespaces.iter().cloned().collect(),
        requested_warmup_scopes: scope.warmup_scopes.iter().cloned().collect(),
        requested_readiness: match scope.readiness {
            Some(BuildReadiness::EarlyServe) => "early_serve".to_owned(),
            Some(BuildReadiness::Correctness) => "correctness".to_owned(),
            Some(BuildReadiness::Performance) => "performance".to_owned(),
            None => "default".to_owned(),
        },
    }
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

fn dedup_and_sort(values: &mut Vec<String>) {
    let deduped = values.iter().cloned().collect::<BTreeSet<_>>();
    *values = deduped.into_iter().collect();
}
