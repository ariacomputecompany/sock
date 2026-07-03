use std::collections::BTreeSet;

use sock_core::BackendFamily;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildReadiness {
    Correctness,
    Performance,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BuildScope {
    pub region_names: BTreeSet<String>,
    pub artifact_scopes: BTreeSet<String>,
    pub backend_families: BTreeSet<BackendFamily>,
    pub cache_namespaces: BTreeSet<String>,
    pub warmup_scopes: BTreeSet<String>,
    pub readiness: Option<BuildReadiness>,
}

impl BuildScope {
    #[must_use]
    pub fn has_subset_selectors(&self) -> bool {
        !self.region_names.is_empty()
            || !self.artifact_scopes.is_empty()
            || !self.backend_families.is_empty()
            || !self.cache_namespaces.is_empty()
            || !self.warmup_scopes.is_empty()
    }

    #[must_use]
    pub fn is_unscoped(&self) -> bool {
        !self.has_subset_selectors() && self.readiness.is_none()
    }

    #[must_use]
    pub fn allows_region(&self, canonical_name: &str) -> bool {
        (self.region_names.is_empty() && self.artifact_scopes.is_empty())
            || self.region_names.contains(canonical_name)
            || self.artifact_scopes.contains(canonical_name)
    }

    #[must_use]
    pub fn allows_artifact_scope(&self, scope: &str) -> bool {
        (self.region_names.is_empty() && self.artifact_scopes.is_empty())
            || self.artifact_scopes.contains(scope)
            || self.region_names.contains(scope)
    }

    #[must_use]
    pub fn allows_backend_family(&self, family: BackendFamily) -> bool {
        self.backend_families.is_empty() || self.backend_families.contains(&family)
    }

    #[must_use]
    pub fn allows_cache_namespace(&self, cache_namespace: &str) -> bool {
        self.cache_namespaces.is_empty() || self.cache_namespaces.contains(cache_namespace)
    }

    #[must_use]
    pub fn allows_warmup_scope(&self, warmup_scope: &str) -> bool {
        self.warmup_scopes.is_empty() || self.warmup_scopes.contains(warmup_scope)
    }
}
