use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildReadiness {
    Correctness,
    Performance,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BuildScope {
    pub region_names: BTreeSet<String>,
    pub artifact_scopes: BTreeSet<String>,
    pub readiness: Option<BuildReadiness>,
}

impl BuildScope {
    #[must_use]
    pub fn is_unscoped(&self) -> bool {
        self.region_names.is_empty() && self.artifact_scopes.is_empty() && self.readiness.is_none()
    }

    #[must_use]
    pub fn allows_region(&self, canonical_name: &str) -> bool {
        self.is_unscoped()
            || self.region_names.contains(canonical_name)
            || self.artifact_scopes.contains(canonical_name)
    }

    #[must_use]
    pub fn allows_artifact_scope(&self, scope: &str) -> bool {
        self.is_unscoped()
            || self.artifact_scopes.contains(scope)
            || self.region_names.contains(scope)
    }
}
