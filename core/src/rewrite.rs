use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RewritePhase {
    NormalizeRequest,
    SurveyEngine,
    SelectBackends,
    DiscoverCompileRegions,
    BuildShapeEnvelope,
    ElaborateWarmup,
    PlanMaterialization,
    ShapeGuarantees,
    EmitArtifacts,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RewritePassContract {
    pub name: String,
    pub phase: RewritePhase,
    pub required_inputs: Vec<String>,
    pub produced_outputs: Vec<String>,
    pub preserved_invariants: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PassTrace {
    pub contract: RewritePassContract,
    pub before_identity: String,
    pub after_identity: String,
    pub matched_rules: Vec<String>,
    pub repairs: Vec<String>,
    pub invalidated_assumptions: Vec<String>,
    pub validated_invariants: Vec<String>,
    pub violations: Vec<String>,
}

impl RewritePassContract {
    #[must_use]
    pub fn new(
        name: &str,
        phase: RewritePhase,
        required_inputs: Vec<&str>,
        produced_outputs: Vec<&str>,
        preserved_invariants: Vec<&str>,
    ) -> Self {
        Self {
            name: name.to_owned(),
            phase,
            required_inputs: required_inputs.into_iter().map(str::to_owned).collect(),
            produced_outputs: produced_outputs.into_iter().map(str::to_owned).collect(),
            preserved_invariants: preserved_invariants
                .into_iter()
                .map(str::to_owned)
                .collect(),
        }
    }
}

impl PassTrace {
    #[must_use]
    pub fn pass_name(&self) -> &str {
        &self.contract.name
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.contract.name.trim().is_empty() {
            return Err("rewrite pass contract is missing a name".to_owned());
        }
        if self.before_identity.trim().is_empty() || self.after_identity.trim().is_empty() {
            return Err(format!(
                "rewrite pass `{}` is missing a before/after identity",
                self.contract.name
            ));
        }
        if self.contract.required_inputs.is_empty() {
            return Err(format!(
                "rewrite pass `{}` has no declared inputs",
                self.contract.name
            ));
        }
        if self.contract.produced_outputs.is_empty() {
            return Err(format!(
                "rewrite pass `{}` has no declared outputs",
                self.contract.name
            ));
        }
        for invariant in &self.validated_invariants {
            if !self.contract.preserved_invariants.contains(invariant) {
                return Err(format!(
                    "rewrite pass `{}` validated undeclared invariant `{invariant}`",
                    self.contract.name
                ));
            }
        }
        if let Some(violation) = self.violations.first() {
            return Err(format!(
                "rewrite pass `{}` violated `{violation}`",
                self.contract.name
            ));
        }
        Ok(())
    }
}
