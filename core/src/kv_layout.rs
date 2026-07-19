use std::str::FromStr;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{AcceleratorVendor, BackendFamily, RequestedEnvironment};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvLayoutId {
    StandardPaged,
    TmhFidelityPaged,
}

impl KvLayoutId {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StandardPaged => "standard",
            Self::TmhFidelityPaged => "tmh",
        }
    }

    #[must_use]
    pub const fn vllm_policy(self) -> &'static str {
        match self {
            Self::StandardPaged => "off",
            Self::TmhFidelityPaged => "accounting",
        }
    }

    #[must_use]
    pub const fn layout_backend_id(self) -> &'static str {
        match self {
            Self::StandardPaged => "standard_paged_kv",
            Self::TmhFidelityPaged => "tmh_fidelity_paged_kv",
        }
    }
}

impl Default for KvLayoutId {
    fn default() -> Self {
        Self::StandardPaged
    }
}

impl FromStr for KvLayoutId {
    type Err = KvLayoutParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "standard" | "standard_paged" | "standard_paged_kv" => Ok(Self::StandardPaged),
            "tmh" | "tmh_fidelity_paged" | "tmh_fidelity_paged_kv" => Ok(Self::TmhFidelityPaged),
            other => Err(KvLayoutParseError {
                value: other.to_owned(),
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("unsupported KV layout {value:?}; supported values are standard, tmh")]
pub struct KvLayoutParseError {
    value: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvLayoutRuntimeMode {
    Standard,
    Accounting,
    Physical,
}

impl KvLayoutRuntimeMode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Accounting => "accounting",
            Self::Physical => "physical",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct KvLayoutPolicy {
    pub layout: KvLayoutId,
    pub runtime_mode: KvLayoutRuntimeMode,
    pub hot_budget_pct: u8,
}

impl KvLayoutPolicy {
    pub const DEFAULT_HOT_BUDGET_PCT: u8 = 25;

    #[must_use]
    pub const fn standard() -> Self {
        Self {
            layout: KvLayoutId::StandardPaged,
            runtime_mode: KvLayoutRuntimeMode::Standard,
            hot_budget_pct: Self::DEFAULT_HOT_BUDGET_PCT,
        }
    }

    #[must_use]
    pub const fn tmh_accounting() -> Self {
        Self {
            layout: KvLayoutId::TmhFidelityPaged,
            runtime_mode: KvLayoutRuntimeMode::Accounting,
            hot_budget_pct: Self::DEFAULT_HOT_BUDGET_PCT,
        }
    }

    #[must_use]
    pub const fn tmh_physical() -> Self {
        Self {
            layout: KvLayoutId::TmhFidelityPaged,
            runtime_mode: KvLayoutRuntimeMode::Physical,
            hot_budget_pct: Self::DEFAULT_HOT_BUDGET_PCT,
        }
    }

    pub fn canonicalize(&mut self) {
        if self.layout == KvLayoutId::StandardPaged {
            self.runtime_mode = KvLayoutRuntimeMode::Standard;
        }
    }

    #[must_use]
    pub const fn vllm_tmh_policy(&self) -> &'static str {
        match self.runtime_mode {
            KvLayoutRuntimeMode::Standard => "off",
            KvLayoutRuntimeMode::Accounting => "accounting",
            KvLayoutRuntimeMode::Physical => "physical",
        }
    }
}

impl Default for KvLayoutPolicy {
    fn default() -> Self {
        Self::standard()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvLayoutBackend {
    pub id: KvLayoutId,
    pub name: String,
    pub physical_storage_supported: bool,
    pub accounting_supported: bool,
    pub supported_vendors: Vec<AcceleratorVendor>,
    pub compatible_attention_backends: Vec<BackendFamily>,
    pub fail_closed_reasons: Vec<String>,
}

impl KvLayoutBackend {
    #[must_use]
    pub fn for_policy(policy: &KvLayoutPolicy) -> Self {
        match policy.layout {
            KvLayoutId::StandardPaged => Self {
                id: KvLayoutId::StandardPaged,
                name: "standard paged KV".to_owned(),
                physical_storage_supported: true,
                accounting_supported: false,
                supported_vendors: vec![AcceleratorVendor::Nvidia, AcceleratorVendor::Amd],
                compatible_attention_backends: vec![
                    BackendFamily::FlashInfer,
                    BackendFamily::Triton,
                    BackendFamily::AotInductor,
                    BackendFamily::CudaGraphs,
                ],
                fail_closed_reasons: Vec::new(),
            },
            KvLayoutId::TmhFidelityPaged => Self {
                id: KvLayoutId::TmhFidelityPaged,
                name: "TMH fidelity paged KV".to_owned(),
                physical_storage_supported: false,
                accounting_supported: true,
                supported_vendors: vec![AcceleratorVendor::Nvidia, AcceleratorVendor::Amd],
                compatible_attention_backends: vec![
                    BackendFamily::FlashInfer,
                    BackendFamily::Triton,
                ],
                fail_closed_reasons: match policy.runtime_mode {
                    KvLayoutRuntimeMode::Physical => vec![
                        "physical TMH requires mixed-fidelity warm-page tensors".to_owned(),
                        "physical TMH requires layout-aware attention kernels".to_owned(),
                    ],
                    _ => Vec::new(),
                },
            },
        }
    }

    pub fn validate(
        &self,
        policy: &KvLayoutPolicy,
        environment: &RequestedEnvironment,
        preferred_families: &[BackendFamily],
    ) -> Result<(), KvLayoutCompatibilityError> {
        if !self
            .supported_vendors
            .contains(&environment.accelerator_vendor)
        {
            return Err(KvLayoutCompatibilityError::UnsupportedVendor {
                layout: self.id,
                vendor: environment.accelerator_vendor,
            });
        }
        if policy.runtime_mode == KvLayoutRuntimeMode::Physical && !self.physical_storage_supported
        {
            return Err(KvLayoutCompatibilityError::PhysicalUnavailable {
                layout: self.id,
                reasons: self.fail_closed_reasons.clone(),
            });
        }
        if policy.runtime_mode == KvLayoutRuntimeMode::Accounting && !self.accounting_supported {
            return Err(KvLayoutCompatibilityError::AccountingUnavailable { layout: self.id });
        }
        if preferred_families
            .iter()
            .all(|family| !self.compatible_attention_backends.contains(family))
        {
            return Err(KvLayoutCompatibilityError::NoCompatibleAttentionBackend {
                layout: self.id,
                preferred_families: preferred_families.to_vec(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum KvLayoutCompatibilityError {
    #[error("KV layout {layout:?} does not support accelerator vendor {vendor:?}")]
    UnsupportedVendor {
        layout: KvLayoutId,
        vendor: AcceleratorVendor,
    },
    #[error("KV layout {layout:?} does not support accounting mode")]
    AccountingUnavailable { layout: KvLayoutId },
    #[error("KV layout {layout:?} physical mode is unavailable: {}", reasons.join("; "))]
    PhysicalUnavailable {
        layout: KvLayoutId,
        reasons: Vec<String>,
    },
    #[error(
        "KV layout {layout:?} has no compatible attention backend among {preferred_families:?}"
    )]
    NoCompatibleAttentionBackend {
        layout: KvLayoutId,
        preferred_families: Vec<BackendFamily>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_public_layout_names() {
        assert_eq!(
            "standard".parse::<KvLayoutId>(),
            Ok(KvLayoutId::StandardPaged)
        );
        assert_eq!(
            "tmh".parse::<KvLayoutId>(),
            Ok(KvLayoutId::TmhFidelityPaged)
        );
    }

    #[test]
    fn physical_tmh_fails_closed() {
        let policy = KvLayoutPolicy::tmh_physical();
        let backend = KvLayoutBackend::for_policy(&policy);
        let env = RequestedEnvironment {
            operating_system: crate::OperatingSystem::Linux,
            accelerator_vendor: AcceleratorVendor::Nvidia,
            gpu_arches: vec!["sm89".to_owned()],
            cuda_version: "12.8".to_owned(),
            driver_version: "570.0".to_owned(),
            python_abi: "cp312".to_owned(),
            libc_abi: "glibc-2.39".to_owned(),
        };

        let error = backend
            .validate(&policy, &env, &[BackendFamily::FlashInfer])
            .expect_err("physical TMH must not silently fall back");

        assert!(matches!(
            error,
            KvLayoutCompatibilityError::PhysicalUnavailable { .. }
        ));
    }
}
