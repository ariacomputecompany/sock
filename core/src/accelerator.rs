use serde::{Deserialize, Serialize};

use crate::{AcceleratorVendor, BackendFamily};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AcceleratorRuntimeProfile {
    Cuda,
    RocmWsl,
    Python,
}

impl AcceleratorRuntimeProfile {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::RocmWsl => "rocm-wsl",
            Self::Python => "python",
        }
    }

    #[must_use]
    pub const fn vllm_target_device(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::RocmWsl => "rocm",
            Self::Python => "cpu",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NvidiaArchitectureClass {
    Ampere,
    Ada,
    Hopper,
    Blackwell,
}

impl NvidiaArchitectureClass {
    #[must_use]
    pub const fn supports_flashinfer(self) -> bool {
        true
    }

    #[must_use]
    pub const fn supports_cuda_graphs(self) -> bool {
        true
    }

    #[must_use]
    pub const fn supports_fp8(self) -> bool {
        matches!(self, Self::Ada | Self::Hopper | Self::Blackwell)
    }

    #[must_use]
    pub const fn supports_tma(self) -> bool {
        matches!(self, Self::Hopper | Self::Blackwell)
    }

    #[must_use]
    pub const fn supports_nvfp4(self) -> bool {
        matches!(self, Self::Blackwell)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AmdArchitectureClass {
    Gfx11,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcceleratorRuntimeContract {
    pub vendor: AcceleratorVendor,
    pub profile: AcceleratorRuntimeProfile,
    pub vllm_target_device: String,
    pub preferred_backend_families: Vec<BackendFamily>,
    pub default_tensor_parallelism: u16,
    pub env_defaults: Vec<(String, String)>,
    pub required_witnesses: Vec<String>,
    pub fail_closed_reasons: Vec<String>,
}

impl AcceleratorRuntimeContract {
    #[must_use]
    pub fn for_host(
        vendor: AcceleratorVendor,
        gpu_arches: &[String],
        device_count: u16,
        flashinfer_prebuilt_available: bool,
    ) -> Self {
        match vendor {
            AcceleratorVendor::Nvidia => {
                cuda_contract(gpu_arches, device_count, flashinfer_prebuilt_available)
            }
            AcceleratorVendor::Amd => rocm_wsl_contract(device_count),
            AcceleratorVendor::Unknown => python_contract(),
        }
    }

    #[must_use]
    pub fn env_defaults(&self) -> Vec<(String, String)> {
        let mut values = vec![
            (
                "SOCK_RUNTIME_PROFILE".to_owned(),
                self.profile.as_str().to_owned(),
            ),
            (
                "VLLM_TARGET_DEVICE".to_owned(),
                self.vllm_target_device.clone(),
            ),
        ];
        values.extend(self.env_defaults.clone());
        values.sort();
        values.dedup();
        values
    }
}

#[must_use]
pub fn parse_nvidia_architecture_class(arch: &str) -> Option<NvidiaArchitectureClass> {
    let sm = arch
        .trim()
        .strip_prefix("sm")
        .or_else(|| arch.trim().strip_prefix("sm_"))
        .and_then(|digits| digits.parse::<u16>().ok())?;
    match sm {
        80 | 86 | 87 => Some(NvidiaArchitectureClass::Ampere),
        89 => Some(NvidiaArchitectureClass::Ada),
        90 => Some(NvidiaArchitectureClass::Hopper),
        100.. => Some(NvidiaArchitectureClass::Blackwell),
        _ => None,
    }
}

#[must_use]
pub fn parse_amd_architecture_class(arch: &str) -> Option<AmdArchitectureClass> {
    if arch.trim().starts_with("gfx11") {
        Some(AmdArchitectureClass::Gfx11)
    } else if arch.trim().starts_with("gfx") {
        Some(AmdArchitectureClass::Other)
    } else {
        None
    }
}

fn cuda_contract(
    gpu_arches: &[String],
    device_count: u16,
    flashinfer_prebuilt_available: bool,
) -> AcceleratorRuntimeContract {
    let classes = gpu_arches
        .iter()
        .filter_map(|arch| parse_nvidia_architecture_class(arch))
        .collect::<Vec<_>>();
    let mut fail_closed_reasons = Vec::new();
    if gpu_arches.is_empty() {
        fail_closed_reasons
            .push("CUDA selected but no NVIDIA compute capability was discovered".to_owned());
    }
    if classes.is_empty() {
        fail_closed_reasons.push(format!(
            "CUDA selected but architectures are unsupported by sock: {}",
            gpu_arches.join(",")
        ));
    }
    if !flashinfer_prebuilt_available {
        fail_closed_reasons.push(
            "FlashInfer was selected by default but no prebuilt/runtime witness is available"
                .to_owned(),
        );
    }

    let mut preferred_backend_families = vec![BackendFamily::Triton];
    if flashinfer_prebuilt_available {
        preferred_backend_families.insert(0, BackendFamily::FlashInfer);
    }
    if classes.iter().any(|class| class.supports_cuda_graphs()) {
        preferred_backend_families.push(BackendFamily::CudaGraphs);
    }
    if classes.iter().any(|class| class.supports_tma()) {
        preferred_backend_families.push(BackendFamily::AotInductor);
    }
    preferred_backend_families.sort();
    preferred_backend_families.dedup();
    preferred_backend_families.sort_by_key(|family| match family {
        BackendFamily::FlashInfer => 0,
        BackendFamily::Triton => 1,
        BackendFamily::CudaGraphs => 2,
        BackendFamily::AotInductor => 3,
    });

    AcceleratorRuntimeContract {
        vendor: AcceleratorVendor::Nvidia,
        profile: AcceleratorRuntimeProfile::Cuda,
        vllm_target_device: "cuda".to_owned(),
        preferred_backend_families,
        default_tensor_parallelism: if device_count >= 2 { 2 } else { 1 },
        env_defaults: vec![
            ("CUDA_DEVICE_ORDER".to_owned(), "PCI_BUS_ID".to_owned()),
            ("CUDA_MODULE_LOADING".to_owned(), "LAZY".to_owned()),
            (
                "VLLM_WORKER_MULTIPROC_METHOD".to_owned(),
                "spawn".to_owned(),
            ),
            ("VLLM_USE_V1".to_owned(), "1".to_owned()),
            ("VLLM_USE_V2_MODEL_RUNNER".to_owned(), "1".to_owned()),
        ],
        required_witnesses: vec![
            "nvidia.compute_capability".to_owned(),
            "nvidia.driver_version".to_owned(),
            "cuda.runtime_version".to_owned(),
            "cuda.device_order.pci_bus_id".to_owned(),
        ],
        fail_closed_reasons,
    }
}

fn rocm_wsl_contract(device_count: u16) -> AcceleratorRuntimeContract {
    AcceleratorRuntimeContract {
        vendor: AcceleratorVendor::Amd,
        profile: AcceleratorRuntimeProfile::RocmWsl,
        vllm_target_device: "rocm".to_owned(),
        preferred_backend_families: vec![BackendFamily::Triton],
        default_tensor_parallelism: device_count.max(1).min(1),
        env_defaults: vec![
            ("VLLM_USE_V2_MODEL_RUNNER".to_owned(), "0".to_owned()),
            ("VLLM_WSL2_ENABLE_PIN_MEMORY".to_owned(), "0".to_owned()),
            (
                "VLLM_WORKER_MULTIPROC_METHOD".to_owned(),
                "spawn".to_owned(),
            ),
        ],
        required_witnesses: vec![
            "rocm.gfx_architecture".to_owned(),
            "rocm.runtime_version".to_owned(),
        ],
        fail_closed_reasons: Vec::new(),
    }
}

fn python_contract() -> AcceleratorRuntimeContract {
    AcceleratorRuntimeContract {
        vendor: AcceleratorVendor::Unknown,
        profile: AcceleratorRuntimeProfile::Python,
        vllm_target_device: "cpu".to_owned(),
        preferred_backend_families: Vec::new(),
        default_tensor_parallelism: 1,
        env_defaults: Vec::new(),
        required_witnesses: Vec::new(),
        fail_closed_reasons: vec!["no supported accelerator was discovered".to_owned()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_4090_contract_is_single_rank_and_flashinfer_first() {
        let contract = AcceleratorRuntimeContract::for_host(
            AcceleratorVendor::Nvidia,
            &["sm89".to_owned()],
            1,
            true,
        );

        assert_eq!(contract.profile, AcceleratorRuntimeProfile::Cuda);
        assert_eq!(contract.default_tensor_parallelism, 1);
        assert_eq!(
            contract.preferred_backend_families[0],
            BackendFamily::FlashInfer
        );
        assert!(contract.fail_closed_reasons.is_empty());
        assert!(
            contract
                .env_defaults()
                .contains(&("CUDA_DEVICE_ORDER".to_owned(), "PCI_BUS_ID".to_owned()))
        );
    }

    #[test]
    fn cuda_missing_runtime_witness_fails_closed() {
        let contract = AcceleratorRuntimeContract::for_host(
            AcceleratorVendor::Nvidia,
            &["sm90".to_owned()],
            1,
            false,
        );

        assert!(
            contract
                .fail_closed_reasons
                .iter()
                .any(|reason| reason.contains("FlashInfer"))
        );
        assert_eq!(
            contract.preferred_backend_families[0],
            BackendFamily::Triton
        );
    }

    #[test]
    fn cuda_architecture_feature_gates_are_generation_specific() {
        let ada = parse_nvidia_architecture_class("sm89").expect("ada");
        let hopper = parse_nvidia_architecture_class("sm90").expect("hopper");
        let blackwell = parse_nvidia_architecture_class("sm100").expect("blackwell");

        assert!(ada.supports_fp8());
        assert!(!ada.supports_tma());
        assert!(hopper.supports_tma());
        assert!(!hopper.supports_nvfp4());
        assert!(blackwell.supports_nvfp4());
    }

    #[test]
    fn rocm_contract_remains_triton_only() {
        let contract = AcceleratorRuntimeContract::for_host(
            AcceleratorVendor::Amd,
            &["gfx1151".to_owned()],
            1,
            false,
        );

        assert_eq!(contract.profile, AcceleratorRuntimeProfile::RocmWsl);
        assert_eq!(
            contract.preferred_backend_families,
            vec![BackendFamily::Triton]
        );
        assert_eq!(contract.vllm_target_device, "rocm");
    }
}
