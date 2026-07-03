use std::path::{Path, PathBuf};

const REVISION: &str = include_str!("../../vllm/REVISION");

#[must_use]
pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../vllm")
}

#[must_use]
pub fn revision() -> &'static str {
    REVISION.lines().next().unwrap_or("unknown")
}
