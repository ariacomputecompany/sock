use std::fmt;

use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct CanonicalHash(String);

impl CanonicalHash {
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CanonicalHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Error)]
pub enum CanonicalError {
    #[error("serialization failed: {0}")]
    Serialize(#[from] serde_json::Error),
}

pub fn canonical_json<T>(value: &T) -> Result<String, CanonicalError>
where
    T: Serialize + ?Sized,
{
    let value = serde_json::to_value(value)?;
    let normalized = sort_json(value);
    Ok(serde_json::to_string_pretty(&normalized)?)
}

pub fn parse_canonical_json<T>(input: &str) -> Result<T, CanonicalError>
where
    T: DeserializeOwned,
{
    Ok(serde_json::from_str(input)?)
}

pub fn canonical_hash<T>(value: &T) -> Result<CanonicalHash, CanonicalError>
where
    T: Serialize + ?Sized,
{
    let rendered = canonical_json(value)?;
    let digest = Sha256::digest(rendered.as_bytes());
    Ok(CanonicalHash::new(hex::encode(digest)))
}

fn sort_json(value: Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.into_iter().map(sort_json).collect()),
        Value::Object(entries) => {
            let mut sorted = Map::new();
            let mut keys = entries.keys().cloned().collect::<Vec<_>>();
            keys.sort();
            for key in keys {
                let value = entries
                    .get(&key)
                    .cloned()
                    .expect("key collected from map must exist");
                sorted.insert(key, sort_json(value));
            }
            Value::Object(sorted)
        }
        other => other,
    }
}
