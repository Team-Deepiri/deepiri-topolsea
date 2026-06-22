use serde::{Deserialize, Serialize};
use std::fmt;

/// Internal monotonic identifier assigned by the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VectorId(pub u64);

impl VectorId {
    pub const INVALID: VectorId = VectorId(u64::MAX);

    pub fn new(id: u64) -> Self {
        Self(id)
    }

    pub fn raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for VectorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// User-facing string identifier for a vector/document.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExternalId(pub String);

impl ExternalId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for ExternalId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for ExternalId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl fmt::Display for ExternalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
