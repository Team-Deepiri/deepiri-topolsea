use thiserror::Error;

pub type Result<T> = std::result::Result<T, TopolseaError>;

#[derive(Debug, Error)]
pub enum TopolseaError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("vector not found: {0}")]
    NotFound(String),

    #[error("collection not found: {0}")]
    CollectionNotFound(String),

    #[error("collection already exists: {0}")]
    CollectionExists(String),

    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("storage error: {0}")]
    Storage(String),

    #[error("index error: {0}")]
    Index(String),

    #[error("metadata error: {0}")]
    Metadata(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}
