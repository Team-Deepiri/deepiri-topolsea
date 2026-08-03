use dv_types::VectorId;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// POST /topolsea/v1/shard/query body.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ShardQueryRequest {
    pub vector: Vec<f32>,
    pub top_k: usize,
    pub ef: usize,
    /// Optional payload filter (JSON dialect) — applied on the shard node (C11).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<Value>,
}

/// Single hit in a shard query response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardQueryHit {
    pub id: Option<String>,
    pub internal_id: u64,
    pub distance: f32,
    pub score: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

impl ShardQueryHit {
    pub fn vector_id(&self) -> VectorId {
        VectorId(self.internal_id)
    }
}

/// JSON response from a shard node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardQueryResponse {
    pub hits: Vec<ShardQueryHit>,
}

/// POST /topolsea/v1/replicate/upsert — primary → backup sync (C10).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicateUpsertRequest {
    pub collection: String,
    pub ids: Vec<String>,
    pub vectors: Vec<Vec<f32>>,
    #[serde(default)]
    pub metadatas: Option<Vec<Option<Value>>>,
    #[serde(default)]
    pub texts: Option<Vec<Option<String>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicateUpsertResponse {
    pub applied: usize,
}

/// POST /topolsea/v1/replicate/delete
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicateDeleteRequest {
    pub collection: String,
    pub ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicateDeleteResponse {
    pub deleted: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardHealthResponse {
    pub status: String,
    pub collection: Option<String>,
    pub vectors: Option<usize>,
}

#[derive(Debug, thiserror::Error)]
pub enum ShardRemoteError {
    #[error("HTTP {status}: {body}")]
    Http { status: u16, body: String },
    #[error("transport: {0}")]
    Transport(String),
    #[error("serde: {0}")]
    Serde(String),
    #[error("circuit open: {0}")]
    CircuitOpen(String),
}

pub const QUERY_PATH: &str = "/topolsea/v1/shard/query";
pub const REPLICATE_UPSERT_PATH: &str = "/topolsea/v1/replicate/upsert";
pub const REPLICATE_DELETE_PATH: &str = "/topolsea/v1/replicate/delete";
pub const SHARD_HEALTH_PATH: &str = "/topolsea/v1/shard/health";
