use dv_types::VectorId;
use serde::{Deserialize, Serialize};

/// POST /topolsea/v1/shard/query body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardQueryRequest {
    pub vector: Vec<f32>,
    pub top_k: usize,
    pub ef: usize,
}

/// Single hit in a shard query response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardQueryHit {
    pub id: Option<String>,
    pub internal_id: u64,
    pub distance: f32,
    pub score: f32,
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

#[derive(Debug, thiserror::Error)]
pub enum ShardRemoteError {
    #[error("HTTP {status}: {body}")]
    Http { status: u16, body: String },
    #[error("transport: {0}")]
    Transport(String),
    #[error("serde: {0}")]
    Serde(String),
}

pub const QUERY_PATH: &str = "/topolsea/v1/shard/query";
