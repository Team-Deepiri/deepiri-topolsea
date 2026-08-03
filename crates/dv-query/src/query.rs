use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Default)]
pub struct QueryOptions {
    pub top_k: usize,
    pub ef: usize,
    pub include_metadata: bool,
}

#[derive(Debug, Clone)]
pub struct UpsertRecord {
    pub external_id: String,
    pub vector: Vec<f32>,
    pub metadata: Option<Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryExplainResult {
    pub index_kind: String,
    pub entry_layer: Option<u8>,
    pub deepest_layer: Option<u8>,
    pub revert_count: u64,
    pub columns_scanned: u64,
    pub column_paths: Vec<String>,
    pub strategy: String,
    pub planner_reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct QueryResult {
    pub id: Option<String>,
    pub internal_id: dv_types::VectorId,
    pub distance: f32,
    pub score: f32,
    pub metadata: Option<Value>,
}
