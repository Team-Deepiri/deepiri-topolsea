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

#[derive(Debug, Clone)]
pub struct QueryResult {
    pub id: Option<String>,
    pub internal_id: dv_types::VectorId,
    pub distance: f32,
    pub score: f32,
    pub metadata: Option<Value>,
}
