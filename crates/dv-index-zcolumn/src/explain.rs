use serde::{Deserialize, Serialize};

/// Explain payload for a Z-Column query — the "callback reverse" audit trail.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryExplain {
    pub entry_layer: u8,
    pub deepest_layer_reached: u8,
    pub revert_count: u64,
    pub columns_scanned: u64,
    pub candidate_pool: usize,
    pub used_fallback_scan: bool,
    pub column_paths: Vec<String>,
    pub strategy: String,
}

impl QueryExplain {
    pub fn new(strategy: impl Into<String>) -> Self {
        Self {
            strategy: strategy.into(),
            ..Default::default()
        }
    }
}
