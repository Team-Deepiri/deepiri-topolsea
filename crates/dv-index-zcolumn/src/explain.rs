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
    /// Hard V_touch budget applied (0 = unlimited).
    #[serde(default)]
    pub touch_budget: usize,
    /// True when search stopped because `candidate_pool` hit the budget.
    #[serde(default)]
    pub hit_touch_budget: bool,
    /// Vectors scored with quantized distances before prune (M3).
    #[serde(default)]
    pub coarse_scored: u64,
    /// Vectors kept after intra-column prune (M3).
    #[serde(default)]
    pub coarse_kept: u64,
}

impl QueryExplain {
    pub fn new(strategy: impl Into<String>) -> Self {
        Self {
            strategy: strategy.into(),
            ..Default::default()
        }
    }
}
