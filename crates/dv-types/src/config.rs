use crate::DistanceMetric;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum IndexKind {
    #[default]
    Hnsw,
    Flat,
    ZColumn,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    pub m: usize,
    pub m_max0: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub seed: u64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m_max0: 32,
            ef_construction: 200,
            ef_search: 64,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZColumnConfig {
    pub outer_grid: (u16, u16),
    pub max_layers: u8,
    pub pitch_ratio: f32,
    pub rebalance_interval: u64,
    pub ef_search: usize,
}

impl Default for ZColumnConfig {
    fn default() -> Self {
        Self {
            outer_grid: (8, 8),
            max_layers: 3,
            pitch_ratio: 0.5,
            rebalance_interval: 1000,
            ef_search: 64,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    pub name: String,
    pub dimension: usize,
    pub metric: DistanceMetric,
    pub index_kind: IndexKind,
    pub hnsw: HnswConfig,
    #[serde(default)]
    pub zcolumn: ZColumnConfig,
}

impl CollectionConfig {
    pub fn new(name: impl Into<String>, dimension: usize, metric: DistanceMetric) -> Self {
        Self {
            name: name.into(),
            dimension,
            metric,
            index_kind: IndexKind::Hnsw,
            hnsw: HnswConfig::default(),
            zcolumn: ZColumnConfig::default(),
        }
    }

    pub fn with_flat_index(mut self) -> Self {
        self.index_kind = IndexKind::Flat;
        self
    }

    pub fn with_zcolumn_index(mut self) -> Self {
        self.index_kind = IndexKind::ZColumn;
        self
    }
}
