use crate::column::ColumnStack;
use crate::compact::CompactionEngine;
use crate::grid::{project_2d, ColumnPath, FractalGrid};
use crate::quant::QuantTier;
use crate::search::{RevertBeamSearch, SearchStats};
use dv_index_api::VectorIndex;
use dv_types::{DistanceMetric, Result, SearchHit, TopolseaError, Vector, VectorId, ZColumnConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Serialize, Deserialize)]
pub struct ZColumnIndex {
    dimension: usize,
    metric: DistanceMetric,
    config: ZColumnConfig,
    grid: FractalGrid,
    vectors: HashMap<VectorId, Vec<f32>>,
    columns: HashMap<String, ColumnStack>,
    #[serde(skip)]
    query_count: AtomicU64,
    #[serde(skip)]
    revert_count: AtomicU64,
    #[serde(skip)]
    columns_scanned: AtomicU64,
    #[serde(skip)]
    compaction: CompactionEngine,
}

impl Clone for ZColumnIndex {
    fn clone(&self) -> Self {
        Self {
            dimension: self.dimension,
            metric: self.metric,
            config: self.config.clone(),
            grid: self.grid.clone(),
            vectors: self.vectors.clone(),
            columns: self.columns.clone(),
            query_count: AtomicU64::new(self.query_count.load(Ordering::Relaxed)),
            revert_count: AtomicU64::new(self.revert_count.load(Ordering::Relaxed)),
            columns_scanned: AtomicU64::new(self.columns_scanned.load(Ordering::Relaxed)),
            compaction: CompactionEngine::new(),
        }
    }
}

impl ZColumnIndex {
    pub fn new(dimension: usize, metric: DistanceMetric, config: ZColumnConfig) -> Self {
        let grid = FractalGrid::new(config.outer_grid, config.max_layers, config.pitch_ratio);
        Self {
            dimension,
            metric,
            config,
            grid,
            vectors: HashMap::new(),
            columns: HashMap::new(),
            query_count: AtomicU64::new(0),
            revert_count: AtomicU64::new(0),
            columns_scanned: AtomicU64::new(0),
            compaction: CompactionEngine::new(),
        }
    }

    fn column_list(&self) -> Vec<ColumnStack> {
        self.columns.values().cloned().collect()
    }

    fn assign_cell(&self, vector: &[f32]) -> (u8, u16, u16) {
        let (px, py) = project_2d(vector);
        let cell = self
            .grid
            .deepest_cell(px, py)
            .unwrap_or(crate::grid::CellCoord::new(0, 0, 0));
        cell.key()
    }

    pub fn rebalance(&mut self) {
        let vectors = self.vectors.clone();
        self.compaction.collapse_and_promote(
            &mut self.grid,
            &mut self.columns,
            &vectors,
            self.dimension,
            self.config.max_layers,
        );
    }

    /// Update access ledgers for columns that served query results.
    pub fn record_access(&mut self, ids: &[VectorId], now_ms: u64) {
        for col in self.columns.values_mut() {
            if col.ids.iter().any(|id| ids.contains(id)) {
                col.ledger.record_hit(now_ms);
            }
        }
        let count = self.query_count.load(Ordering::Relaxed);
        if self.config.rebalance_interval > 0
            && count > 0
            && count.is_multiple_of(self.config.rebalance_interval)
        {
            self.rebalance();
        }
    }

    /// Rebuild column stacks from on-disk fractal segments (fallback when index blob is empty).
    pub fn restore_from_segments(&mut self, dimension: usize, layers: &[(u8, Vec<ColumnStack>)]) {
        self.dimension = dimension;
        self.columns.clear();
        for (_layer, stacks) in layers {
            for stack in stacks {
                if let Some(cell) = stack.cell() {
                    let key = Self::column_key(cell.key());
                    self.columns.insert(key, stack.clone());
                }
            }
        }
    }

    pub fn search_stats(&self) -> SearchStats {
        SearchStats {
            revert_count: self.revert_count.load(Ordering::Relaxed),
            columns_scanned: self.columns_scanned.load(Ordering::Relaxed),
        }
    }

    pub fn compaction_events(&self) -> u64 {
        self.compaction.events
    }

    pub fn grid(&self) -> &FractalGrid {
        &self.grid
    }

    pub fn columns(&self) -> &HashMap<String, ColumnStack> {
        &self.columns
    }

    fn column_key(key: (u8, u16, u16)) -> String {
        format!("{}:{}:{}", key.0, key.1, key.2)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(TopolseaError::Serde)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut idx: Self = serde_json::from_slice(bytes).map_err(TopolseaError::Serde)?;
        idx.query_count = AtomicU64::new(0);
        idx.revert_count = AtomicU64::new(0);
        idx.columns_scanned = AtomicU64::new(0);
        idx.compaction = CompactionEngine::new();
        Ok(idx)
    }

    pub fn ids(&self) -> impl Iterator<Item = VectorId> + '_ {
        self.vectors.keys().copied()
    }
}

impl VectorIndex for ZColumnIndex {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn insert(&mut self, id: VectorId, vector: Vector) -> Result<()> {
        vector.validate_dimension(self.dimension)?;
        if self.vectors.contains_key(&id) {
            return Err(TopolseaError::Index(format!("duplicate id {id}")));
        }

        let data = vector.data;
        let key = self.assign_cell(&data);
        let tier = QuantTier::for_layer(key.0, self.config.max_layers);
        let path = ColumnPath::from_cell(crate::grid::CellCoord::new(key.0, key.1, key.2));

        let col = self
            .columns
            .entry(Self::column_key(key))
            .or_insert_with(|| ColumnStack::new(path, self.dimension, tier));
        col.push(id, &data);
        self.vectors.insert(id, data);
        Ok(())
    }

    fn remove(&mut self, id: VectorId) -> Result<()> {
        if self.vectors.remove(&id).is_none() {
            return Err(TopolseaError::NotFound(id.to_string()));
        }
        for col in self.columns.values_mut() {
            col.remove_id(id);
        }
        self.columns.retain(|_, col| !col.is_empty());
        Ok(())
    }

    fn get_vector(&self, id: VectorId) -> Result<Vector> {
        self.vectors
            .get(&id)
            .cloned()
            .map(Vector::new)
            .ok_or_else(|| TopolseaError::NotFound(id.to_string()))
    }

    fn search(&self, query: &[f32], top_k: usize, ef: usize) -> Result<Vec<SearchHit>> {
        if query.len() != self.dimension {
            return Err(TopolseaError::DimensionMismatch {
                expected: self.dimension,
                got: query.len(),
            });
        }
        if self.vectors.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        let ef = ef.max(top_k).max(self.config.ef_search);
        let columns = self.column_list();
        let mut stats = SearchStats::default();
        let mut searcher = RevertBeamSearch::new(
            &self.grid,
            &columns,
            &self.vectors,
            self.metric,
            self.dimension,
            &mut stats,
        );
        let results = searcher.run(query, top_k, ef);

        self.revert_count
            .fetch_add(stats.revert_count, Ordering::Relaxed);
        self.columns_scanned
            .fetch_add(stats.columns_scanned, Ordering::Relaxed);
        self.query_count.fetch_add(1, Ordering::Relaxed);

        Ok(results
            .into_iter()
            .map(|(id, dist)| SearchHit::new(id, dist))
            .collect())
    }

    fn contains(&self, id: VectorId) -> bool {
        self.vectors.contains_key(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dv_index_api::VectorIndex;

    #[test]
    fn insert_and_search() {
        let mut idx = ZColumnIndex::new(4, DistanceMetric::L2, ZColumnConfig::default());
        idx.insert(VectorId(1), Vector::new(vec![1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        idx.insert(VectorId(2), Vector::new(vec![0.9, 0.1, 0.0, 0.0]))
            .unwrap();
        idx.insert(VectorId(3), Vector::new(vec![0.0, 1.0, 0.0, 0.0]))
            .unwrap();

        let hits = idx.search(&[1.0, 0.0, 0.0, 0.0], 2, 64).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, VectorId(1));
    }

    #[test]
    fn remove_vector() {
        let mut idx = ZColumnIndex::new(2, DistanceMetric::L2, ZColumnConfig::default());
        idx.insert(VectorId(1), Vector::new(vec![0.5, 0.5]))
            .unwrap();
        idx.remove(VectorId(1)).unwrap();
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn roundtrip_bytes() {
        let mut idx = ZColumnIndex::new(3, DistanceMetric::Cosine, ZColumnConfig::default());
        idx.insert(VectorId(1), Vector::new(vec![1.0, 0.0, 0.0]))
            .unwrap();
        let bytes = idx.to_bytes().unwrap();
        let loaded = ZColumnIndex::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.len(), 1);
    }
}
