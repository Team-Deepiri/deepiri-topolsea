use crate::column::ColumnStack;
use crate::compact::CompactionEngine;
use crate::explain::QueryExplain;
use crate::grid::{CellCoord, ColumnPath, FractalGrid};
use crate::projection::RoutingProjection;
use crate::search::{RevertBeamSearch, SearchStats};
use dv_index_api::VectorIndex;
use dv_metrics::distance;
use dv_types::{
    DistanceMetric, QuantTier, Result, SearchHit, TopolseaError, Vector, VectorId, ZColumnConfig,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Serialize, Deserialize)]
pub struct ZColumnIndex {
    dimension: usize,
    metric: DistanceMetric,
    config: ZColumnConfig,
    grid: FractalGrid,
    #[serde(skip)]
    projection: RoutingProjection,
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
            projection: self.projection.clone(),
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
        let projection = RoutingProjection::new(dimension, config.projection_seed);
        Self {
            dimension,
            metric,
            config,
            grid,
            projection,
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
        let (px, py) = self.projection.project(vector);
        let cell = self
            .grid
            .deepest_cell(px, py)
            .unwrap_or(crate::grid::CellCoord::new(0, 0, 0));
        cell.key()
    }

    fn hybrid_rerank(
        &self,
        query: &[f32],
        candidates: Vec<(VectorId, f32)>,
        top_k: usize,
    ) -> Vec<(VectorId, f32)> {
        let mut scored: Vec<(VectorId, f32)> = candidates
            .into_iter()
            .filter_map(|(id, _)| {
                self.vectors
                    .get(&id)
                    .map(|v| (id, distance(self.metric, query, v)))
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    /// Re-insert all vectors from a cold store (disaster recovery).
    pub fn rebuild_from_vectors(&mut self, records: &[(VectorId, Vec<f32>)]) -> Result<()> {
        self.vectors.clear();
        self.columns.clear();
        for &(id, ref vec) in records {
            self.insert(id, Vector::new(vec.clone()))?;
        }
        Ok(())
    }

    pub fn search_with_explain(
        &self,
        query: &[f32],
        top_k: usize,
        ef: usize,
    ) -> Result<(Vec<SearchHit>, QueryExplain)> {
        if query.len() != self.dimension {
            return Err(TopolseaError::DimensionMismatch {
                expected: self.dimension,
                got: query.len(),
            });
        }
        if self.vectors.is_empty() || top_k == 0 {
            return Ok((Vec::new(), QueryExplain::new("empty")));
        }

        let pool = top_k
            .saturating_mul(self.config.hybrid_rerank_pool.max(1))
            .max(top_k);
        let ef = ef.max(pool).max(self.config.ef_search);
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
        let (coarse, mut explain) = searcher.run_with_explain(query, pool, ef);
        let refined = self.hybrid_rerank(query, coarse, top_k);
        if self.config.hybrid_rerank_pool > 1 {
            explain.strategy = "predictive_revert_hybrid_rerank".into();
        }

        self.revert_count
            .fetch_add(stats.revert_count, Ordering::Relaxed);
        self.columns_scanned
            .fetch_add(stats.columns_scanned, Ordering::Relaxed);
        self.query_count.fetch_add(1, Ordering::Relaxed);

        let hits = refined
            .into_iter()
            .map(|(id, dist)| SearchHit::new(id, dist))
            .collect();
        Ok((hits, explain))
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
                    let key = cell.to_string();
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

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(TopolseaError::Serde)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut idx: Self = serde_json::from_slice(bytes).map_err(TopolseaError::Serde)?;
        idx.query_count = AtomicU64::new(0);
        idx.revert_count = AtomicU64::new(0);
        idx.columns_scanned = AtomicU64::new(0);
        idx.compaction = CompactionEngine::new();
        idx.projection = RoutingProjection::new(idx.dimension, idx.config.projection_seed);
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
            .entry(CellCoord::new(key.0, key.1, key.2).to_string())
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
        self.search_with_explain(query, top_k, ef)
            .map(|(hits, _)| hits)
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

    #[test]
    fn from_bytes_rebuilds_projection() {
        let mut idx = ZColumnIndex::new(128, DistanceMetric::Cosine, ZColumnConfig::default());
        idx.insert(VectorId(1), Vector::new(vec![0.1; 128]))
            .unwrap();
        let bytes = idx.to_bytes().unwrap();
        let loaded = ZColumnIndex::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.projection.dimension(), 128);
    }

    #[test]
    fn search_explain_returns_paths() {
        let mut idx = ZColumnIndex::new(8, DistanceMetric::Cosine, ZColumnConfig::default());
        for i in 0..20u64 {
            let v: Vec<f32> = (0..8)
                .map(|d| (i as f32 * 0.1 + d as f32 * 0.01).sin())
                .collect();
            idx.insert(VectorId(i), Vector::new(v)).unwrap();
        }
        let (hits, explain) = idx.search_with_explain(&[0.1; 8], 3, 32).unwrap();
        assert!(!hits.is_empty());
        assert!(!explain.column_paths.is_empty());
    }
}
