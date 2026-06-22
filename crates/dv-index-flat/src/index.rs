use dv_index_api::VectorIndex;
use dv_metrics::distance;
use dv_topk::{Candidate, TopKHeap};
use dv_types::{DistanceMetric, Result, SearchHit, TopolseaError, Vector, VectorId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatIndex {
    dimension: usize,
    metric: DistanceMetric,
    vectors: HashMap<VectorId, Vec<f32>>,
}

impl FlatIndex {
    pub fn new(dimension: usize, metric: DistanceMetric) -> Self {
        Self {
            dimension,
            metric,
            vectors: HashMap::new(),
        }
    }

    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    pub fn ids(&self) -> impl Iterator<Item = VectorId> + '_ {
        self.vectors.keys().copied()
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(TopolseaError::Serde)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).map_err(TopolseaError::Serde)
    }
}

impl VectorIndex for FlatIndex {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn insert(&mut self, id: VectorId, vector: Vector) -> Result<()> {
        vector.validate_dimension(self.dimension)?;
        self.vectors.insert(id, vector.data);
        Ok(())
    }

    fn remove(&mut self, id: VectorId) -> Result<()> {
        self.vectors
            .remove(&id)
            .ok_or_else(|| TopolseaError::NotFound(id.to_string()))?;
        Ok(())
    }

    fn get_vector(&self, id: VectorId) -> Result<Vector> {
        self.vectors
            .get(&id)
            .cloned()
            .map(Vector::new)
            .ok_or_else(|| TopolseaError::NotFound(id.to_string()))
    }

    fn search(&self, query: &[f32], top_k: usize, _ef: usize) -> Result<Vec<SearchHit>> {
        if query.len() != self.dimension {
            return Err(TopolseaError::DimensionMismatch {
                expected: self.dimension,
                got: query.len(),
            });
        }
        if self.vectors.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        let mut heap = TopKHeap::new(top_k);
        for (&id, vec) in &self.vectors {
            let dist = distance(self.metric, query, vec);
            heap.push(Candidate { id, distance: dist });
        }

        Ok(heap
            .into_sorted_vec()
            .into_iter()
            .map(|c| SearchHit::new(c.id, c.distance))
            .collect())
    }

    fn contains(&self, id: VectorId) -> bool {
        self.vectors.contains_key(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_nearest_neighbor() {
        let mut idx = FlatIndex::new(2, DistanceMetric::L2);
        idx.insert(VectorId(0), Vector::new(vec![0.0, 0.0]))
            .unwrap();
        idx.insert(VectorId(1), Vector::new(vec![1.0, 0.0]))
            .unwrap();
        idx.insert(VectorId(2), Vector::new(vec![5.0, 5.0]))
            .unwrap();

        let hits = idx.search(&[0.9, 0.0], 1, 0).unwrap();
        assert_eq!(hits[0].id.raw(), 1);
    }
}
