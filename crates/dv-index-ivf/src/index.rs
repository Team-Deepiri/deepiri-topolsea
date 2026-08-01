use crate::pq::{asymmetric_distance, encode_pq, train_pq_codebooks, PqCodebooks};
use dv_index_api::VectorIndex;
use dv_metrics::distance;
use dv_topk::{Candidate, TopKHeap};
use dv_types::{DistanceMetric, IvfConfig, Result, SearchHit, TopolseaError, Vector, VectorId};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IvfEntry {
    id: VectorId,
    /// Full vector when PQ disabled; empty when codes are used.
    vector: Vec<f32>,
    codes: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvfIndex {
    dimension: usize,
    metric: DistanceMetric,
    config: IvfConfig,
    centroids: Vec<Vec<f32>>,
    lists: Vec<Vec<IvfEntry>>,
    /// id → (list_idx, pos) for O(1) remove; rebuilt on deserialize if needed
    #[serde(skip)]
    locate: HashMap<VectorId, (usize, usize)>,
    vectors: HashMap<VectorId, Vec<f32>>,
    pq: Option<PqCodebooks>,
    trained: bool,
}

impl IvfIndex {
    pub fn new(dimension: usize, metric: DistanceMetric, config: IvfConfig) -> Self {
        let nlist = config.nlist.max(1);
        Self {
            dimension,
            metric,
            config: IvfConfig { nlist, ..config },
            centroids: Vec::new(),
            lists: vec![Vec::new(); nlist],
            locate: HashMap::new(),
            vectors: HashMap::new(),
            pq: None,
            trained: false,
        }
    }

    pub fn config(&self) -> &IvfConfig {
        &self.config
    }

    pub fn ids(&self) -> impl Iterator<Item = VectorId> + '_ {
        self.vectors.keys().copied()
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(TopolseaError::Serde)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut idx: Self = serde_json::from_slice(bytes).map_err(TopolseaError::Serde)?;
        idx.rebuild_locate();
        Ok(idx)
    }

    fn rebuild_locate(&mut self) {
        self.locate.clear();
        for (li, list) in self.lists.iter().enumerate() {
            for (pos, e) in list.iter().enumerate() {
                self.locate.insert(e.id, (li, pos));
            }
        }
    }

    fn train_from(&mut self, sample: &[Vec<f32>]) {
        let nlist = self.config.nlist.max(1);
        let mut rng = StdRng::seed_from_u64(self.config.seed);
        let mut centroids = Vec::with_capacity(nlist);
        let mut picks: Vec<_> = sample.to_vec();
        picks.shuffle(&mut rng);
        for i in 0..nlist {
            if picks.is_empty() {
                centroids.push(vec![0.0; self.dimension]);
            } else {
                centroids.push(picks[i % picks.len()].clone());
            }
        }
        // Lloyd iterations
        for _ in 0..8 {
            let mut sums = vec![vec![0.0f32; self.dimension]; nlist];
            let mut counts = vec![0u32; nlist];
            for v in sample {
                let c = nearest_centroid(&centroids, v, self.metric);
                for (a, b) in sums[c].iter_mut().zip(v.iter()) {
                    *a += *b;
                }
                counts[c] += 1;
            }
            for i in 0..nlist {
                if counts[i] > 0 {
                    for x in &mut sums[i] {
                        *x /= counts[i] as f32;
                    }
                    centroids[i] = sums[i].clone();
                }
            }
        }
        self.centroids = centroids;

        if let Some(m) = self.config.pq_m {
            if m > 0 && self.dimension.is_multiple_of(m) {
                self.pq = Some(train_pq_codebooks(
                    sample,
                    self.dimension,
                    m,
                    self.config.seed,
                ));
            }
        }

        // Reassign all vectors into lists.
        self.lists = vec![Vec::new(); nlist];
        self.locate.clear();
        let ids: Vec<_> = self.vectors.keys().copied().collect();
        for id in ids {
            let vec = self.vectors.get(&id).cloned().unwrap();
            self.assign(id, &vec);
        }
        self.trained = true;
    }

    fn assign(&mut self, id: VectorId, vector: &[f32]) {
        if self.centroids.is_empty() {
            // Park in list 0 until trained.
            let entry = IvfEntry {
                id,
                vector: if self.config.pq_m.is_some() {
                    Vec::new()
                } else {
                    vector.to_vec()
                },
                codes: self.pq.as_ref().map(|pq| encode_pq(pq, vector)),
            };
            let pos = self.lists[0].len();
            self.lists[0].push(entry);
            self.locate.insert(id, (0, pos));
            return;
        }
        let li = nearest_centroid(&self.centroids, vector, self.metric);
        let entry = IvfEntry {
            id,
            vector: if self.pq.is_some() {
                Vec::new()
            } else {
                vector.to_vec()
            },
            codes: self.pq.as_ref().map(|pq| encode_pq(pq, vector)),
        };
        let pos = self.lists[li].len();
        self.lists[li].push(entry);
        self.locate.insert(id, (li, pos));
    }

    pub fn search_filtered(
        &self,
        query: &[f32],
        top_k: usize,
        nprobe: Option<usize>,
        eligible: Option<&dyn Fn(VectorId) -> bool>,
    ) -> Result<Vec<SearchHit>> {
        if query.len() != self.dimension {
            return Err(TopolseaError::DimensionMismatch {
                expected: self.dimension,
                got: query.len(),
            });
        }
        if self.vectors.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        // Untrained: exact scan (small corpora / before first train).
        if !self.trained || self.centroids.is_empty() {
            return self.exact_scan(query, top_k, eligible);
        }

        let nprobe = nprobe
            .unwrap_or(self.config.nprobe)
            .max(1)
            .min(self.centroids.len());
        let mut centroid_order: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, distance(self.metric, query, c)))
            .collect();
        centroid_order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut heap = TopKHeap::new(top_k);
        for &(li, _) in centroid_order.iter().take(nprobe) {
            for entry in &self.lists[li] {
                if let Some(pred) = eligible {
                    if !pred(entry.id) {
                        continue;
                    }
                }
                let dist = if let (Some(pq), Some(codes)) = (&self.pq, &entry.codes) {
                    asymmetric_distance(pq, query, codes)
                } else if !entry.vector.is_empty() {
                    distance(self.metric, query, &entry.vector)
                } else if let Some(v) = self.vectors.get(&entry.id) {
                    distance(self.metric, query, v)
                } else {
                    continue;
                };
                heap.push(Candidate {
                    id: entry.id,
                    distance: dist,
                });
            }
        }

        Ok(heap
            .into_sorted_vec()
            .into_iter()
            .map(|c| SearchHit::new(c.id, c.distance))
            .collect())
    }

    fn exact_scan(
        &self,
        query: &[f32],
        top_k: usize,
        eligible: Option<&dyn Fn(VectorId) -> bool>,
    ) -> Result<Vec<SearchHit>> {
        let mut heap = TopKHeap::new(top_k);
        for (&id, vec) in &self.vectors {
            if let Some(pred) = eligible {
                if !pred(id) {
                    continue;
                }
            }
            let dist = distance(self.metric, query, vec);
            heap.push(Candidate { id, distance: dist });
        }
        Ok(heap
            .into_sorted_vec()
            .into_iter()
            .map(|c| SearchHit::new(c.id, c.distance))
            .collect())
    }
}

fn nearest_centroid(centroids: &[Vec<f32>], vector: &[f32], metric: DistanceMetric) -> usize {
    let mut best = 0usize;
    let mut best_d = f32::MAX;
    for (i, c) in centroids.iter().enumerate() {
        let d = distance(metric, vector, c);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

impl VectorIndex for IvfIndex {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn insert(&mut self, id: VectorId, vector: Vector) -> Result<()> {
        vector.validate_dimension(self.dimension)?;
        if self.vectors.contains_key(&id) {
            self.remove(id)?;
        }
        self.vectors.insert(id, vector.data.clone());
        // Train once we have enough data relative to nlist.
        if !self.trained && self.vectors.len() >= self.config.nlist.max(1) {
            let all: Vec<_> = self.vectors.values().cloned().collect();
            self.train_from(&all);
        } else {
            self.assign(id, &vector.data);
        }
        Ok(())
    }

    fn remove(&mut self, id: VectorId) -> Result<()> {
        self.vectors
            .remove(&id)
            .ok_or_else(|| TopolseaError::NotFound(id.to_string()))?;
        if let Some((li, pos)) = self.locate.remove(&id) {
            if li < self.lists.len() && pos < self.lists[li].len() {
                self.lists[li].swap_remove(pos);
                if pos < self.lists[li].len() {
                    let moved = self.lists[li][pos].id;
                    self.locate.insert(moved, (li, pos));
                }
            }
        }
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
        self.search_filtered(query, top_k, None, None)
    }

    fn contains(&self, id: VectorId) -> bool {
        self.vectors.contains_key(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dv_types::IvfConfig;

    #[test]
    fn finds_near_neighbor() {
        let cfg = IvfConfig {
            nlist: 4,
            nprobe: 2,
            pq_m: None,
            seed: 7,
        };
        let mut idx = IvfIndex::new(4, DistanceMetric::L2, cfg);
        for i in 0..40u64 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            idx.insert(VectorId(i), Vector::new(v)).unwrap();
        }
        let hits = idx.search(&[5.0, 0.0, 0.0, 0.0], 3, 0).unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].id, VectorId(5));
    }

    #[test]
    fn pq_path_runs() {
        let cfg = IvfConfig {
            nlist: 4,
            nprobe: 4,
            pq_m: Some(2),
            seed: 1,
        };
        let mut idx = IvfIndex::new(4, DistanceMetric::L2, cfg);
        for i in 0..32u64 {
            idx.insert(VectorId(i), Vector::new(vec![i as f32, 1.0, 2.0, 3.0]))
                .unwrap();
        }
        let hits = idx.search(&[3.0, 1.0, 2.0, 3.0], 5, 0).unwrap();
        assert!(!hits.is_empty());
    }
}
