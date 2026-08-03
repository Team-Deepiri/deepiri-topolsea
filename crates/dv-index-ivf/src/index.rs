use crate::pq::{asymmetric_distance, decode_pq, encode_pq, train_pq_codebooks, PqCodebooks};
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
    /// Full-precision vectors. Cleared after segment flush when `memory_bound`.
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

    pub fn memory_bytes_estimate(&self) -> usize {
        let mut bytes = self.centroids.iter().map(|c| c.len() * 4).sum::<usize>();
        for list in &self.lists {
            for e in list {
                bytes += e.vector.len() * 4;
                bytes += e.codes.as_ref().map(|c| c.len()).unwrap_or(0);
            }
        }
        bytes += self.vectors.values().map(|v| v.len() * 4).sum::<usize>();
        bytes
    }

    pub fn ids(&self) -> impl Iterator<Item = VectorId> + '_ {
        if !self.locate.is_empty() {
            itertools_ids(&self.locate)
        } else {
            Box::new(self.vectors.keys().copied())
        }
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

    /// Drop full-precision RAM copy after vectors are durable in sealed segments.
    pub fn release_raw_if_memory_bound(&mut self) {
        if self.config.memory_bound && self.pq.is_some() {
            self.vectors.clear();
        }
    }

    pub fn is_memory_bound_active(&self) -> bool {
        self.config.memory_bound && self.pq.is_some() && self.vectors.is_empty() && !self.is_empty()
    }

    /// Cap training set size so large in-memory corpora do not explode k-means / PQ cost.
    /// Target ~256 points per list, clamped to `[256, 10_000]`.
    fn training_sample_cap(nlist: usize) -> usize {
        nlist.saturating_mul(256).clamp(256, 10_000)
    }

    fn subsample_for_training(
        vectors: &HashMap<VectorId, Vec<f32>>,
        nlist: usize,
        seed: u64,
    ) -> Vec<Vec<f32>> {
        let cap = Self::training_sample_cap(nlist);
        let mut all: Vec<_> = vectors.values().cloned().collect();
        if all.len() <= cap {
            return all;
        }
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(0x51f1_ca11));
        all.shuffle(&mut rng);
        all.truncate(cap);
        all
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
        let use_pq = self.pq.is_some();
        let entry = IvfEntry {
            id,
            vector: if use_pq { Vec::new() } else { vector.to_vec() },
            codes: self.pq.as_ref().map(|pq| encode_pq(pq, vector)),
        };
        let li = if self.centroids.is_empty() {
            0
        } else {
            nearest_centroid(&self.centroids, vector, self.metric)
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
        if self.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

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
                let Some(dist) = self.entry_distance(query, entry) else {
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

    fn entry_distance(&self, query: &[f32], entry: &IvfEntry) -> Option<f32> {
        if let (Some(pq), Some(codes)) = (&self.pq, &entry.codes) {
            return Some(asymmetric_distance(pq, query, codes));
        }
        if !entry.vector.is_empty() {
            return Some(distance(self.metric, query, &entry.vector));
        }
        self.vectors
            .get(&entry.id)
            .map(|v| distance(self.metric, query, v))
    }

    fn exact_scan(
        &self,
        query: &[f32],
        top_k: usize,
        eligible: Option<&dyn Fn(VectorId) -> bool>,
    ) -> Result<Vec<SearchHit>> {
        let mut heap = TopKHeap::new(top_k);
        if !self.vectors.is_empty() {
            for (&id, vec) in &self.vectors {
                if let Some(pred) = eligible {
                    if !pred(id) {
                        continue;
                    }
                }
                let dist = distance(self.metric, query, vec);
                heap.push(Candidate { id, distance: dist });
            }
        } else {
            for list in &self.lists {
                for entry in list {
                    if let Some(pred) = eligible {
                        if !pred(entry.id) {
                            continue;
                        }
                    }
                    let Some(dist) = self.entry_distance(query, entry) else {
                        continue;
                    };
                    heap.push(Candidate {
                        id: entry.id,
                        distance: dist,
                    });
                }
            }
        }
        Ok(heap
            .into_sorted_vec()
            .into_iter()
            .map(|c| SearchHit::new(c.id, c.distance))
            .collect())
    }

    fn lookup_entry(&self, id: VectorId) -> Option<&IvfEntry> {
        let (li, pos) = *self.locate.get(&id)?;
        self.lists.get(li)?.get(pos)
    }
}

fn itertools_ids(
    locate: &HashMap<VectorId, (usize, usize)>,
) -> Box<dyn Iterator<Item = VectorId> + '_> {
    Box::new(locate.keys().copied())
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
        if !self.locate.is_empty() {
            self.locate.len()
        } else {
            self.vectors.len()
        }
    }

    fn insert(&mut self, id: VectorId, vector: Vector) -> Result<()> {
        vector.validate_dimension(self.dimension)?;
        if self.contains(id) {
            self.remove(id)?;
        }
        self.vectors.insert(id, vector.data.clone());
        if !self.trained && self.vectors.len() >= self.config.nlist.max(1) {
            let sample =
                Self::subsample_for_training(&self.vectors, self.config.nlist, self.config.seed);
            self.train_from(&sample);
        } else {
            self.assign(id, &vector.data);
        }
        Ok(())
    }

    fn remove(&mut self, id: VectorId) -> Result<()> {
        let had_vec = self.vectors.remove(&id).is_some();
        let had_loc = self.locate.contains_key(&id);
        if !had_vec && !had_loc {
            return Err(TopolseaError::NotFound(id.to_string()));
        }
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
        if let Some(v) = self.vectors.get(&id) {
            return Ok(Vector::new(v.clone()));
        }
        // Memory-bound PQ path: reconstruct from codes (lossy).
        if let Some(pq) = &self.pq {
            if let Some(entry) = self.lookup_entry(id) {
                if let Some(codes) = &entry.codes {
                    return Ok(Vector::new(decode_pq(pq, codes)));
                }
            }
        }
        if let Some(entry) = self.lookup_entry(id) {
            if !entry.vector.is_empty() {
                return Ok(Vector::new(entry.vector.clone()));
            }
        }
        Err(TopolseaError::NotFound(id.to_string()))
    }

    fn search(&self, query: &[f32], top_k: usize, ef: usize) -> Result<Vec<SearchHit>> {
        let nprobe = if ef > 0 { Some(ef) } else { None };
        self.search_filtered(query, top_k, nprobe, None)
    }

    fn contains(&self, id: VectorId) -> bool {
        self.vectors.contains_key(&id) || self.locate.contains_key(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_near_neighbor() {
        let cfg = IvfConfig {
            nlist: 4,
            nprobe: 2,
            pq_m: None,
            seed: 7,
            memory_bound: false,
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
    fn memory_bound_pq_drops_raw() {
        let cfg = IvfConfig {
            nlist: 4,
            nprobe: 4,
            pq_m: Some(2),
            seed: 1,
            memory_bound: true,
        };
        let mut idx = IvfIndex::new(4, DistanceMetric::L2, cfg);
        for i in 0..32u64 {
            idx.insert(VectorId(i), Vector::new(vec![i as f32, 1.0, 2.0, 3.0]))
                .unwrap();
        }
        assert!(!idx.vectors.is_empty());
        idx.release_raw_if_memory_bound();
        assert!(idx.vectors.is_empty());
        assert_eq!(idx.len(), 32);
        let hits = idx.search(&[3.0, 1.0, 2.0, 3.0], 5, 0).unwrap();
        assert!(!hits.is_empty());
        let reconstructed = idx.get_vector(VectorId(3)).unwrap();
        assert_eq!(reconstructed.data.len(), 4);
        let before = idx.memory_bytes_estimate();
        assert!(before < 32 * 4 * 4, "PQ codes should beat full f32 storage");
    }
}
