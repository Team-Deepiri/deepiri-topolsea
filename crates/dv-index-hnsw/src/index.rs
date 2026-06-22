use crate::graph::{connect, random_level, NodeGraph};
use dv_index_api::VectorIndex;
use dv_metrics::distance;
use dv_topk::{Candidate, TopKHeap};
use dv_types::{DistanceMetric, HnswConfig, Result, SearchHit, TopolseaError, Vector, VectorId};
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Serialize, Deserialize)]
pub struct HnswIndex {
    dimension: usize,
    metric: DistanceMetric,
    config: HnswConfig,
    vectors: HashMap<VectorId, Vec<f32>>,
    graphs: HashMap<VectorId, NodeGraph>,
    entry_point: Option<VectorId>,
    max_layer: i32,
    #[serde(skip, default = "default_rng")]
    rng: StdRng,
}

fn default_rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

impl Clone for HnswIndex {
    fn clone(&self) -> Self {
        Self {
            dimension: self.dimension,
            metric: self.metric,
            config: self.config.clone(),
            vectors: self.vectors.clone(),
            graphs: self.graphs.clone(),
            entry_point: self.entry_point,
            max_layer: self.max_layer,
            rng: StdRng::seed_from_u64(self.config.seed),
        }
    }
}

impl HnswIndex {
    pub fn new(dimension: usize, metric: DistanceMetric, config: HnswConfig) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        Self {
            dimension,
            metric,
            config,
            vectors: HashMap::new(),
            graphs: HashMap::new(),
            entry_point: None,
            max_layer: -1,
            rng,
        }
    }

    fn dist_query(&self, query: &[f32], id: VectorId) -> f32 {
        distance(self.metric, query, &self.vectors[&id])
    }

    fn search_layer(
        &self,
        query: &[f32],
        entry: VectorId,
        ef: usize,
        layer: i32,
    ) -> Vec<(VectorId, f32)> {
        let mut visited = HashSet::new();
        let mut candidates = VecDeque::new();
        let mut results = TopKHeap::new(ef.max(1));

        let entry_dist = self.dist_query(query, entry);
        visited.insert(entry);
        candidates.push_back((entry, entry_dist));
        results.push(Candidate {
            id: entry,
            distance: entry_dist,
        });

        while let Some((current, cur_dist)) = candidates.pop_front() {
            if let Some(worst) = results.farthest_distance() {
                if cur_dist > worst && results.len() >= ef {
                    continue;
                }
            }

            let neighbors = self
                .graphs
                .get(&current)
                .and_then(|g| g.layers.get(layer as usize))
                .cloned()
                .unwrap_or_default();

            for neighbor in neighbors {
                if visited.insert(neighbor) {
                    let d = self.dist_query(query, neighbor);
                    if results.len() < ef {
                        candidates.push_back((neighbor, d));
                        results.push(Candidate {
                            id: neighbor,
                            distance: d,
                        });
                    } else if let Some(worst) = results.farthest_distance() {
                        if d < worst {
                            candidates.push_back((neighbor, d));
                            results.push(Candidate {
                                id: neighbor,
                                distance: d,
                            });
                        }
                    }
                }
            }
        }

        results
            .into_sorted_vec()
            .into_iter()
            .map(|c| (c.id, c.distance))
            .collect()
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(TopolseaError::Serde)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut idx: Self = serde_json::from_slice(bytes).map_err(TopolseaError::Serde)?;
        idx.rng = StdRng::seed_from_u64(idx.config.seed);
        Ok(idx)
    }

    pub fn ids(&self) -> impl Iterator<Item = VectorId> + '_ {
        self.vectors.keys().copied()
    }
}

impl VectorIndex for HnswIndex {
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

        self.vectors.insert(id, vector.data);
        self.graphs.entry(id).or_default();

        if self.entry_point.is_none() {
            self.entry_point = Some(id);
            self.max_layer = 0;
            return Ok(());
        }

        let level = random_level(&mut self.rng, self.config.m);
        let entry = self.entry_point.unwrap();

        // Greedy search from top layer down to level+1
        let mut current = entry;
        for l in (level + 1..=self.max_layer as usize).rev() {
            let found = self.search_layer(&self.vectors[&id], current, 1, l as i32);
            if let Some((best, _)) = found.first() {
                current = *best;
            }
        }

        // Insert and connect at each layer from min(level, max_layer) down to 0
        let max_l = level.min(self.max_layer as usize);
        for l in (0..=max_l).rev() {
            let ef = self.config.ef_construction;
            let mut candidates = self.search_layer(&self.vectors[&id], current, ef, l as i32);
            let m = if l == 0 {
                self.config.m_max0
            } else {
                self.config.m
            };
            let neighbors = crate::graph::select_neighbors(&mut candidates, m);
            for &n in &neighbors {
                connect(&mut self.graphs, &self.vectors, self.metric, id, n, l, m);
            }
            if let Some(&first) = neighbors.first() {
                current = first;
            }
        }

        // New layers above current max
        if level as i32 > self.max_layer {
            for l in (self.max_layer as usize + 1)..=level {
                self.graphs.entry(id).or_default().ensure_layer(l);
                if let Some(ep) = self.entry_point {
                    let max_deg = if l == 0 {
                        self.config.m_max0
                    } else {
                        self.config.m
                    };
                    connect(
                        &mut self.graphs,
                        &self.vectors,
                        self.metric,
                        id,
                        ep,
                        l,
                        max_deg,
                    );
                }
            }
            self.entry_point = Some(id);
            self.max_layer = level as i32;
        }

        Ok(())
    }

    fn remove(&mut self, id: VectorId) -> Result<()> {
        if self.vectors.remove(&id).is_none() {
            return Err(TopolseaError::NotFound(id.to_string()));
        }
        self.graphs.remove(&id);
        // Remove back-links (O(n) — acceptable for MVP)
        for g in self.graphs.values_mut() {
            for layer in &mut g.layers {
                layer.retain(|&n| n != id);
            }
        }
        if self.entry_point == Some(id) {
            self.entry_point = self.vectors.keys().next().copied();
            self.max_layer = self
                .entry_point
                .and_then(|ep| self.graphs.get(&ep).map(|g| g.max_layer()))
                .unwrap_or(-1);
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

    fn search(&self, query: &[f32], top_k: usize, ef: usize) -> Result<Vec<SearchHit>> {
        if query.len() != self.dimension {
            return Err(TopolseaError::DimensionMismatch {
                expected: self.dimension,
                got: query.len(),
            });
        }
        if self.entry_point.is_none() || top_k == 0 {
            return Ok(Vec::new());
        }

        let ef = ef.max(top_k).max(self.config.ef_search);
        let mut current = self.entry_point.unwrap();

        for l in (1..=self.max_layer).rev() {
            let found = self.search_layer(query, current, 1, l);
            if let Some((best, _)) = found.first() {
                current = *best;
            }
        }

        let mut candidates = self.search_layer(query, current, ef, 0);
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(top_k);

        Ok(candidates
            .into_iter()
            .map(|(id, dist)| SearchHit::new(id, dist))
            .collect())
    }

    fn contains(&self, id: VectorId) -> bool {
        self.vectors.contains_key(&id)
    }
}
