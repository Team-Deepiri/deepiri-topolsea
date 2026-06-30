use dv_types::VectorId;
use rand::rngs::StdRng;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Per-node graph layer adjacency lists.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeGraph {
    pub layers: Vec<Vec<VectorId>>,
}

impl NodeGraph {
    pub fn max_layer(&self) -> i32 {
        self.layers.len() as i32 - 1
    }

    pub fn ensure_layer(&mut self, layer: usize) {
        while self.layers.len() <= layer {
            self.layers.push(Vec::new());
        }
    }
}

/// Random level generator: ml = 1/ln(M), level ~ floor(-ln(U) * ml)
pub fn random_level(rng: &mut StdRng, m: usize) -> usize {
    let ml = 1.0 / (m as f64).ln();
    let u: f64 = rng.gen_range(f64::EPSILON..1.0);
    (-u.ln() * ml).floor() as usize
}

/// Select `m` closest neighbors from candidates using simple heuristic.
pub fn select_neighbors(candidates: &mut [(VectorId, f32)], m: usize) -> Vec<VectorId> {
    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.iter().take(m).map(|(id, _)| *id).collect()
}

/// Bidirectional link with pruning to max degree.
pub fn connect(
    graphs: &mut HashMap<VectorId, NodeGraph>,
    vectors: &HashMap<VectorId, Vec<f32>>,
    metric: dv_types::DistanceMetric,
    from: VectorId,
    to: VectorId,
    layer: usize,
    max_degree: usize,
) {
    let dist = |a: VectorId, b: VectorId| -> f32 {
        dv_metrics::distance(metric, &vectors[&a], &vectors[&b])
    };

    {
        let g = graphs.entry(from).or_default();
        g.ensure_layer(layer);
        let neighbors = &mut g.layers[layer];
        if !neighbors.contains(&to) {
            neighbors.push(to);
        }
        if neighbors.len() > max_degree {
            let mut scored: Vec<(VectorId, f32)> =
                neighbors.iter().map(|&n| (n, dist(from, n))).collect();
            *neighbors = select_neighbors(&mut scored, max_degree);
        }
    }

    {
        let g = graphs.entry(to).or_default();
        g.ensure_layer(layer);
        let neighbors = &mut g.layers[layer];
        if !neighbors.contains(&from) {
            neighbors.push(from);
        }
        if neighbors.len() > max_degree {
            let mut scored: Vec<(VectorId, f32)> =
                neighbors.iter().map(|&n| (n, dist(to, n))).collect();
            *neighbors = select_neighbors(&mut scored, max_degree);
        }
    }
}
