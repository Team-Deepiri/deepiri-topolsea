//! Neighbor graph over nonempty column centroids (Track M — M-graph).

use crate::column::ColumnStack;
use dv_metrics::distance;
use dv_types::DistanceMetric;
use std::collections::HashMap;

/// Undirected kNN graph over column centroids.
#[derive(Debug, Clone, Default)]
pub struct CentroidGraph {
    /// column_key → (neighbor_key, centroid_distance)
    neighbors: HashMap<String, Vec<(String, f32)>>,
}

impl CentroidGraph {
    pub fn build(
        columns: &HashMap<String, ColumnStack>,
        metric: DistanceMetric,
        degree: usize,
    ) -> Self {
        let degree = degree.max(1);
        let nodes: Vec<(String, Vec<f32>)> = columns
            .iter()
            .filter_map(|(k, c)| {
                if c.is_empty() || c.centroid.is_empty() {
                    None
                } else {
                    Some((k.clone(), c.centroid.clone()))
                }
            })
            .collect();

        let mut neighbors: HashMap<String, Vec<(String, f32)>> = HashMap::new();
        for (i, (ki, ci)) in nodes.iter().enumerate() {
            let mut dists: Vec<(f32, String)> = nodes
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, (kj, cj))| (distance(metric, ci, cj), kj.clone()))
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            dists.truncate(degree);
            neighbors.insert(ki.clone(), dists.into_iter().map(|(d, k)| (k, d)).collect());
        }
        Self { neighbors }
    }

    pub fn is_empty(&self) -> bool {
        self.neighbors.is_empty()
    }

    pub fn neighbors_of(&self, key: &str) -> &[(String, f32)] {
        self.neighbors.get(key).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Seed the beam with the `b` columns whose centroids are closest to `query`.
    pub fn nearest_seeds(
        &self,
        columns: &HashMap<String, ColumnStack>,
        metric: DistanceMetric,
        query: &[f32],
        b: usize,
    ) -> Vec<(String, f32)> {
        let mut ranked: Vec<(String, f32)> = columns
            .iter()
            .filter_map(|(k, c)| {
                if c.is_empty() || c.centroid.is_empty() {
                    return None;
                }
                Some((k.clone(), distance(metric, query, &c.centroid)))
            })
            .collect();
        ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(b.max(1));
        ranked
    }
}
