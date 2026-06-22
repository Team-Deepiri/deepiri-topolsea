use crate::scalar::{cosine_distance, dot_product, l2_squared};
use dv_types::DistanceMetric;

/// Compute distance between two vectors for the given metric.
#[inline]
pub fn distance(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        DistanceMetric::L2 => l2_squared(a, b),
        DistanceMetric::Cosine => cosine_distance(a, b),
        DistanceMetric::DotProduct => dot_product(a, b),
    }
}

/// Compute distances from query to all candidates.
pub fn batch_distances(metric: DistanceMetric, query: &[f32], candidates: &[&[f32]]) -> Vec<f32> {
    candidates
        .iter()
        .map(|c| distance(metric, query, c))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_matches_scalar() {
        let q = [1.0, 0.0];
        let c1 = [1.0, 0.0];
        let c2 = [0.0, 1.0];
        let refs = [c1.as_slice(), c2.as_slice()];
        let batch = batch_distances(DistanceMetric::L2, &q, &refs);
        assert_eq!(batch[0], distance(DistanceMetric::L2, &q, &c1));
        assert!(batch[1] > batch[0]);
    }
}
