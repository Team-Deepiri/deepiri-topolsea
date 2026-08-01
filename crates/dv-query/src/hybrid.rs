//! Reciprocal Rank Fusion for hybrid dense + sparse ranking.

use dv_types::VectorId;
use std::collections::HashMap;

/// Default RRF constant `k` (Cormack et al.).
pub const DEFAULT_RRF_K: f32 = 60.0;

/// Fuse multiple ranked lists with Reciprocal Rank Fusion.
///
/// `lists` are ordered best-first. Each entry is `(id, optional_native_score)` —
/// native scores are ignored for RRF (rank-only).
pub fn reciprocal_rank_fusion(
    lists: &[Vec<(VectorId, f32)>],
    top_k: usize,
    k: f32,
) -> Vec<(VectorId, f32)> {
    if top_k == 0 || lists.is_empty() {
        return Vec::new();
    }
    let mut scores: HashMap<VectorId, f32> = HashMap::new();
    for list in lists {
        for (rank, (id, _)) in list.iter().enumerate() {
            let contrib = 1.0 / (k + rank as f32 + 1.0);
            *scores.entry(*id).or_default() += contrib;
        }
    }
    let mut ranked: Vec<_> = scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(top_k);
    ranked
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefers_consensus() {
        let dense = vec![(VectorId(1), 0.9), (VectorId(2), 0.8), (VectorId(3), 0.7)];
        let sparse = vec![(VectorId(2), 5.0), (VectorId(1), 4.0), (VectorId(4), 3.0)];
        let fused = reciprocal_rank_fusion(&[dense, sparse], 3, DEFAULT_RRF_K);
        // ids 1 and 2 appear in both lists → highest RRF
        assert!(fused[0].0 == VectorId(1) || fused[0].0 == VectorId(2));
        assert!(fused.iter().any(|(id, _)| *id == VectorId(1)));
        assert!(fused.iter().any(|(id, _)| *id == VectorId(2)));
    }
}
