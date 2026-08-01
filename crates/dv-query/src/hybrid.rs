//! Hybrid dense + sparse ranking fusion.

use dv_types::VectorId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Default RRF constant `k` (Cormack et al.).
pub const DEFAULT_RRF_K: f32 = 60.0;

/// Default dense weight for linear fusion (`alpha`).
pub const DEFAULT_DENSE_WEIGHT: f32 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FusionMethod {
    /// Reciprocal Rank Fusion (rank-only; production default for hybrid).
    #[default]
    Rrf,
    /// Score fusion: `alpha * dense_norm + (1 - alpha) * sparse_norm`.
    Linear,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridOptions {
    pub top_k: usize,
    pub ef: usize,
    #[serde(default)]
    pub fusion: FusionMethod,
    /// RRF `k` (ignored for linear).
    #[serde(default)]
    pub rrf_k: Option<f32>,
    /// Dense weight `alpha` for linear fusion in `[0, 1]`.
    #[serde(default)]
    pub dense_weight: Option<f32>,
    /// Candidate pool per channel before fusion (default `top_k * 5`).
    #[serde(default)]
    pub prefetch: Option<usize>,
}

impl HybridOptions {
    pub fn new(top_k: usize, ef: usize) -> Self {
        Self {
            top_k,
            ef,
            fusion: FusionMethod::Rrf,
            rrf_k: None,
            dense_weight: None,
            prefetch: None,
        }
    }

    pub fn prefetch_k(&self) -> usize {
        self.prefetch
            .unwrap_or_else(|| self.top_k.saturating_mul(5).max(self.top_k))
    }
}

/// Fuse multiple ranked lists with Reciprocal Rank Fusion.
///
/// `lists` are ordered best-first. Native scores are ignored (rank-only).
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
    sort_truncate(scores, top_k)
}

/// Weighted linear fusion after min-max normalizing each channel's scores.
///
/// `dense_weight` (`alpha`) in `[0, 1]`; sparse weight is `1 - alpha`.
/// Expects exactly two lists: `[dense, sparse]` (missing ids score as 0 after norm).
pub fn linear_score_fusion(
    dense: &[(VectorId, f32)],
    sparse: &[(VectorId, f32)],
    top_k: usize,
    dense_weight: f32,
) -> Vec<(VectorId, f32)> {
    if top_k == 0 {
        return Vec::new();
    }
    let alpha = dense_weight.clamp(0.0, 1.0);
    let dense_n = min_max_norm(dense);
    let sparse_n = min_max_norm(sparse);
    let mut scores: HashMap<VectorId, f32> = HashMap::new();
    for (id, s) in dense_n {
        *scores.entry(id).or_default() += alpha * s;
    }
    for (id, s) in sparse_n {
        *scores.entry(id).or_default() += (1.0 - alpha) * s;
    }
    sort_truncate(scores, top_k)
}

pub fn fuse(
    dense: Vec<(VectorId, f32)>,
    sparse: Vec<(VectorId, f32)>,
    opts: &HybridOptions,
) -> Vec<(VectorId, f32)> {
    match opts.fusion {
        FusionMethod::Rrf => reciprocal_rank_fusion(
            &[dense, sparse],
            opts.top_k,
            opts.rrf_k.unwrap_or(DEFAULT_RRF_K),
        ),
        FusionMethod::Linear => linear_score_fusion(
            &dense,
            &sparse,
            opts.top_k,
            opts.dense_weight.unwrap_or(DEFAULT_DENSE_WEIGHT),
        ),
    }
}

fn min_max_norm(list: &[(VectorId, f32)]) -> Vec<(VectorId, f32)> {
    if list.is_empty() {
        return Vec::new();
    }
    let mut min_s = f32::MAX;
    let mut max_s = f32::MIN;
    for (_, s) in list {
        min_s = min_s.min(*s);
        max_s = max_s.max(*s);
    }
    let span = (max_s - min_s).max(1e-9);
    list.iter()
        .map(|(id, s)| (*id, (s - min_s) / span))
        .collect()
}

fn sort_truncate(scores: HashMap<VectorId, f32>, top_k: usize) -> Vec<(VectorId, f32)> {
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
        assert!(fused[0].0 == VectorId(1) || fused[0].0 == VectorId(2));
        assert!(fused.iter().any(|(id, _)| *id == VectorId(1)));
        assert!(fused.iter().any(|(id, _)| *id == VectorId(2)));
    }

    #[test]
    fn linear_respects_dense_weight() {
        let dense = vec![(VectorId(1), 10.0), (VectorId(2), 1.0)];
        let sparse = vec![(VectorId(2), 10.0), (VectorId(1), 1.0)];
        let dense_heavy = linear_score_fusion(&dense, &sparse, 1, 1.0);
        assert_eq!(dense_heavy[0].0, VectorId(1));
        let sparse_heavy = linear_score_fusion(&dense, &sparse, 1, 0.0);
        assert_eq!(sparse_heavy[0].0, VectorId(2));
    }
}
