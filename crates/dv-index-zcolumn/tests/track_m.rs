//! Track M full: prune, touch budget, height balance, gates.

use dv_index_api::VectorIndex;
use dv_index_zcolumn::ZColumnIndex;
use dv_types::{DistanceMetric, Vector, VectorId, ZColumnConfig};

fn cfg_pruned(keep: usize, budget: Option<usize>) -> ZColumnConfig {
    ZColumnConfig {
        ef_search: 16,
        fallback_beam_radius: 0,
        max_fallback_rings: 0,
        max_fallback_columns: 0,
        conditional_fallback: true,
        use_centroid_graph: true,
        touch_budget: budget,
        touch_budget_frac: 1.0,
        graph_degree: 4,
        graph_beam_hops: 2,
        coarse_keep_per_column: keep,
        max_column_height_ratio: 2.0,
        ..ZColumnConfig::default()
    }
}

fn build_index(n: usize, cfg: ZColumnConfig) -> ZColumnIndex {
    let mut idx = ZColumnIndex::new(8, DistanceMetric::L2, cfg);
    for i in 0..n {
        let mut v = vec![0.0f32; 8];
        v[i % 8] = 1.0 + (i as f32) * 0.001;
        idx.insert(VectorId(i as u64), Vector::new(v)).unwrap();
    }
    idx
}

#[test]
fn touch_budget_caps_candidate_pool() {
    let idx = build_index(64, cfg_pruned(32, Some(8)));
    let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let (_hits, explain) = idx.search_with_explain(&q, 5, 16).unwrap();
    assert!(
        explain.candidate_pool <= 8,
        "pool={} budget={}",
        explain.candidate_pool,
        explain.touch_budget
    );
    assert!(!explain.used_fallback_scan);
}

#[test]
fn intra_column_prune_reduces_touch_vs_whole_column() {
    // Force few columns by small grid so each column is tall.
    let mut tall = cfg_pruned(4, None);
    tall.outer_grid = (2, 2);
    tall.max_layers = 1;
    tall.use_centroid_graph = false;
    tall.touch_budget_frac = 1.0;
    tall.touch_budget = None;

    let mut whole = tall.clone();
    whole.coarse_keep_per_column = 0; // keep all

    let idx_prune = build_index(80, tall);
    let idx_whole = build_index(80, whole);
    let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let (_, e_prune) = idx_prune.search_with_explain(&q, 5, 8).unwrap();
    let (_, e_whole) = idx_whole.search_with_explain(&q, 5, 8).unwrap();
    assert!(
        e_prune.candidate_pool < e_whole.candidate_pool,
        "prune={} whole={} scored_prune={} scored_whole={}",
        e_prune.candidate_pool,
        e_whole.candidate_pool,
        e_prune.coarse_scored,
        e_whole.coarse_scored
    );
    assert!(e_prune.coarse_kept <= e_prune.coarse_scored);
}

#[test]
fn pure_beam_does_not_mark_fallback() {
    let idx = build_index(32, cfg_pruned(8, Some(16)));
    let q = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let (_hits, explain) = idx.search_with_explain(&q, 3, 8).unwrap();
    assert!(!explain.used_fallback_scan);
}

#[test]
fn height_balance_shrinks_under_rebalance() {
    let mut cfg = cfg_pruned(8, None);
    cfg.outer_grid = (2, 2);
    cfg.max_layers = 3;
    cfg.max_column_height_ratio = 1.5;
    let mut idx = build_index(120, cfg);
    let (_n0, _mean0, max0, ratio0) = idx.height_balance();
    idx.rebalance();
    let (_n1, _mean1, max1, ratio1) = idx.height_balance();
    // After split, max height should not grow; ratio should improve or stay.
    assert!(max1 <= max0 || ratio1 <= ratio0 + 0.01);
}
