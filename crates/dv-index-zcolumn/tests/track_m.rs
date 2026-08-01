//! Track M: touch budget, conditional fallback, height balance.

use dv_index_api::VectorIndex;
use dv_index_zcolumn::ZColumnIndex;
use dv_types::{DistanceMetric, Vector, VectorId, ZColumnConfig};

fn tiny_index(n: usize) -> ZColumnIndex {
    let cfg = ZColumnConfig {
        ef_search: 16,
        fallback_beam_radius: 0,
        max_fallback_rings: 0,
        max_fallback_columns: 0,
        conditional_fallback: true,
        use_centroid_graph: true,
        touch_budget: Some(8),
        graph_degree: 4,
        ..ZColumnConfig::default()
    };
    let mut idx = ZColumnIndex::new(4, DistanceMetric::L2, cfg);
    for i in 0..n {
        let mut v = vec![0.0f32; 4];
        v[i % 4] = 1.0 + (i as f32) * 0.01;
        idx.insert(VectorId(i as u64), Vector::new(v)).unwrap();
    }
    idx
}

#[test]
fn touch_budget_caps_candidate_pool() {
    let idx = tiny_index(64);
    let q = vec![1.0, 0.0, 0.0, 0.0];
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
fn pure_beam_does_not_mark_fallback() {
    let idx = tiny_index(32);
    let q = vec![0.0, 1.0, 0.0, 0.0];
    let (_hits, explain) = idx.search_with_explain(&q, 3, 8).unwrap();
    assert!(!explain.used_fallback_scan);
}

#[test]
fn rebalance_reports_height_stats() {
    let mut idx = tiny_index(40);
    idx.rebalance();
    let (n, mean, max, ratio) = idx.height_balance();
    assert!(n > 0);
    assert!(mean > 0.0);
    assert!(max >= 1);
    assert!(ratio >= 1.0);
}
