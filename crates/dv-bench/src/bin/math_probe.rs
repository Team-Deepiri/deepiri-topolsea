//! Applied-math probe: measure Z-Column under bounded fallback regimes.
//! Falsifies "fractal search is sublinear" vs "fallback collapses to exhaustive".

use dv_index_api::VectorIndex;
use dv_index_flat::FlatIndex;
use dv_index_hnsw::HnswIndex;
use dv_index_zcolumn::ZColumnIndex;
use dv_types::{DistanceMetric, HnswConfig, Vector, VectorId, ZColumnConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;
use std::time::Instant;

fn unit_vecs(rng: &mut StdRng, n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > f32::EPSILON {
                for x in &mut v {
                    *x /= norm;
                }
            }
            v
        })
        .collect()
}

fn recall_mean(
    index: &dyn VectorIndex,
    queries: &[Vec<f32>],
    ground: &[Vec<VectorId>],
    k: usize,
    ef: usize,
) -> f32 {
    let mut total = 0.0f32;
    for (q, truth) in queries.iter().zip(ground) {
        let got: HashSet<_> = index
            .search(q, k, ef)
            .unwrap()
            .into_iter()
            .map(|h| h.id)
            .collect();
        let truth_set: HashSet<_> = truth.iter().take(k).copied().collect();
        total += got.intersection(&truth_set).count() as f32 / k as f32;
    }
    total / queries.len().max(1) as f32
}

fn main() {
    let n = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000usize);
    let dim = 128usize;
    let k = 10usize;
    let ef = 128usize;
    let nq = 40usize;
    let mut rng = StdRng::seed_from_u64(42);
    let vectors = unit_vecs(&mut rng, n, dim);
    let queries = unit_vecs(&mut rng, nq, dim);

    let mut flat = FlatIndex::new(dim, DistanceMetric::Cosine);
    let mut hnsw = HnswIndex::new(dim, DistanceMetric::Cosine, HnswConfig::default());
    for (i, v) in vectors.iter().enumerate() {
        let id = VectorId(i as u64);
        flat.insert(id, Vector::new(v.clone())).unwrap();
        hnsw.insert(id, Vector::new(v.clone())).unwrap();
    }
    let ground: Vec<Vec<VectorId>> = queries
        .iter()
        .map(|q| {
            flat.search(q, k, 0)
                .unwrap()
                .into_iter()
                .map(|h| h.id)
                .collect()
        })
        .collect();

    let hnsw_recall = recall_mean(&hnsw, &queries, &ground, k, ef);
    let t0 = Instant::now();
    for q in &queries {
        let _ = hnsw.search(q, k, ef).unwrap();
    }
    let hnsw_p50 = {
        let mut lat = Vec::new();
        for q in &queries {
            let t = Instant::now();
            let _ = hnsw.search(q, k, ef).unwrap();
            lat.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        lat[lat.len() / 2]
    };
    let _ = t0;

    println!(
        "n={n} dim={dim} k={k} ef={ef} queries={nq} | HNSW recall@10={hnsw_recall:.4} p50_ms={hnsw_p50:.3}"
    );
    println!(
        "{:>6} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "rings", "fcols", "recall", "vsHNSW", "p50_ms", "×HNSW", "cands", "cols"
    );

    let regimes = [
        (0u16, 0usize),
        (1, 16),
        (2, 32),
        (2, 96),
        (4, 96),
        (8, 96),
        (8, 10_000),
    ];

    for &(rings, fcols) in &regimes {
        let mut cfg = ZColumnConfig::default();
        cfg.max_fallback_rings = rings;
        cfg.fallback_beam_radius = rings.min(2).max(if rings == 0 { 0 } else { 1 });
        cfg.max_fallback_columns = fcols;

        let mut zcol = ZColumnIndex::new(dim, DistanceMetric::Cosine, cfg);
        for (i, v) in vectors.iter().enumerate() {
            zcol.insert(VectorId(i as u64), Vector::new(v.clone()))
                .unwrap();
        }

        let mut lats = Vec::new();
        let mut cands = 0f64;
        let mut cols = 0f64;
        let mut reverts = 0f64;
        for q in &queries {
            let t = Instant::now();
            let (_, ex) = zcol.search_with_explain(q, k, ef).unwrap();
            lats.push(t.elapsed().as_secs_f64() * 1000.0);
            cands += ex.candidate_pool as f64;
            cols += ex.columns_scanned as f64;
            reverts += ex.revert_count as f64;
        }
        lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = lats[lats.len() / 2];
        let recall = recall_mean(&zcol, &queries, &ground, k, ef);
        let vs = recall / hnsw_recall.max(1e-6);
        let lat_x = p50 / hnsw_p50.max(1e-9);
        let nq_f = nq as f64;
        println!(
            "{:>6} {:>6} {:>8.4} {:>8.4} {:>8.3} {:>8.3} {:>8.1} {:>8.1}  revert_avg={:.2}",
            rings,
            fcols,
            recall,
            vs,
            p50,
            lat_x,
            cands / nq_f,
            cols / nq_f,
            reverts / nq_f
        );
    }

    let cfg = ZColumnConfig::default();
    let mut zcol = ZColumnIndex::new(dim, DistanceMetric::Cosine, cfg.clone());
    for (i, v) in vectors.iter().enumerate() {
        zcol.insert(VectorId(i as u64), Vector::new(v.clone()))
            .unwrap();
    }
    let recall = recall_mean(&zcol, &queries, &ground, k, ef);
    let mut lats = Vec::new();
    let mut cands = 0f64;
    let mut reverts = 0f64;
    for q in &queries {
        let t = Instant::now();
        let (_, ex) = zcol.search_with_explain(q, k, ef).unwrap();
        lats.push(t.elapsed().as_secs_f64() * 1000.0);
        cands += ex.candidate_pool as f64;
        reverts += ex.revert_count as f64;
    }
    lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = lats[lats.len() / 2];
    let touch = (cands / nq as f64) / n as f64;
    let recall_ok = (recall / hnsw_recall.max(1e-6)) >= 0.98;
    let lat_ok = (p50 / hnsw_p50) <= 1.5;
    let revert_ok = (reverts / nq as f64) < 0.30; // protocol: <30% of queries — we report avg count
    let touch_ok = touch < 0.5;
    println!();
    println!("DEFAULT CONFIG go/no-go (n={n}):");
    println!(
        "  recall within 2% of HNSW: {} (ratio={:.4})",
        if recall_ok { "GO" } else { "NO-GO" },
        recall / hnsw_recall.max(1e-6)
    );
    println!(
        "  p50 ≤ 1.5× HNSW: {} (ratio={:.3})",
        if lat_ok { "GO" } else { "NO-GO" },
        p50 / hnsw_p50
    );
    println!(
        "  corpus touch < 50%: {} (touch={:.3})",
        if touch_ok { "GO" } else { "NO-GO" },
        touch
    );
    println!(
        "  avg revert_count < 0.3: {} (avg={:.2}) — note protocol said <30% of queries; metric differs",
        if revert_ok { "GO" } else { "NO-GO" },
        reverts / nq as f64
    );
}
