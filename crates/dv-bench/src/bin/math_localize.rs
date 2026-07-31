//! Phase-2 applied-math probe: localize GT neighbors in fractal address space,
//! compare centroid-kNN column expand vs an oracle column order, and test whether
//! any B can hit G1∧G2∧G3 (M-graph hypothesis).
//!
//!   cargo run -p dv-bench --release --bin topolsea-math-localize -- --n=10000

use dv_index_api::VectorIndex;
use dv_index_flat::FlatIndex;
use dv_index_hnsw::HnswIndex;
use dv_index_zcolumn::ZColumnIndex;
use dv_metrics::{decode, distance};
use dv_types::{DistanceMetric, HnswConfig, Vector, VectorId, ZColumnConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::{HashMap, HashSet};
use std::env;
use std::time::Instant;

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > f32::EPSILON {
        for x in &mut v {
            *x /= n;
        }
    }
    v
}

fn unit_sphere(rng: &mut StdRng, n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|_| normalize((0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect()))
        .collect()
}

fn clusters(rng: &mut StdRng, n: usize, dim: usize, n_clusters: usize, sigma: f32) -> Vec<Vec<f32>> {
    let centers: Vec<_> = (0..n_clusters)
        .map(|_| normalize((0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect()))
        .collect();
    (0..n)
        .map(|i| {
            let c = &centers[i % n_clusters];
            normalize(
                c.iter()
                    .map(|&x| x + rng.gen_range(-sigma..sigma))
                    .collect(),
            )
        })
        .collect()
}

fn recall_fraction(got: &HashSet<VectorId>, truth: &[VectorId], k: usize) -> f32 {
    let want: HashSet<_> = truth.iter().copied().take(k).collect();
    if want.is_empty() {
        return 0.0;
    }
    got.intersection(&want).count() as f32 / want.len() as f32
}

fn percentile_sorted(sorted: &[usize], p: f32) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f32 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Scan the B columns whose centroids are nearest the query; FP32 rerank via decode.
fn centroid_knn_search(
    zcol: &ZColumnIndex,
    query: &[f32],
    top_k: usize,
    num_cols: usize,
    metric: DistanceMetric,
) -> (Vec<VectorId>, usize, usize) {
    let mut ranked: Vec<(f32, &String, &dv_index_zcolumn::ColumnStack)> = zcol
        .columns()
        .iter()
        .filter(|(_, c)| !c.centroid.is_empty() && !c.ids.is_empty())
        .map(|(key, c)| (distance(metric, query, &c.centroid), key, c))
        .collect();
    ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(num_cols.max(1));

    let mut scored: Vec<(VectorId, f32)> = Vec::new();
    let mut touched = 0usize;
    let cols = ranked.len();
    for (_, _, col) in &ranked {
        for (i, &id) in col.ids.iter().enumerate() {
            let bytes = &col.quantized[i];
            let vec = decode(bytes, col.quant_tier, query.len());
            scored.push((id, distance(metric, query, &vec)));
            touched += 1;
        }
    }
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);
    (
        scored.into_iter().map(|(id, _)| id).collect(),
        touched,
        cols,
    )
}

/// Oracle: scan the B columns that actually contain the most GT mass (cheating).
/// Upper bound on any column-expand operator that only picks columns.
fn oracle_gt_columns(
    truth: &[VectorId],
    k: usize,
    id_to_col: &HashMap<VectorId, String>,
    zcol: &ZColumnIndex,
    query: &[f32],
    top_k: usize,
    num_cols: usize,
    metric: DistanceMetric,
) -> (Vec<VectorId>, usize, usize) {
    let mut freq: HashMap<String, usize> = HashMap::new();
    for id in truth.iter().take(k) {
        if let Some(ck) = id_to_col.get(id) {
            *freq.entry(ck.clone()).or_default() += 1;
        }
    }
    let mut keys: Vec<(usize, String)> = freq.into_iter().map(|(k, n)| (n, k)).collect();
    keys.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    // Pad with nearest-centroid columns if GT spans fewer than num_cols (fairer τ).
    if keys.len() < num_cols {
        let mut ranked: Vec<(f32, String)> = zcol
            .columns()
            .iter()
            .filter(|(_, c)| !c.ids.is_empty())
            .map(|(key, c)| (distance(metric, query, &c.centroid), key.clone()))
            .collect();
        ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let have: HashSet<_> = keys.iter().map(|(_, k)| k.clone()).collect();
        for (_, key) in ranked {
            if keys.len() >= num_cols {
                break;
            }
            if !have.contains(&key) {
                keys.push((0, key));
            }
        }
    }
    keys.truncate(num_cols.max(1));

    let mut scored: Vec<(VectorId, f32)> = Vec::new();
    let mut touched = 0usize;
    let cols = keys.len();
    for (_, key) in &keys {
        let Some(col) = zcol.columns().get(key) else {
            continue;
        };
        for (i, &id) in col.ids.iter().enumerate() {
            let bytes = &col.quantized[i];
            let vec = decode(bytes, col.quant_tier, query.len());
            scored.push((id, distance(metric, query, &vec)));
            touched += 1;
        }
    }
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);
    (
        scored.into_iter().map(|(id, _)| id).collect(),
        touched,
        cols,
    )
}

fn g123(recall: f32, latx: f64, touch: f64) -> (bool, bool, bool) {
    (recall >= 0.98, latx <= 1.5, touch < 0.5)
}

fn flag(b: bool) -> char {
    if b {
        'Y'
    } else {
        'n'
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let n: usize = args
        .iter()
        .find_map(|a| a.strip_prefix("--n=").and_then(|s| s.parse().ok()))
        .unwrap_or(10_000);
    let dim = 128usize;
    let k = 10usize;
    let nq = 40usize;
    let metric = DistanceMetric::Cosine;

    println!("=== math_localize n={n} dim={dim} k={k} queries={nq} ===");
    println!("G1=recall≥0.98 vs flat  G2=p50≤1.5×HNSW  G3=τ=V_touch/N<0.5\n");

    for (dname, make) in [
        (
            "sphere",
            Box::new(|rng: &mut StdRng| unit_sphere(rng, n, dim))
                as Box<dyn Fn(&mut StdRng) -> Vec<Vec<f32>>>,
        ),
        (
            "clusters32",
            Box::new(|rng: &mut StdRng| clusters(rng, n, dim, 32, 0.08)),
        ),
    ] {
        let mut rng = StdRng::seed_from_u64(42 + n as u64);
        let vectors = make(&mut rng);
        // Important: generate a full batch then take nq — do NOT average n queries / nq.
        let queries: Vec<Vec<f32>> = make(&mut rng).into_iter().take(nq).collect();

        let mut flat = FlatIndex::new(dim, metric);
        let mut hnsw = HnswIndex::new(dim, metric, HnswConfig::default());
        let mut zcfg = ZColumnConfig::default();
        zcfg.max_fallback_rings = 0;
        zcfg.fallback_beam_radius = 0;
        zcfg.max_fallback_columns = 0;
        let mut zcol = ZColumnIndex::new(dim, metric, zcfg);

        for (i, v) in vectors.iter().enumerate() {
            let id = VectorId(i as u64);
            flat.insert(id, Vector::new(v.clone())).unwrap();
            hnsw.insert(id, Vector::new(v.clone())).unwrap();
            zcol.insert(id, Vector::new(v.clone())).unwrap();
        }

        let mut id_to_col: HashMap<VectorId, String> = HashMap::new();
        for (key, col) in zcol.columns() {
            for &id in &col.ids {
                id_to_col.insert(id, key.clone());
            }
        }

        let nonempty = zcol.columns().len();
        let phi = n as f32 / nonempty.max(1) as f32;
        println!("--- {dname}: nonempty_cols={nonempty} mean_height φ={phi:.1} ---");

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

        // Absolute recalls vs flat GT
        let mut hnsw_rec = 0.0f32;
        for (q, t) in queries.iter().zip(&ground) {
            let got: HashSet<_> = hnsw
                .search(q, k, 128)
                .unwrap()
                .into_iter()
                .map(|h| h.id)
                .collect();
            hnsw_rec += recall_fraction(&got, t, k);
        }
        hnsw_rec /= queries.len() as f32;

        // --- Localization ---
        let mut in_query_cell = 0usize;
        let mut in_top_b = [0usize; 7];
        let b_vals = [1usize, 2, 4, 8, 16, 32, 64];
        let mut max_cent_rank: Vec<usize> = Vec::new();
        let mut gt_cols_per_query: Vec<usize> = Vec::new();
        let mut gt_total = 0usize;

        for (q, truth) in queries.iter().zip(&ground) {
            let mut ranked: Vec<(f32, String)> = zcol
                .columns()
                .iter()
                .filter(|(_, c)| !c.centroid.is_empty())
                .map(|(key, c)| (distance(metric, q, &c.centroid), key.clone()))
                .collect();
            ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let rank_of: HashMap<String, usize> = ranked
                .iter()
                .enumerate()
                .map(|(i, (_, key))| (key.clone(), i))
                .collect();
            let qcell = ranked.first().map(|(_, key)| key.clone());

            let mut uniq = HashSet::new();
            let mut worst_rank = 0usize;
            for &id in truth.iter().take(k) {
                gt_total += 1;
                let Some(ck) = id_to_col.get(&id) else {
                    continue;
                };
                uniq.insert(ck.clone());
                if qcell.as_ref() == Some(ck) {
                    in_query_cell += 1;
                }
                let r = *rank_of.get(ck).unwrap_or(&usize::MAX);
                if r != usize::MAX {
                    worst_rank = worst_rank.max(r);
                    max_cent_rank.push(r);
                    for (bi, &b) in b_vals.iter().enumerate() {
                        if r < b {
                            in_top_b[bi] += 1;
                        }
                    }
                }
            }
            gt_cols_per_query.push(uniq.len());
            let _ = worst_rank;
        }

        max_cent_rank.sort_unstable();
        let mut gt_cols_sorted = gt_cols_per_query.clone();
        gt_cols_sorted.sort_unstable();

        println!("GT localization (fraction of true top-{k} neighbors):");
        println!(
            "  in nearest-centroid cell: {:.3}",
            in_query_cell as f32 / gt_total.max(1) as f32
        );
        for (bi, &b) in b_vals.iter().enumerate() {
            if b > nonempty {
                break;
            }
            println!(
                "  in top-{b:2} centroid columns: {:.3}",
                in_top_b[bi] as f32 / gt_total.max(1) as f32
            );
        }
        println!(
            "  GT centroid-rank of neighbor's column: p50={} p90={} p95={} p99={} (of {nonempty})",
            percentile_sorted(&max_cent_rank, 0.50),
            percentile_sorted(&max_cent_rank, 0.90),
            percentile_sorted(&max_cent_rank, 0.95),
            percentile_sorted(&max_cent_rank, 0.99),
        );
        println!(
            "  unique columns housing one query's GT@{k}: p50={} p90={} p99={} (oracle lower bound on B)",
            percentile_sorted(&gt_cols_sorted, 0.50),
            percentile_sorted(&gt_cols_sorted, 0.90),
            percentile_sorted(&gt_cols_sorted, 0.99),
        );
        println!("  HNSW recall@10 vs flat: {hnsw_rec:.4}");

        // HNSW latency baseline
        let mut hlat = Vec::new();
        for q in &queries {
            let t0 = Instant::now();
            let _ = hnsw.search(q, k, 128).unwrap();
            hlat.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        hlat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let hnsw_p50 = hlat[hlat.len() / 2];

        println!(
            "{:>10} {:>7} {:>7} {:>8} {:>7} {:>7} {:>6} {:>5} {:>4}",
            "method", "recall", "vsHNSW", "p50_ms", "lat×", "touch", "cands", "cols", "G123"
        );

        // Reference Z regimes
        for &(label, rings, fcols) in &[
            ("Zbeam0", 0u16, 0usize),
            ("Zring1", 1, 16),
            ("Zring2", 2, 32),
        ] {
            let mut cfg = ZColumnConfig::default();
            cfg.max_fallback_rings = rings;
            cfg.fallback_beam_radius = if rings == 0 { 0 } else { 1 };
            cfg.max_fallback_columns = fcols;
            cfg.ef_search = 32;
            let mut zx = ZColumnIndex::new(dim, metric, cfg);
            for (i, v) in vectors.iter().enumerate() {
                zx.insert(VectorId(i as u64), Vector::new(v.clone()))
                    .unwrap();
            }
            let mut rec = 0.0f32;
            let mut touch = 0.0f64;
            let mut lats = Vec::new();
            for (q, t) in queries.iter().zip(&ground) {
                let t0 = Instant::now();
                let (hits, ex) = zx.search_with_explain(q, k, 32).unwrap();
                lats.push(t0.elapsed().as_secs_f64() * 1000.0);
                touch += ex.candidate_pool as f64;
                let got: HashSet<_> = hits.into_iter().map(|h| h.id).collect();
                rec += recall_fraction(&got, t, k);
            }
            rec /= nq as f32;
            let touch_f = (touch / nq as f64) / n as f64;
            lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50 = lats[lats.len() / 2];
            let vs = rec / hnsw_rec.max(1e-6);
            let latx = p50 / hnsw_p50.max(1e-9);
            let (g1, g2, g3) = g123(rec, latx, touch_f);
            println!(
                "{:>10} {:>7.4} {:>7.4} {:>8.3} {:>7.2} {:>7.3} {:>6.0} {:>5} {}{}{}",
                label,
                rec,
                vs,
                p50,
                latx,
                touch_f,
                touch / nq as f64,
                "-",
                flag(g1),
                flag(g2),
                flag(g3),
            );
        }

        // Centroid-kNN expand (online M-graph / IVF-ish)
        for &b in &[1usize, 2, 4, 8, 12, 16, 24, 32] {
            if b > nonempty {
                break;
            }
            let mut rec = 0.0f32;
            let mut touch = 0.0f64;
            let mut cols_scanned = 0.0f64;
            let mut lats = Vec::new();
            for (q, t) in queries.iter().zip(&ground) {
                let t0 = Instant::now();
                let (ids, touched, cols) = centroid_knn_search(&zcol, q, k, b, metric);
                lats.push(t0.elapsed().as_secs_f64() * 1000.0);
                touch += touched as f64;
                cols_scanned += cols as f64;
                let got: HashSet<_> = ids.into_iter().collect();
                rec += recall_fraction(&got, t, k);
            }
            rec /= nq as f32;
            let touch_f = (touch / nq as f64) / n as f64;
            lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50 = lats[lats.len() / 2];
            let vs = rec / hnsw_rec.max(1e-6);
            let latx = p50 / hnsw_p50.max(1e-9);
            let (g1, g2, g3) = g123(rec, latx, touch_f);
            println!(
                "{:>10} {:>7.4} {:>7.4} {:>8.3} {:>7.2} {:>7.3} {:>6.0} {:>5.1} {}{}{}",
                format!("centB{b}"),
                rec,
                vs,
                p50,
                latx,
                touch_f,
                touch / nq as f64,
                cols_scanned / nq as f64,
                flag(g1),
                flag(g2),
                flag(g3),
            );
        }

        // Oracle column expand — upper bound for ANY column picker
        for &b in &[1usize, 2, 4, 8, 12, 16, 24, 32] {
            if b > nonempty {
                break;
            }
            let mut rec = 0.0f32;
            let mut touch = 0.0f64;
            let mut cols_scanned = 0.0f64;
            let mut lats = Vec::new();
            for (q, t) in queries.iter().zip(&ground) {
                let t0 = Instant::now();
                let (ids, touched, cols) =
                    oracle_gt_columns(t, k, &id_to_col, &zcol, q, k, b, metric);
                lats.push(t0.elapsed().as_secs_f64() * 1000.0);
                touch += touched as f64;
                cols_scanned += cols as f64;
                let got: HashSet<_> = ids.into_iter().collect();
                rec += recall_fraction(&got, t, k);
            }
            rec /= nq as f32;
            let touch_f = (touch / nq as f64) / n as f64;
            lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50 = lats[lats.len() / 2];
            let vs = rec / hnsw_rec.max(1e-6);
            let latx = p50 / hnsw_p50.max(1e-9);
            let (g1, g2, g3) = g123(rec, latx, touch_f);
            println!(
                "{:>10} {:>7.4} {:>7.4} {:>8.3} {:>7.2} {:>7.3} {:>6.0} {:>5.1} {}{}{}",
                format!("orclB{b}"),
                rec,
                vs,
                p50,
                latx,
                touch_f,
                touch / nq as f64,
                cols_scanned / nq as f64,
                flag(g1),
                flag(g2),
                flag(g3),
            );
        }
        println!();
    }

    println!(
        "Done.\n\
         Interpretation:\n\
         - centB recall tracks the localization CDF (centroid scoring is consistent).\n\
         - Gap centB → orclB at fixed B = value of a better column graph / scorer (M-graph).\n\
         - If orclB needs B≈6–8 with τ>0.5 for G1: GT columns are tall — G1∧G3 needs M3 intra-column prune and/or M4 height balance, not just M-graph.\n\
         - unique-columns-housing-GT is the information-theoretic floor on B for whole-column expand."
    );
}
