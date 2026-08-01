//! Applied-math discovery probe for Z-Column.
#![allow(clippy::field_reassign_with_default, clippy::too_many_arguments)]
//! Empirically maps dimensionless groups (τ, ρ_r, λ, β, φ) and finds Pareto
//! regimes that could pass G1∧G2∧G3 — propelling Track M toward production.

use dv_index_api::VectorIndex;
use dv_index_flat::FlatIndex;
use dv_index_hnsw::HnswIndex;
use dv_index_zcolumn::ZColumnIndex;
use dv_types::{DistanceMetric, HnswConfig, Vector, VectorId, ZColumnConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::collections::HashSet;
use std::env;
use std::time::Instant;

#[derive(Clone, Copy)]
enum Distro {
    UnitSphere,
    GaussianClusters { clusters: usize, sigma: f32 },
}

fn gen_vectors(rng: &mut StdRng, n: usize, dim: usize, distro: Distro) -> Vec<Vec<f32>> {
    match distro {
        Distro::UnitSphere => (0..n)
            .map(|_| normalize((0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect()))
            .collect(),
        Distro::GaussianClusters { clusters, sigma } => {
            let centers: Vec<Vec<f32>> = (0..clusters)
                .map(|_| normalize((0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect()))
                .collect();
            (0..n)
                .map(|i| {
                    let c = &centers[i % clusters];
                    let v: Vec<f32> = c
                        .iter()
                        .map(|&x| x + rng.gen_range(-sigma..sigma))
                        .collect();
                    normalize(v)
                })
                .collect()
        }
    }
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
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

fn p50_ms(index: &dyn VectorIndex, queries: &[Vec<f32>], k: usize, ef: usize) -> f32 {
    let mut lat = Vec::with_capacity(queries.len());
    for q in queries {
        let t = Instant::now();
        let _ = index.search(q, k, ef).unwrap();
        lat.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    lat[lat.len() / 2] as f32
}

#[derive(Serialize, Clone)]
struct Row {
    distro: String,
    n: usize,
    ef: usize,
    rings: u16,
    beam_r: u16,
    fcols: usize,
    recall: f32,
    vs_hnsw: f32,
    p50_ms: f32,
    lat_x: f32,
    touch: f32,
    avg_cands: f32,
    avg_cols: f32,
    revert_avg: f32,
    revert_frac: f32,
    nonempty_cols: usize,
    mean_height: f32,
    g1: bool,
    g2: bool,
    g3: bool,
    score: f32,
}

fn run_zcol(
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    ground: &[Vec<VectorId>],
    dim: usize,
    k: usize,
    ef: usize,
    rings: u16,
    beam_r: u16,
    fcols: usize,
    hnsw_recall: f32,
    hnsw_p50: f32,
    distro_name: &str,
) -> Row {
    let n = vectors.len();
    let mut cfg = ZColumnConfig::default();
    cfg.max_fallback_rings = rings;
    cfg.fallback_beam_radius = beam_r;
    cfg.max_fallback_columns = fcols;
    cfg.ef_search = ef;

    let mut zcol = ZColumnIndex::new(dim, DistanceMetric::Cosine, cfg);
    for (i, v) in vectors.iter().enumerate() {
        zcol.insert(VectorId(i as u64), Vector::new(v.clone()))
            .unwrap();
    }

    let nonempty = zcol.columns().len();
    let mean_height = if nonempty == 0 {
        0.0
    } else {
        n as f32 / nonempty as f32
    };

    let mut lats = Vec::new();
    let mut cands = 0f64;
    let mut cols = 0f64;
    let mut reverts = 0f64;
    let mut revert_hits = 0usize;
    let nq = queries.len();
    for q in queries {
        let t = Instant::now();
        let (_, ex) = zcol.search_with_explain(q, k, ef).unwrap();
        lats.push(t.elapsed().as_secs_f64() * 1000.0);
        cands += ex.candidate_pool as f64;
        cols += ex.columns_scanned as f64;
        reverts += ex.revert_count as f64;
        if ex.revert_count > 0 {
            revert_hits += 1;
        }
    }
    lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = lats[lats.len() / 2] as f32;
    let recall = recall_mean(&zcol, queries, ground, k, ef);
    let vs = recall / hnsw_recall.max(1e-6);
    let lat_x = p50 / hnsw_p50.max(1e-9);
    let avg_cands = (cands / nq as f64) as f32;
    let touch = avg_cands / n as f32;
    let g1 = vs >= 0.98;
    let g2 = lat_x <= 1.5;
    let g3 = touch < 0.5;
    // Score: reward gate passes; penalize distance from gates (for Pareto ranking).
    let score = (if g1 { 10.0 } else { vs * 5.0 })
        + (if g2 {
            10.0
        } else {
            (1.5 / lat_x.max(0.1)).min(5.0)
        })
        + (if g3 {
            10.0
        } else {
            ((0.5 - touch).max(0.0) * 10.0) + (1.0 - touch) * 2.0
        });

    Row {
        distro: distro_name.into(),
        n,
        ef,
        rings,
        beam_r,
        fcols,
        recall,
        vs_hnsw: vs,
        p50_ms: p50,
        lat_x,
        touch,
        avg_cands,
        avg_cols: (cols / nq as f64) as f32,
        revert_avg: (reverts / nq as f64) as f32,
        revert_frac: revert_hits as f32 / nq as f32,
        nonempty_cols: nonempty,
        mean_height,
        g1,
        g2,
        g3,
        score,
    }
}

fn print_row(r: &Row) {
    println!(
        "{:<10} {:>6} {:>4} {:>5} {:>5} {:>5} {:>7.4} {:>7.4} {:>7.2} {:>6.2} {:>6.3} {:>7.0} {:>6.1} {:>5.2} {:>5.2} {}{}{} {:>5.1}",
        r.distro,
        r.n,
        r.ef,
        r.rings,
        r.beam_r,
        r.fcols,
        r.recall,
        r.vs_hnsw,
        r.p50_ms,
        r.lat_x,
        r.touch,
        r.avg_cands,
        r.avg_cols,
        r.revert_avg,
        r.revert_frac,
        if r.g1 { "Y" } else { "n" },
        if r.g2 { "Y" } else { "n" },
        if r.g3 { "Y" } else { "n" },
        r.score
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");
    let json_out = args.iter().any(|a| a == "--json");

    let dim = 128usize;
    let k = 10usize;
    let nq = if quick { 25usize } else { 40usize };
    let scales: Vec<usize> = if quick {
        vec![2_000, 10_000]
    } else {
        vec![2_000, 10_000, 50_000]
    };

    // Regime grid aimed at G1∧G2∧G3 discovery (ef is the main lever for τ).
    let regimes: Vec<(usize, u16, u16, usize)> = if quick {
        vec![
            (16, 0, 0, 0),
            (32, 0, 0, 0),
            (64, 0, 0, 0),
            (32, 1, 1, 16),
            (32, 2, 1, 32),
            (64, 1, 1, 16),
            (128, 0, 0, 0),
            (128, 8, 2, 96), // default-ish
        ]
    } else {
        vec![
            // Pure beam — isolate fractal walk
            (8, 0, 0, 0),
            (16, 0, 0, 0),
            (24, 0, 0, 0),
            (32, 0, 0, 0),
            (48, 0, 0, 0),
            (64, 0, 0, 0),
            (96, 0, 0, 0),
            (128, 0, 0, 0),
            // Light fallback
            (16, 1, 1, 8),
            (24, 1, 1, 16),
            (32, 1, 1, 16),
            (32, 2, 1, 32),
            (48, 1, 1, 16),
            (48, 2, 1, 32),
            (64, 1, 1, 16),
            (64, 2, 2, 32),
            (64, 2, 2, 64),
            // Heavier / default
            (96, 2, 2, 64),
            (128, 2, 2, 96),
            (128, 8, 2, 96),
        ]
    };

    let distros: Vec<(Distro, &str)> = vec![
        (Distro::UnitSphere, "sphere"),
        (
            Distro::GaussianClusters {
                clusters: 32,
                sigma: 0.08,
            },
            "clusters32",
        ),
    ];

    println!(
        "{:<10} {:>6} {:>4} {:>5} {:>5} {:>5} {:>7} {:>7} {:>7} {:>6} {:>6} {:>7} {:>6} {:>5} {:>5} G123 score",
        "distro", "n", "ef", "ring", "beam", "fcol", "recall", "vsH", "p50ms", "lat×", "touch", "cands", "cols", "revμ", "rev%"
    );

    let mut rows: Vec<Row> = Vec::new();

    for &n in &scales {
        for &(distro, dname) in &distros {
            let mut rng = StdRng::seed_from_u64(42 + n as u64);
            let vectors = gen_vectors(&mut rng, n, dim, distro);
            let queries = gen_vectors(&mut rng, nq, dim, distro);

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

            // HNSW baseline at ef=128 (protocol default) for vs ratios.
            let hnsw_ef = 128usize;
            let hnsw_recall = recall_mean(&hnsw, &queries, &ground, k, hnsw_ef);
            let hnsw_p50 = p50_ms(&hnsw, &queries, k, hnsw_ef);
            println!(
                "# baseline n={n} {dname}: HNSW recall@10={hnsw_recall:.4} p50={hnsw_p50:.3}ms (ef={hnsw_ef})"
            );

            let regime_iter: Vec<(usize, u16, u16, usize)> = if !quick && n >= 50_000 {
                regimes
                    .iter()
                    .copied()
                    .filter(|(ef, rings, _, _)| *ef <= 64 || (*ef == 128 && *rings == 0))
                    .collect()
            } else {
                regimes.clone()
            };

            for &(ef, rings, beam_r, fcols) in &regime_iter {
                let row = run_zcol(
                    &vectors,
                    &queries,
                    &ground,
                    dim,
                    k,
                    ef,
                    rings,
                    beam_r,
                    fcols,
                    hnsw_recall,
                    hnsw_p50,
                    dname,
                );
                print_row(&row);
                rows.push(row);
            }
        }
    }

    // Projection-seed symmetry break (description symmetry of observer).
    {
        let n = 2_000usize;
        let mut rng = StdRng::seed_from_u64(7);
        let vectors = gen_vectors(&mut rng, n, dim, Distro::UnitSphere);
        let queries = gen_vectors(&mut rng, nq, dim, Distro::UnitSphere);
        let mut flat = FlatIndex::new(dim, DistanceMetric::Cosine);
        for (i, v) in vectors.iter().enumerate() {
            flat.insert(VectorId(i as u64), Vector::new(v.clone()))
                .unwrap();
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
        println!("# seed-sensitivity n=2000 ef=32 rings=0 (description symmetry)");
        for seed in [1u64, 42, 999] {
            let mut cfg = ZColumnConfig::default();
            cfg.projection_seed = seed;
            cfg.max_fallback_rings = 0;
            cfg.fallback_beam_radius = 0;
            cfg.max_fallback_columns = 0;
            cfg.ef_search = 32;
            let mut zcol = ZColumnIndex::new(dim, DistanceMetric::Cosine, cfg);
            for (i, v) in vectors.iter().enumerate() {
                zcol.insert(VectorId(i as u64), Vector::new(v.clone()))
                    .unwrap();
            }
            let recall = recall_mean(&zcol, &queries, &ground, k, 32);
            let mut cands = 0f64;
            for q in &queries {
                let (_, ex) = zcol.search_with_explain(q, k, 32).unwrap();
                cands += ex.candidate_pool as f64;
            }
            println!(
                "  seed={seed} recall={recall:.4} touch={:.3} nonempty={}",
                (cands / nq as f64) / n as f64,
                zcol.columns().len()
            );
        }
    }

    // Pareto / best candidates toward all three gates.
    println!();
    println!("=== CLOSEST TO G1∧G2∧G3 (by score) ===");
    let mut ranked = rows.clone();
    ranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    for r in ranked.iter().take(12) {
        print_row(r);
    }

    let all_three: Vec<_> = rows.iter().filter(|r| r.g1 && r.g2 && r.g3).collect();
    println!();
    if all_three.is_empty() {
        println!("=== NO REGIME PASSED G1∧G2∧G3 ===");
        // Best G3 among near-G1
        let mut near: Vec<_> = rows.iter().filter(|r| r.vs_hnsw >= 0.90).cloned().collect();
        near.sort_by(|a, b| a.touch.partial_cmp(&b.touch).unwrap());
        println!("=== Lowest touch among recall≥0.90×HNSW ===");
        for r in near.iter().take(8) {
            print_row(r);
        }
        let mut fast: Vec<_> = rows.iter().filter(|r| r.vs_hnsw >= 0.90).cloned().collect();
        fast.sort_by(|a, b| a.lat_x.partial_cmp(&b.lat_x).unwrap());
        println!("=== Lowest latency× among recall≥0.90×HNSW ===");
        for r in fast.iter().take(8) {
            print_row(r);
        }
    } else {
        println!("=== REGIMES PASSING ALL GATES ===");
        for r in &all_three {
            print_row(r);
        }
    }

    // Dimensionless summary
    println!();
    println!("=== DIMENSIONLESS GROUPS (sphere n=10000, pure beam rings=0) ===");
    for r in rows
        .iter()
        .filter(|r| r.distro == "sphere" && r.n == 10_000 && r.rings == 0 && r.fcols == 0)
    {
        println!(
            "  β=ef/k={:.1}  τ={:.3}  ρ_r={:.3}  λ={:.2}  φ=mean_h={:.1}  nonempty={}",
            r.ef as f32 / k as f32,
            r.touch,
            r.vs_hnsw,
            r.lat_x,
            r.mean_height,
            r.nonempty_cols
        );
    }

    if json_out {
        let path = "/tmp/topolsea-math-probe-full.json";
        std::fs::write(path, serde_json::to_string_pretty(&rows).unwrap()).unwrap();
        eprintln!("wrote {path}");
    }
}
