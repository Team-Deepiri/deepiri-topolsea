//! Commercial proof harness — recall, QPS, and footprint at scale.
//! All cost/TCO math lives here, not in the search hot path.

use dv_index_api::VectorIndex;
use dv_index_flat::FlatIndex;
use dv_index_hnsw::HnswIndex;
use dv_index_zcolumn::ZColumnIndex;
use dv_types::{DistanceMetric, HnswConfig, Vector, VectorId, ZColumnConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProveConfig {
    pub dimension: usize,
    pub k: usize,
    pub ef: usize,
    pub num_queries: usize,
    pub seed: u64,
    pub scales: Vec<usize>,
}

impl Default for ProveConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            k: 10,
            ef: 128,
            num_queries: 50,
            seed: 42,
            scales: vec![10_000, 100_000],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommercialProofReport {
    pub config: ProveConfig,
    pub scales: Vec<ScaleProof>,
    pub buyer_summary: BuyerSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleProof {
    pub n_vectors: usize,
    pub flat: IndexProof,
    pub hnsw: IndexProof,
    pub zcolumn: IndexProof,
    pub zcolumn_vs_hnsw_recall_ratio: f32,
    pub zcolumn_vs_hnsw_qps_ratio: f32,
    pub zcolumn_vs_hnsw_footprint_ratio: f32,
    pub zcolumn_corpus_touch_fraction: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexProof {
    pub index: String,
    pub recall_at_k_mean: f32,
    pub qps: f32,
    pub p50_query_ms: f32,
    pub index_bytes: u64,
    pub quantized_bytes: u64,
    pub fp32_resident_bytes: u64,
    pub avg_candidates_touched: f32,
    pub avg_columns_scanned: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuyerSummary {
    pub ai_infra_tco: String,
    pub on_prem_edge: String,
    pub deepiri_internal: String,
}

pub fn run(config: ProveConfig) -> CommercialProofReport {
    let mut scales = Vec::new();
    for &n in &config.scales {
        scales.push(prove_scale(&config, n));
    }
    let buyer_summary = summarize(&scales);
    CommercialProofReport {
        config,
        scales,
        buyer_summary,
    }
}

fn prove_scale(config: &ProveConfig, n: usize) -> ScaleProof {
    let mut rng = StdRng::seed_from_u64(config.seed);
    let vectors = random_unit_vectors(&mut rng, n, config.dimension);

    let mut flat = FlatIndex::new(config.dimension, DistanceMetric::Cosine);
    let mut hnsw = HnswIndex::new(
        config.dimension,
        DistanceMetric::Cosine,
        HnswConfig::default(),
    );
    let mut zcol = ZColumnIndex::new(
        config.dimension,
        DistanceMetric::Cosine,
        ZColumnConfig::default(),
    );

    for (i, v) in vectors.iter().enumerate() {
        let id = VectorId(i as u64);
        flat.insert(id, Vector::new(v.clone())).unwrap();
        hnsw.insert(id, Vector::new(v.clone())).unwrap();
        zcol.insert(id, Vector::new(v.clone())).unwrap();
    }

    let queries: Vec<Vec<f32>> =
        random_unit_vectors(&mut rng, config.num_queries, config.dimension);
    let ground = build_ground_truth(&flat, &queries, config.k);

    let flat_proof = bench_index(
        "flat",
        &flat,
        &queries,
        &ground,
        config.k,
        config.ef,
        flat_footprint(&flat),
    );
    let hnsw_proof = bench_index(
        "hnsw",
        &hnsw,
        &queries,
        &ground,
        config.k,
        config.ef,
        hnsw_footprint(&hnsw),
    );
    let zcol_proof = bench_zcolumn(&zcol, &queries, &ground, config.k, config.ef);

    let zcolumn_vs_hnsw_recall_ratio =
        zcol_proof.recall_at_k_mean / hnsw_proof.recall_at_k_mean.max(1e-6);
    let zcolumn_vs_hnsw_qps_ratio = zcol_proof.qps / hnsw_proof.qps.max(1e-6);
    let zcolumn_vs_hnsw_footprint_ratio =
        zcol_proof.index_bytes as f32 / hnsw_proof.index_bytes.max(1) as f32;
    let zcolumn_corpus_touch_fraction = zcol_proof.avg_candidates_touched / n as f32;

    ScaleProof {
        n_vectors: n,
        flat: flat_proof,
        hnsw: hnsw_proof,
        zcolumn: zcol_proof,
        zcolumn_vs_hnsw_recall_ratio,
        zcolumn_vs_hnsw_qps_ratio,
        zcolumn_vs_hnsw_footprint_ratio,
        zcolumn_corpus_touch_fraction,
    }
}

fn bench_index(
    name: &str,
    index: &dyn VectorIndex,
    queries: &[Vec<f32>],
    ground: &[Vec<VectorId>],
    k: usize,
    ef: usize,
    footprint: (u64, u64, u64),
) -> IndexProof {
    let mut latencies = Vec::with_capacity(queries.len());
    let start = Instant::now();
    for q in queries {
        let t0 = Instant::now();
        let _ = index.search(q, k, ef).unwrap();
        latencies.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let qps = queries.len() as f64 / elapsed;

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p50 = latencies[latencies.len() / 2];

    IndexProof {
        index: name.into(),
        recall_at_k_mean: recall_at_k_mean(index, queries, ground, k, ef),
        qps: qps as f32,
        p50_query_ms: p50 as f32,
        index_bytes: footprint.0,
        quantized_bytes: footprint.1,
        fp32_resident_bytes: footprint.2,
        avg_candidates_touched: 0.0,
        avg_columns_scanned: 0.0,
    }
}

fn bench_zcolumn(
    index: &ZColumnIndex,
    queries: &[Vec<f32>],
    ground: &[Vec<VectorId>],
    k: usize,
    ef: usize,
) -> IndexProof {
    let footprint = zcolumn_footprint(index);
    let mut latencies = Vec::with_capacity(queries.len());
    let mut candidates = 0f64;
    let mut columns = 0f64;

    let start = Instant::now();
    for q in queries {
        let t0 = Instant::now();
        let (_, explain) = index.search_with_explain(q, k, ef).unwrap();
        latencies.push(t0.elapsed().as_secs_f64() * 1000.0);
        candidates += explain.candidate_pool as f64;
        columns += explain.columns_scanned as f64;
    }
    let elapsed = start.elapsed().as_secs_f64();
    let qps = queries.len() as f64 / elapsed;
    let nq = queries.len() as f64;

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p50 = latencies[latencies.len() / 2];

    IndexProof {
        index: "zcolumn".into(),
        recall_at_k_mean: recall_at_k_mean(index, queries, ground, k, ef),
        qps: qps as f32,
        p50_query_ms: p50 as f32,
        index_bytes: footprint.0,
        quantized_bytes: footprint.1,
        fp32_resident_bytes: footprint.2,
        avg_candidates_touched: (candidates / nq) as f32,
        avg_columns_scanned: (columns / nq) as f32,
    }
}

fn flat_footprint(index: &FlatIndex) -> (u64, u64, u64) {
    let bytes = index.to_bytes().unwrap_or_default().len() as u64;
    (bytes, 0, bytes)
}

fn hnsw_footprint(index: &HnswIndex) -> (u64, u64, u64) {
    let bytes = index.to_bytes().unwrap_or_default().len() as u64;
    (bytes, 0, bytes)
}

fn zcolumn_footprint(index: &ZColumnIndex) -> (u64, u64, u64) {
    let mut quant = 0u64;
    for col in index.columns().values() {
        for payload in &col.quantized {
            quant += payload.len() as u64;
        }
    }
    let fp32 = (index.len() * index.dimension() * 4) as u64;
    let serialized = index.to_bytes().unwrap_or_default().len() as u64;
    (serialized.max(quant + fp32), quant, fp32)
}

fn random_unit_vectors(rng: &mut StdRng, n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > f32::EPSILON {
                for x in &mut v {
                    *x /= norm;
                }
            }
            v
        })
        .collect()
}

fn build_ground_truth(flat: &FlatIndex, queries: &[Vec<f32>], k: usize) -> Vec<Vec<VectorId>> {
    queries
        .iter()
        .map(|q| {
            flat.search(q, k, 0)
                .unwrap()
                .into_iter()
                .map(|h| h.id)
                .collect()
        })
        .collect()
}

fn recall_at_k_mean(
    index: &dyn VectorIndex,
    queries: &[Vec<f32>],
    ground: &[Vec<VectorId>],
    k: usize,
    ef: usize,
) -> f32 {
    let mut total = 0.0f32;
    for (query, truth) in queries.iter().zip(ground) {
        let results = index.search(query, k, ef).unwrap();
        let got: HashSet<_> = results.iter().map(|h| h.id).collect();
        let truth_set: HashSet<_> = truth.iter().take(k).copied().collect();
        let overlap = got.intersection(&truth_set).count();
        total += overlap as f32 / k as f32;
    }
    total / queries.len().max(1) as f32
}

fn summarize(scales: &[ScaleProof]) -> BuyerSummary {
    let largest = scales.last();
    let Some(s) = largest else {
        return BuyerSummary {
            ai_infra_tco: "no data".into(),
            on_prem_edge: "no data".into(),
            deepiri_internal: "no data".into(),
        };
    };

    let recall_ok = s.zcolumn_vs_hnsw_recall_ratio >= 0.98;
    let touch = s.zcolumn_corpus_touch_fraction;
    let fp_ratio = s.zcolumn_vs_hnsw_footprint_ratio;
    let qps_ratio = s.zcolumn_vs_hnsw_qps_ratio;

    BuyerSummary {
        ai_infra_tco: format!(
            "At {} vectors: Z-Column recall {:.1}% of HNSW, touches {:.1}% of corpus per query, index footprint {:.0}% of HNSW. {}",
            s.n_vectors,
            s.zcolumn_vs_hnsw_recall_ratio * 100.0,
            touch * 100.0,
            fp_ratio * 100.0,
            if recall_ok && touch < 0.5 {
                "TCO story: fixed-recall ANN with less data scanned per query."
            } else if recall_ok {
                "TCO story: recall parity held; optimize candidate pool for scan cost."
            } else {
                "TCO story: recall below HNSW band — tune ef/hybrid pool before selling TCO."
            }
        ),
        on_prem_edge: format!(
            "Quantized column payload {} MB, FP32 resident {} MB, {} fractal columns — mmap-friendly segments, {:.0}% HNSW index bytes.",
            s.zcolumn.quantized_bytes / 1_048_576,
            s.zcolumn.fp32_resident_bytes / 1_048_576,
            s.zcolumn.avg_columns_scanned as u64,
            fp_ratio * 100.0,
        ),
        deepiri_internal: format!(
            "vs flat: {:.1}x QPS; vs HNSW: {:.1}x QPS at {:.1}% recall. Recommend Z-Column as default Topolsea retrieval layer when recall ≥98% HNSW and QPS ≥{:.0}% HNSW.",
            s.hnsw.qps / s.flat.qps.max(1e-6),
            qps_ratio * 100.0,
            s.zcolumn_vs_hnsw_recall_ratio * 100.0,
            80.0,
        ),
    }
}
