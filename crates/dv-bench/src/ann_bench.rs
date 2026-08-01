//! ANN-Benchmarks-style harness: load fvecs/ivecs, measure recall@k and QPS.

use dv_index_api::VectorIndex;
use dv_index_flat::FlatIndex;
use dv_index_hnsw::HnswIndex;
use dv_index_ivf::IvfIndex;
use dv_index_zcolumn::ZColumnIndex;
use dv_types::{DistanceMetric, HnswConfig, IndexKind, IvfConfig, Vector, VectorId, ZColumnConfig};
use rand::Rng;
use serde::Serialize;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct AnnDataset {
    pub base: Vec<Vec<f32>>,
    pub queries: Vec<Vec<f32>>,
    pub groundtruth: Vec<Vec<u32>>,
    pub dimension: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnnBenchReport {
    pub dataset: String,
    pub index: String,
    pub metric: String,
    pub n_base: usize,
    pub n_queries: usize,
    pub top_k: usize,
    pub recall_at_k: f64,
    pub qps: f64,
    pub build_secs: f64,
    pub notes: String,
}

/// Read little-endian `.fvecs` (dim:i32 + dim×f32 per vector).
pub fn read_fvecs(path: impl AsRef<Path>) -> std::io::Result<Vec<Vec<f32>>> {
    let mut file = BufReader::new(File::open(path)?);
    let mut out = Vec::new();
    loop {
        let mut dim_buf = [0u8; 4];
        match file.read_exact(&mut dim_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        let dim = i32::from_le_bytes(dim_buf) as usize;
        let mut vec = vec![0.0f32; dim];
        for slot in &mut vec {
            let mut b = [0u8; 4];
            file.read_exact(&mut b)?;
            *slot = f32::from_le_bytes(b);
        }
        out.push(vec);
    }
    Ok(out)
}

/// Read little-endian `.ivecs` (dim:i32 + dim×i32 per row).
pub fn read_ivecs(path: impl AsRef<Path>) -> std::io::Result<Vec<Vec<u32>>> {
    let mut file = BufReader::new(File::open(path)?);
    let mut out = Vec::new();
    loop {
        let mut dim_buf = [0u8; 4];
        match file.read_exact(&mut dim_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        let dim = i32::from_le_bytes(dim_buf) as usize;
        let mut row = vec![0u32; dim];
        for slot in &mut row {
            let mut b = [0u8; 4];
            file.read_exact(&mut b)?;
            *slot = i32::from_le_bytes(b) as u32;
        }
        out.push(row);
    }
    Ok(out)
}

pub fn load_dataset(base: &Path, query: &Path, gt: &Path) -> std::io::Result<AnnDataset> {
    let base_v = read_fvecs(base)?;
    let queries = read_fvecs(query)?;
    let groundtruth = read_ivecs(gt)?;
    let dimension = base_v.first().map(|v| v.len()).unwrap_or(0);
    Ok(AnnDataset {
        base: base_v,
        queries,
        groundtruth,
        dimension,
    })
}

/// Synthetic fallback when no external dataset is present (CI-friendly).
pub fn synthetic_dataset(n_base: usize, n_query: usize, dim: usize, seed: u64) -> AnnDataset {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    use rand::SeedableRng;
    let mut base = Vec::with_capacity(n_base);
    for i in 0..n_base {
        let mut v = vec![0.0f32; dim];
        v[0] = i as f32;
        for x in v.iter_mut().skip(1) {
            *x = rng.gen::<f32>();
        }
        base.push(v);
    }
    let mut queries = Vec::with_capacity(n_query);
    let mut groundtruth = Vec::with_capacity(n_query);
    for q in 0..n_query {
        let target = (q * 7) % n_base;
        queries.push(base[target].clone());
        // Exact neighbor is `target` under L2 for this construction.
        groundtruth.push(vec![target as u32]);
    }
    AnnDataset {
        base,
        queries,
        groundtruth,
        dimension: dim,
    }
}

fn build_index(
    kind: IndexKind,
    dim: usize,
    metric: DistanceMetric,
    data: &[(VectorId, Vec<f32>)],
) -> Box<dyn VectorIndex> {
    match kind {
        IndexKind::Flat => {
            let mut idx = FlatIndex::new(dim, metric);
            for (id, v) in data {
                idx.insert(*id, Vector::new(v.clone())).unwrap();
            }
            Box::new(idx)
        }
        IndexKind::Hnsw => {
            let mut idx = HnswIndex::new(dim, metric, HnswConfig::default());
            for (id, v) in data {
                idx.insert(*id, Vector::new(v.clone())).unwrap();
            }
            Box::new(idx)
        }
        IndexKind::ZColumn => {
            let mut idx = ZColumnIndex::new(dim, metric, ZColumnConfig::default());
            for (id, v) in data {
                idx.insert(*id, Vector::new(v.clone())).unwrap();
            }
            Box::new(idx)
        }
        IndexKind::Ivf => {
            let mut idx = IvfIndex::new(
                dim,
                metric,
                IvfConfig {
                    nlist: 32.min(data.len().max(1)),
                    nprobe: 8,
                    pq_m: None,
                    seed: 42,
                },
            );
            for (id, v) in data {
                idx.insert(*id, Vector::new(v.clone())).unwrap();
            }
            Box::new(idx)
        }
    }
}

pub fn run_ann_bench(
    dataset_name: &str,
    ds: &AnnDataset,
    kind: IndexKind,
    metric: DistanceMetric,
    top_k: usize,
) -> AnnBenchReport {
    let data: Vec<_> = ds
        .base
        .iter()
        .enumerate()
        .map(|(i, v)| (VectorId(i as u64), v.clone()))
        .collect();

    let t0 = Instant::now();
    let index = build_index(kind, ds.dimension, metric, &data);
    let build_secs = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let mut hits_ok = 0usize;
    let mut total = 0usize;
    for (qi, q) in ds.queries.iter().enumerate() {
        let hits = index.search(q, top_k, 64).unwrap();
        let gt = ds.groundtruth.get(qi).map(|g| g.as_slice()).unwrap_or(&[]);
        let want: std::collections::HashSet<u64> =
            gt.iter().take(top_k).map(|x| *x as u64).collect();
        if want.is_empty() {
            continue;
        }
        total += 1;
        if hits.iter().any(|h| want.contains(&h.id.raw())) {
            hits_ok += 1;
        }
    }
    let elapsed = t1.elapsed().as_secs_f64().max(1e-9);
    let qps = ds.queries.len() as f64 / elapsed;
    let recall = if total == 0 {
        0.0
    } else {
        hits_ok as f64 / total as f64
    };

    let notes = match kind {
        IndexKind::ZColumn => {
            "Z-Column numbers are diagnostic only — do not claim production ANN win without G1∧G2∧G3"
                .into()
        }
        IndexKind::Hnsw => "Product-default ANN path".into(),
        IndexKind::Ivf => "Memory-bound IVF path (optional PQ)".into(),
        IndexKind::Flat => "Exact baseline".into(),
    };

    AnnBenchReport {
        dataset: dataset_name.into(),
        index: format!("{kind:?}"),
        metric: metric.to_string(),
        n_base: ds.base.len(),
        n_queries: ds.queries.len(),
        top_k,
        recall_at_k: recall,
        qps,
        build_secs,
        notes,
    }
}

pub fn try_load_sift_or_synthetic(data_dir: Option<PathBuf>) -> (String, AnnDataset) {
    if let Some(dir) = data_dir {
        let base = dir.join("sift_base.fvecs");
        let query = dir.join("sift_query.fvecs");
        let gt = dir.join("sift_groundtruth.ivecs");
        if base.exists() && query.exists() && gt.exists() {
            if let Ok(ds) = load_dataset(&base, &query, &gt) {
                return ("sift1m".into(), ds);
            }
        }
    }
    (
        "synthetic_2k_d32".into(),
        synthetic_dataset(2_000, 50, 32, 42),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_hnsw_has_nonzero_recall() {
        let ds = synthetic_dataset(500, 20, 16, 1);
        let report = run_ann_bench("syn", &ds, IndexKind::Hnsw, DistanceMetric::L2, 1);
        assert!(report.recall_at_k > 0.5, "{report:?}");
    }
}
