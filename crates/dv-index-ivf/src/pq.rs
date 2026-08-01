//! Product Quantization helpers (m subspaces × 256 centroids).

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqCodebooks {
    pub m: usize,
    pub sub_dim: usize,
    /// [m][256][sub_dim]
    pub centroids: Vec<Vec<Vec<f32>>>,
}

pub fn train_pq_codebooks(vectors: &[Vec<f32>], dim: usize, m: usize, seed: u64) -> PqCodebooks {
    assert!(m > 0 && dim.is_multiple_of(m), "dim must be divisible by m");
    let sub_dim = dim / m;
    let mut rng = StdRng::seed_from_u64(seed);

    let mut centroids = Vec::with_capacity(m);
    for subspace in 0..m {
        let start = subspace * sub_dim;
        let end = start + sub_dim;
        let mut code_cents = Vec::with_capacity(256);
        for i in 0..256 {
            if vectors.is_empty() {
                code_cents.push(vec![0.0; sub_dim]);
            } else {
                let src = &vectors[i % vectors.len()];
                let mut c: Vec<f32> = src[start..end].to_vec();
                for v in &mut c {
                    *v += rng.gen::<f32>() * 1e-3;
                }
                code_cents.push(c);
            }
        }
        for _ in 0..5 {
            let mut sums = vec![vec![0.0f32; sub_dim]; 256];
            let mut counts = vec![0u32; 256];
            for v in vectors {
                let sub = &v[start..end];
                let mut best = 0usize;
                let mut best_d = f32::MAX;
                for (ci, c) in code_cents.iter().enumerate() {
                    let d = l2(sub, c);
                    if d < best_d {
                        best_d = d;
                        best = ci;
                    }
                }
                for (a, b) in sums[best].iter_mut().zip(sub.iter()) {
                    *a += *b;
                }
                counts[best] += 1;
            }
            for i in 0..256 {
                if counts[i] > 0 {
                    for v in &mut sums[i] {
                        *v /= counts[i] as f32;
                    }
                    code_cents[i] = sums[i].clone();
                }
            }
        }
        centroids.push(code_cents);
    }
    PqCodebooks {
        m,
        sub_dim,
        centroids,
    }
}

pub fn encode_pq(codebooks: &PqCodebooks, vector: &[f32]) -> Vec<u8> {
    let mut codes = Vec::with_capacity(codebooks.m);
    for subspace in 0..codebooks.m {
        let start = subspace * codebooks.sub_dim;
        let end = start + codebooks.sub_dim;
        let sub = &vector[start..end];
        let mut best = 0u8;
        let mut best_d = f32::MAX;
        for (ci, c) in codebooks.centroids[subspace].iter().enumerate() {
            let d = l2(sub, c);
            if d < best_d {
                best_d = d;
                best = ci as u8;
            }
        }
        codes.push(best);
    }
    codes
}

pub fn asymmetric_distance(codebooks: &PqCodebooks, query: &[f32], codes: &[u8]) -> f32 {
    let mut dist = 0.0f32;
    for (subspace, &code) in codes.iter().enumerate().take(codebooks.m) {
        let start = subspace * codebooks.sub_dim;
        let end = start + codebooks.sub_dim;
        let sub = &query[start..end];
        let c = &codebooks.centroids[subspace][code as usize];
        dist += l2(sub, c);
    }
    dist
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}
