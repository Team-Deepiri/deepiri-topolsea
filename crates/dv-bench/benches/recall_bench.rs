use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dv_index_api::VectorIndex;
use dv_index_flat::FlatIndex;
use dv_index_hnsw::HnswIndex;
use dv_index_zcolumn::ZColumnIndex;
use dv_types::{DistanceMetric, HnswConfig, Vector, VectorId, ZColumnConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;

fn random_vectors(rng: &mut StdRng, n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

fn recall_at_k(
    index: &dyn VectorIndex,
    queries: &[Vec<f32>],
    ground: &[Vec<VectorId>],
    k: usize,
) -> f32 {
    let mut hits = 0usize;
    for (query, truth) in queries.iter().zip(ground) {
        let results = index.search(query, k, 64).unwrap();
        let got: HashSet<_> = results.iter().map(|h| h.id).collect();
        if truth.iter().take(k).any(|id| got.contains(id)) {
            hits += 1;
        }
    }
    hits as f32 / queries.len().max(1) as f32
}

fn recall_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall_at_10");
    group.sample_size(10);

    let dim = 64;
    let n = 500;
    let k = 10;
    let mut rng = StdRng::seed_from_u64(42);
    let vectors = random_vectors(&mut rng, n, dim);

    let mut flat = FlatIndex::new(dim, DistanceMetric::Cosine);
    let mut hnsw = HnswIndex::new(dim, DistanceMetric::Cosine, HnswConfig::default());
    let mut zcol = ZColumnIndex::new(dim, DistanceMetric::Cosine, ZColumnConfig::default());

    for (i, v) in vectors.iter().enumerate() {
        let id = VectorId(i as u64);
        flat.insert(id, Vector::new(v.clone())).unwrap();
        hnsw.insert(id, Vector::new(v.clone())).unwrap();
        zcol.insert(id, Vector::new(v.clone())).unwrap();
    }

    let queries: Vec<Vec<f32>> = (0..10)
        .map(|_| random_vectors(&mut rng, 1, dim).pop().unwrap())
        .collect();
    let mut ground = Vec::new();
    for q in &queries {
        ground.push(
            flat.search(q, k, 0)
                .unwrap()
                .into_iter()
                .map(|h| h.id)
                .collect(),
        );
    }

    group.bench_function("flat_recall", |b| {
        let r = recall_at_k(&flat, &queries, &ground, k);
        b.iter(|| black_box(r));
    });
    group.bench_function("hnsw_recall", |b| {
        let r = recall_at_k(&hnsw, &queries, &ground, k);
        b.iter(|| black_box(r));
    });
    group.bench_function("zcolumn_recall", |b| {
        let r = recall_at_k(&zcol, &queries, &ground, k);
        b.iter(|| black_box(r));
    });

    group.finish();
}

criterion_group!(benches, recall_comparison);
criterion_main!(benches);
