use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use dv_index_api::VectorIndex;
use dv_index_flat::FlatIndex;
use dv_index_hnsw::HnswIndex;
use dv_index_zcolumn::ZColumnIndex;
use dv_types::{DistanceMetric, HnswConfig, Vector, VectorId, ZColumnConfig};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

fn random_vectors(rng: &mut StdRng, n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

fn bench_flat(c: &mut Criterion) {
    let mut group = c.benchmark_group("flat_search");
    for n in [100, 1000] {
        let dim = 128;
        let mut rng = StdRng::seed_from_u64(7);
        let vectors = random_vectors(&mut rng, n, dim);
        let mut idx = FlatIndex::new(dim, DistanceMetric::L2);
        for (i, v) in vectors.iter().enumerate() {
            idx.insert(VectorId(i as u64), Vector::new(v.clone()))
                .unwrap();
        }
        let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| idx.search(black_box(&query), 10, 0).unwrap());
        });
    }
    group.finish();
}

fn bench_hnsw(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search");
    for n in [100, 1000] {
        let dim = 128;
        let mut rng = StdRng::seed_from_u64(7);
        let vectors = random_vectors(&mut rng, n, dim);
        let mut idx = HnswIndex::new(dim, DistanceMetric::L2, HnswConfig::default());
        for (i, v) in vectors.iter().enumerate() {
            idx.insert(VectorId(i as u64), Vector::new(v.clone()))
                .unwrap();
        }
        let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| idx.search(black_box(&query), 10, 64).unwrap());
        });
    }
    group.finish();
}

fn bench_zcolumn(c: &mut Criterion) {
    let mut group = c.benchmark_group("zcolumn_search");
    for n in [100, 1000] {
        let dim = 128;
        let mut rng = StdRng::seed_from_u64(7);
        let vectors = random_vectors(&mut rng, n, dim);
        let mut idx = ZColumnIndex::new(dim, DistanceMetric::L2, ZColumnConfig::default());
        for (i, v) in vectors.iter().enumerate() {
            idx.insert(VectorId(i as u64), Vector::new(v.clone()))
                .unwrap();
        }
        let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| idx.search(black_box(&query), 10, 64).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_flat, bench_hnsw, bench_zcolumn);
criterion_main!(benches);
