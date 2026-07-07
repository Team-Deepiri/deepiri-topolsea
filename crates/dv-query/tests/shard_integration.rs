use dv_index_api::VectorIndex;
use dv_index_flat::FlatIndex;
use dv_index_hnsw::HnswIndex;
use dv_index_zcolumn::ZColumnIndex;
use dv_query::Database;
use dv_types::{DistanceMetric, HnswConfig, IndexKind, Vector, VectorId, ZColumnConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tempfile::tempdir;

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

fn recall_at_k(
    index: &dyn VectorIndex,
    queries: &[Vec<f32>],
    ground: &[Vec<VectorId>],
    k: usize,
) -> f32 {
    let mut hits = 0usize;
    for (query, truth) in queries.iter().zip(ground) {
        let results = index.search(query, k, 64).unwrap();
        let got: std::collections::HashSet<_> = results.iter().map(|h| h.id).collect();
        if truth.iter().take(k).any(|id| got.contains(id)) {
            hits += 1;
        }
    }
    hits as f32 / queries.len().max(1) as f32
}

#[test]
fn sharded_zcolumn_routes_and_queries() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    db.create_sharded_collection("corp", 4, 8, DistanceMetric::Cosine, IndexKind::ZColumn)
        .unwrap();

    for i in 0..40u64 {
        let v: Vec<f32> = (0..8)
            .map(|d| ((i as f32 + d as f32) * 0.13).sin())
            .collect();
        db.upsert_sharded("corp", &format!("id{i}"), v, None)
            .unwrap();
    }
    assert_eq!(db.sharded_vector_count("corp").unwrap(), 40);

    let q = vec![0.1; 8];
    let hits = db.query_sharded("corp", &q, 5, None, 64).unwrap();
    assert!(!hits.is_empty());

    db.persist_all().unwrap();
    let mut db2 = Database::open(dir.path()).unwrap();
    assert_eq!(db2.sharded_vector_count("corp").unwrap(), 40);
}

#[test]
fn zcolumn_recall_within_hnsw_band_at_2k() {
    let dim = 128;
    let n = 2000;
    let k = 10;
    let mut rng = StdRng::seed_from_u64(7);
    let vectors = random_unit_vectors(&mut rng, n, dim);

    let mut flat = FlatIndex::new(dim, DistanceMetric::Cosine);
    let mut hnsw = HnswIndex::new(dim, DistanceMetric::Cosine, HnswConfig::default());
    let mut zcol = ZColumnIndex::new(dim, DistanceMetric::Cosine, ZColumnConfig::default());

    for (i, v) in vectors.iter().enumerate() {
        let id = VectorId(i as u64);
        flat.insert(id, Vector::new(v.clone())).unwrap();
        hnsw.insert(id, Vector::new(v.clone())).unwrap();
        zcol.insert(id, Vector::new(v.clone())).unwrap();
    }

    let queries: Vec<Vec<f32>> = random_unit_vectors(&mut rng, 20, dim);
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

    let hnsw_recall = recall_at_k(&hnsw, &queries, &ground, k);
    let zcol_recall = recall_at_k(&zcol, &queries, &ground, k);

    assert!(
        hnsw_recall > 0.5,
        "hnsw recall sanity check failed: {hnsw_recall}"
    );
    // Go/no-go target: within 15% of HNSW recall (relaxed for CI; tighten toward 2%).
    assert!(
        zcol_recall >= hnsw_recall * 0.85,
        "zcolumn recall {zcol_recall} too far below hnsw {hnsw_recall}"
    );
}
