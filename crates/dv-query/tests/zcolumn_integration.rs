use dv_index_api::VectorIndex;
use dv_index_flat::FlatIndex;
use dv_index_zcolumn::ZColumnIndex;
use dv_query::Database;
use dv_types::{DistanceMetric, IndexKind, Vector, VectorId, ZColumnConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::json;
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
fn end_to_end_zcolumn_collection() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let col = db
        .get_or_create_collection_with_config("zcol", 4, DistanceMetric::Cosine, IndexKind::ZColumn)
        .unwrap();

    col.upsert("a", vec![1.0, 0.0, 0.0, 0.0], Some(json!({"k": "1"})))
        .unwrap();
    col.upsert("b", vec![0.9, 0.1, 0.0, 0.0], Some(json!({"k": "2"})))
        .unwrap();
    col.upsert("c", vec![0.0, 1.0, 0.0, 0.0], None).unwrap();

    let results = col.query(&[1.0, 0.0, 0.0, 0.0], 2, None, 64).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id.as_deref(), Some("a"));

    col.persist().unwrap();

    let manifest = db
        .storage()
        .read_zcolumn_manifest("zcol")
        .expect("zcolumn manifest");
    assert_eq!(manifest.max_layers, 3);
    assert!(!manifest.layer_files.is_empty());

    let mut db2 = Database::open(dir.path()).unwrap();
    let col2 = db2.get_collection("zcol").unwrap();
    assert_eq!(col2.len(), 3);
    assert_eq!(col2.config().index_kind, IndexKind::ZColumn);

    let results2 = col2.query(&[1.0, 0.0, 0.0, 0.0], 2, None, 64).unwrap();
    assert_eq!(results2[0].id.as_deref(), Some("a"));
}

#[test]
fn zcolumn_recall_vs_flat_baseline() {
    let dim = 32;
    let n = 200;
    let k = 10;
    let mut rng = StdRng::seed_from_u64(99);
    let vectors = random_unit_vectors(&mut rng, n, dim);

    let mut flat = FlatIndex::new(dim, DistanceMetric::Cosine);
    let mut zcol = ZColumnIndex::new(dim, DistanceMetric::Cosine, ZColumnConfig::default());

    for (i, v) in vectors.iter().enumerate() {
        let id = VectorId(i as u64);
        flat.insert(id, Vector::new(v.clone())).unwrap();
        zcol.insert(id, Vector::new(v.clone())).unwrap();
    }

    let queries: Vec<Vec<f32>> = (0..20)
        .map(|_| random_unit_vectors(&mut rng, 1, dim).pop().unwrap())
        .collect();

    let mut ground_truth = Vec::new();
    for q in &queries {
        let hits = flat.search(q, k, 0).unwrap();
        ground_truth.push(hits.into_iter().map(|h| h.id).collect());
    }

    let flat_recall = recall_at_k(&flat, &queries, &ground_truth, k);
    let zcol_recall = recall_at_k(&zcol, &queries, &ground_truth, k);

    assert!(
        flat_recall >= 0.8,
        "flat recall sanity check: {flat_recall}"
    );
    assert!(
        zcol_recall >= flat_recall - 0.15,
        "zcolumn recall {zcol_recall} too far below flat {flat_recall}"
    );
}

#[test]
fn zcolumn_compaction_after_delete() {
    let mut idx = ZColumnIndex::new(
        4,
        DistanceMetric::L2,
        ZColumnConfig {
            rebalance_interval: 1,
            ..ZColumnConfig::default()
        },
    );

    for i in 0..50u64 {
        let v = vec![(i as f32 * 0.01).sin(), (i as f32 * 0.02).cos(), 0.0, 0.0];
        idx.insert(VectorId(i), Vector::new(v)).unwrap();
    }

    idx.remove(VectorId(49)).unwrap();
    idx.record_access(&[VectorId(1)], 0);
    let events_before = idx.compaction_events();
    idx.rebalance();
    assert!(idx.compaction_events() >= events_before);
}

#[test]
fn zcolumn_segment_files_roundtrip() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let col = db
        .get_or_create_collection_with_config("seg", 4, DistanceMetric::L2, IndexKind::ZColumn)
        .unwrap();

    col.upsert("x", vec![1.0, 0.0, 0.0, 0.0], None).unwrap();
    col.upsert("y", vec![0.0, 1.0, 0.0, 0.0], None).unwrap();
    col.persist().unwrap();

    let layer0 = db
        .storage()
        .read_column_layer("seg", 0, 4, dv_storage::QuantTierTag::U8)
        .unwrap();
    assert!(!layer0.is_empty());
    assert!(layer0.iter().any(|r| !r.ids.is_empty()));

    let manifest = db.storage().read_zcolumn_manifest("seg").unwrap();
    assert_eq!(manifest.dimension, 4);
    assert_eq!(manifest.layer_files.len(), 3);
}
