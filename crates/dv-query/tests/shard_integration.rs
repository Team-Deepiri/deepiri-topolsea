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
    // Go/no-go target: within 15% of HNSW recall (relaxed smoke; see 10k gate for 98%).
    assert!(
        zcol_recall >= hnsw_recall * 0.85,
        "zcolumn recall {zcol_recall} too far below hnsw {hnsw_recall}"
    );
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
        let got: std::collections::HashSet<_> = results.iter().map(|h| h.id).collect();
        let truth_set: std::collections::HashSet<_> = truth.iter().take(k).copied().collect();
        let overlap = got.intersection(&truth_set).count();
        total += overlap as f32 / k as f32;
    }
    total / queries.len().max(1) as f32
}

#[test]
fn zcolumn_recall_within_hnsw_band_at_10k() {
    let dim = 128;
    let n = 10_000;
    let k = 10;
    let ef = 128;
    let mut rng = StdRng::seed_from_u64(42);
    let vectors = random_unit_vectors(&mut rng, n, dim);

    let mut flat = FlatIndex::new(dim, DistanceMetric::Cosine);
    let mut hnsw = HnswIndex::new(dim, DistanceMetric::Cosine, HnswConfig::default());
    let zcfg = ZColumnConfig::default();
    let mut zcol = ZColumnIndex::new(dim, DistanceMetric::Cosine, zcfg);

    for (i, v) in vectors.iter().enumerate() {
        let id = VectorId(i as u64);
        flat.insert(id, Vector::new(v.clone())).unwrap();
        hnsw.insert(id, Vector::new(v.clone())).unwrap();
        zcol.insert(id, Vector::new(v.clone())).unwrap();
    }

    let queries: Vec<Vec<f32>> = random_unit_vectors(&mut rng, 50, dim);
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

    let hnsw_recall = recall_at_k_mean(&hnsw, &queries, &ground, k, ef);
    let zcol_recall = recall_at_k_mean(&zcol, &queries, &ground, k, ef);

    assert!(
        hnsw_recall > 0.7,
        "hnsw recall sanity check failed: {hnsw_recall}"
    );
    // Protocol go/no-go: recall@10 within 2% of HNSW at 10k/128d.
    assert!(
        zcol_recall >= hnsw_recall * 0.98,
        "zcolumn recall {zcol_recall} below 98% of hnsw {hnsw_recall}"
    );
}

#[test]
fn http_shard_server_responds() {
    use dv_query::{ShardQueryServer, ShardServerConfig};
    use dv_shard_remote::{ShardQueryClient, ShardQueryRequest};
    use dv_types::CollectionConfig;

    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    db.create_collection(
        CollectionConfig::new("fast", 4, DistanceMetric::Cosine).with_flat_index(),
    )
    .unwrap();
    db.get_collection("fast")
        .unwrap()
        .upsert("a", vec![1.0, 0.0, 0.0, 0.0], None)
        .unwrap();
    db.persist_all().unwrap();
    drop(db);

    let server = ShardQueryServer::start(ShardServerConfig {
        data_dir: dir.path().to_path_buf(),
        collection: "fast".into(),
        bind_addr: "127.0.0.1:0".into(),
    })
    .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(50));

    let client = ShardQueryClient::new(5_000);
    let resp = client
        .query(
            &server.base_url(),
            &ShardQueryRequest {
                vector: vec![1.0, 0.0, 0.0, 0.0],
                top_k: 1,
                ef: 0,
            },
        )
        .expect("shard HTTP query");
    assert!(!resp.hits.is_empty());
    server.shutdown();
}

#[test]
fn distributed_shard_fanout_via_http() {
    use dv_query::{ShardQueryServer, ShardServerConfig};

    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    db.create_sharded_collection("remote", 2, 8, DistanceMetric::Cosine, IndexKind::ZColumn)
        .unwrap();

    for i in 0..24u64 {
        let v: Vec<f32> = (0..8)
            .map(|d| ((i as f32 + d as f32) * 0.11).sin())
            .collect();
        db.upsert_sharded("remote", &format!("id{i}"), v, None)
            .unwrap();
    }
    db.persist_all().unwrap();

    let manifest = db.storage().read_shard_manifest("remote").unwrap();
    let s0 = manifest.physical_name(0);
    let s1 = manifest.physical_name(1);
    drop(db);

    let server0 = ShardQueryServer::start(ShardServerConfig {
        data_dir: dir.path().to_path_buf(),
        collection: s0.clone(),
        bind_addr: "127.0.0.1:0".into(),
    })
    .unwrap();
    let server1 = ShardQueryServer::start(ShardServerConfig {
        data_dir: dir.path().to_path_buf(),
        collection: s1.clone(),
        bind_addr: "127.0.0.1:0".into(),
    })
    .unwrap();

    let mut db = Database::open(dir.path()).unwrap();
    db.set_shard_endpoint("remote", 0, server0.base_url()).unwrap();
    db.set_shard_endpoint("remote", 1, server1.base_url()).unwrap();

    let q = vec![0.1; 8];
    let hits = db.query_sharded("remote", &q, 5, None, 64).unwrap();
    assert!(!hits.is_empty());

    server0.shutdown();
    server1.shutdown();
}
