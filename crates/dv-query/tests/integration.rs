use dv_query::Database;
use dv_types::DistanceMetric;
use serde_json::json;
use tempfile::tempdir;

#[test]
fn end_to_end_hnsw_collection() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let col = db
        .get_or_create_collection("test", 4, DistanceMetric::Cosine)
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

    // Reopen
    let mut db2 = Database::open(dir.path()).unwrap();
    let col2 = db2.get_collection("test").unwrap();
    assert_eq!(col2.len(), 3);
}

#[test]
fn metadata_filter() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let col = db
        .get_or_create_collection("f", 2, DistanceMetric::L2)
        .unwrap();

    col.upsert("x", vec![0.0, 0.0], Some(json!({"tag": "alpha"})))
        .unwrap();
    col.upsert("y", vec![0.1, 0.0], Some(json!({"tag": "beta"})))
        .unwrap();

    use dv_metadata::Filter;
    let filter = Filter::from_json(&json!({"tag": "alpha"})).unwrap();
    let results = col.query(&[0.0, 0.0], 5, Some(&filter), 0).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id.as_deref(), Some("x"));
}
