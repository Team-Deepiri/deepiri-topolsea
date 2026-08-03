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

    col.write()
        .upsert("a", vec![1.0, 0.0, 0.0, 0.0], Some(json!({"k": "1"})))
        .unwrap();
    col.write()
        .upsert("b", vec![0.9, 0.1, 0.0, 0.0], Some(json!({"k": "2"})))
        .unwrap();
    col.write()
        .upsert("c", vec![0.0, 1.0, 0.0, 0.0], None)
        .unwrap();

    let results = col
        .read()
        .query(&[1.0, 0.0, 0.0, 0.0], 2, None, 64)
        .unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id.as_deref(), Some("a"));

    col.write().persist().unwrap();

    // Reopen
    let mut db2 = Database::open(dir.path()).unwrap();
    let col2 = db2.get_collection("test").unwrap();
    assert_eq!(col2.read().len(), 3);
}

#[test]
fn metadata_filter() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let col = db
        .get_or_create_collection("f", 2, DistanceMetric::L2)
        .unwrap();

    col.write()
        .upsert("x", vec![0.0, 0.0], Some(json!({"tag": "alpha"})))
        .unwrap();
    col.write()
        .upsert("y", vec![0.1, 0.0], Some(json!({"tag": "beta"})))
        .unwrap();

    use dv_metadata::Filter;
    let filter = Filter::from_json(&json!({"tag": "alpha"})).unwrap();
    let results = col.read().query(&[0.0, 0.0], 5, Some(&filter), 0).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id.as_deref(), Some("x"));
}

#[test]
fn wal_recovers_without_persist() {
    let dir = tempdir().unwrap();
    {
        let mut db = Database::open(dir.path()).unwrap();
        let col = db
            .get_or_create_collection("wal", 2, DistanceMetric::L2)
            .unwrap();
        col.write()
            .upsert("a", vec![1.0, 0.0], Some(json!({"tag": "keep"})))
            .unwrap();
        col.write()
            .upsert("b", vec![0.0, 1.0], Some(json!({"tag": "drop"})))
            .unwrap();
        // No persist — durability is WAL only.
    }
    let mut db2 = Database::open(dir.path()).unwrap();
    let col2 = db2.get_collection("wal").unwrap();
    assert_eq!(col2.read().len(), 2);
    let filter = dv_metadata::Filter::from_json(&json!({"tag": "keep"})).unwrap();
    let hits = col2.read().query(&[1.0, 0.0], 5, Some(&filter), 0).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id.as_deref(), Some("a"));
}

#[test]
fn filter_ops_ne_gt_in() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let col = db
        .get_or_create_collection("ops", 2, DistanceMetric::L2)
        .unwrap();
    col.write()
        .upsert("a", vec![0.0, 0.0], Some(json!({"tag": "x", "n": 1})))
        .unwrap();
    col.write()
        .upsert("b", vec![0.1, 0.0], Some(json!({"tag": "y", "n": 10})))
        .unwrap();
    col.write()
        .upsert("c", vec![0.2, 0.0], Some(json!({"tag": "z", "n": 5})))
        .unwrap();

    use dv_metadata::Filter;
    let ne = Filter::from_json(&json!({"tag": {"$ne": "x"}})).unwrap();
    assert_eq!(
        col.read()
            .query(&[0.0, 0.0], 10, Some(&ne), 0)
            .unwrap()
            .len(),
        2
    );

    let gt = Filter::from_json(&json!({"n": {"$gt": 5}})).unwrap();
    let hits = col.read().query(&[0.0, 0.0], 10, Some(&gt), 0).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id.as_deref(), Some("b"));

    let inn = Filter::from_json(&json!({"tag": {"$in": ["x", "z"]}})).unwrap();
    assert_eq!(
        col.read()
            .query(&[0.0, 0.0], 10, Some(&inn), 0)
            .unwrap()
            .len(),
        2
    );

    let and = Filter::from_json(&json!({
        "$and": [{"tag": {"$ne": "x"}}, {"n": {"$gte": 5}}]
    }))
    .unwrap();
    let hits = col.read().query(&[0.0, 0.0], 10, Some(&and), 0).unwrap();
    assert_eq!(hits.len(), 2);

    let or = Filter::from_json(&json!({
        "$or": [{"tag": "x"}, {"n": {"$gt": 9}}]
    }))
    .unwrap();
    assert_eq!(
        col.read()
            .query(&[0.0, 0.0], 10, Some(&or), 0)
            .unwrap()
            .len(),
        2
    );

    let multi = Filter::from_json(&json!({"tag": "y", "n": {"$lte": 10}})).unwrap();
    let hits = col.read().query(&[0.0, 0.0], 10, Some(&multi), 0).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id.as_deref(), Some("b"));
}

#[test]
fn concurrent_readers_with_writer() {
    use std::sync::Arc;
    use std::thread;

    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let col = db
        .get_or_create_collection("conc", 4, DistanceMetric::L2)
        .unwrap();
    for i in 0..100 {
        col.write()
            .upsert(
                &format!("id{i}"),
                vec![i as f32, 0.0, 0.0, 0.0],
                Some(json!({"bucket": i % 10})),
            )
            .unwrap();
    }

    let col_r = Arc::clone(&col);
    let readers: Vec<_> = (0..8)
        .map(|_| {
            let c = Arc::clone(&col_r);
            thread::spawn(move || {
                for _ in 0..50 {
                    let _ = c.read().query(&[1.0, 0.0, 0.0, 0.0], 5, None, 32).unwrap();
                }
            })
        })
        .collect();

    let writer = {
        let c = Arc::clone(&col);
        thread::spawn(move || {
            for i in 100..120 {
                c.write()
                    .upsert(&format!("id{i}"), vec![i as f32, 0.0, 0.0, 0.0], None)
                    .unwrap();
            }
        })
    };

    for t in readers {
        t.join().unwrap();
    }
    writer.join().unwrap();
    assert!(col.read().len() >= 100);
}

#[test]
fn payload_aware_filter_selectivity() {
    for (pct, tag_every) in [(1usize, 100usize), (10, 10), (50, 2)] {
        let dir = tempdir().unwrap();
        let mut db = Database::open(dir.path()).unwrap();
        let col = db
            .get_or_create_collection(&format!("sel{pct}"), 4, DistanceMetric::L2)
            .unwrap();
        let n = 200usize;
        for i in 0..n {
            let tag = if i % tag_every == 0 { "keep" } else { "drop" };
            col.write()
                .upsert(
                    &format!("v{i}"),
                    vec![i as f32, 0.0, 0.0, 0.0],
                    Some(json!({"tag": tag})),
                )
                .unwrap();
        }
        let filter = dv_metadata::Filter::from_json(&json!({"tag": "keep"})).unwrap();
        let q = vec![0.0, 0.0, 0.0, 0.0];
        let hits = col.read().query(&q, 20, Some(&filter), 64).unwrap();
        assert!(!hits.is_empty(), "selectivity {pct}% returned no hits");
        assert!(
            hits.iter().all(|h| {
                h.metadata
                    .as_ref()
                    .and_then(|m| m.get("tag"))
                    .and_then(|v| v.as_str())
                    == Some("keep")
            }),
            "selectivity {pct}% leaked non-matching tags"
        );
        // Recall vs filtered flat: every keep id nearer than drop for query at origin
        // among keep set — at least the nearest keep vector should appear.
        let expected_nearest = "v0";
        assert!(
            hits.iter()
                .any(|h| h.id.as_deref() == Some(expected_nearest)),
            "selectivity {pct}% missed nearest eligible {expected_nearest}"
        );
    }
}
