//! Phase B integration: hybrid RRF, segmented storage, IVF.

use dv_query::Database;
use dv_types::{CollectionConfig, DistanceMetric, IndexKind, IvfConfig};
use serde_json::json;
use std::fs;
use tempfile::tempdir;

#[test]
fn hybrid_rrf_surfaces_text_relevant_doc() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let mut cfg = CollectionConfig::new("hybrid", 4, DistanceMetric::L2);
    cfg.index_kind = IndexKind::Flat;
    db.create_collection(cfg).unwrap();
    let col = db.get_collection("hybrid").unwrap();

    {
        let mut g = col.write();
        g.upsert_with_text(
            "dense_only",
            vec![1.0, 0.0, 0.0, 0.0],
            None,
            Some("completely unrelated filler words"),
        )
        .unwrap();
        g.upsert_with_text(
            "text_match",
            vec![0.2, 0.8, 0.0, 0.0],
            None,
            Some("quantum topology fractal column search"),
        )
        .unwrap();
        g.upsert_with_text(
            "both",
            vec![0.9, 0.1, 0.0, 0.0],
            None,
            Some("quantum search over fractal topology"),
        )
        .unwrap();
    }

    let hits = col
        .read()
        .query_hybrid(
            &[1.0, 0.0, 0.0, 0.0],
            "quantum fractal topology",
            2,
            None,
            16,
            None,
        )
        .unwrap();
    assert!(!hits.is_empty());
    let ids: Vec<_> = hits.iter().filter_map(|h| h.id.clone()).collect();
    assert!(
        ids.contains(&"both".to_string()) || ids.contains(&"text_match".to_string()),
        "expected text-relevant ids, got {ids:?}"
    );
}

#[test]
fn segmented_storage_incremental_seal() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let cfg = CollectionConfig::new("segs", 2, DistanceMetric::L2).with_flat_index();
    db.create_collection(cfg).unwrap();
    let col = db.get_collection("segs").unwrap();

    {
        let mut g = col.write();
        g.upsert("a", vec![1.0, 0.0], None).unwrap();
        g.upsert("b", vec![0.0, 1.0], None).unwrap();
        g.persist().unwrap();
    }

    let seg_dir = dir.path().join("segs").join("segments");
    assert!(seg_dir.join("manifest.json").exists());
    let segs_before: Vec<_> = fs::read_dir(&seg_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("seg_"))
        .collect();
    assert_eq!(segs_before.len(), 1);

    {
        let mut g = col.write();
        g.upsert("c", vec![0.5, 0.5], None).unwrap();
        g.persist().unwrap();
    }

    let segs_after: Vec<_> = fs::read_dir(&seg_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("seg_"))
        .collect();
    assert_eq!(
        segs_after.len(),
        2,
        "second persist should seal only the delta"
    );

    // Reopen and recover from segments.
    drop(col);
    drop(db);
    let mut db2 = Database::open(dir.path()).unwrap();
    let col2 = db2.get_collection("segs").unwrap();
    assert_eq!(col2.read().len(), 3);
    let hits = col2.read().query(&[0.5, 0.5], 1, None, 16).unwrap();
    assert_eq!(hits[0].id.as_deref(), Some("c"));
}

#[test]
fn ivf_index_roundtrip() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let mut cfg = CollectionConfig::new("ivf", 4, DistanceMetric::L2).with_ivf_index();
    cfg.ivf = IvfConfig {
        nlist: 8,
        nprobe: 4,
        pq_m: Some(2),
        seed: 3,
        memory_bound: true,
    };
    db.create_collection(cfg).unwrap();
    let col = db.get_collection("ivf").unwrap();
    {
        let mut g = col.write();
        for i in 0..48u64 {
            g.upsert(
                &format!("v{i}"),
                vec![i as f32, 0.0, 1.0, 2.0],
                Some(json!({"i": i})),
            )
            .unwrap();
        }
        g.persist().unwrap();
    }
    let hits = col
        .read()
        .query(&[10.0, 0.0, 1.0, 2.0], 5, None, 0)
        .unwrap();
    assert!(!hits.is_empty());
    let ids: Vec<_> = hits.iter().filter_map(|h| h.id.clone()).collect();
    // IVF+PQ is approximate — expect a near neighbor in the top-5.
    assert!(
        ids.iter().any(|id| {
            id.trim_start_matches('v')
                .parse::<i64>()
                .map(|n| (n - 10).abs() <= 3)
                .unwrap_or(false)
        }),
        "expected neighbor of v10 in {ids:?}"
    );
}

#[test]
fn compact_collapses_segments() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let cfg = CollectionConfig::new("cmp", 2, DistanceMetric::L2).with_flat_index();
    db.create_collection(cfg).unwrap();
    let col = db.get_collection("cmp").unwrap();
    {
        let mut g = col.write();
        g.upsert("a", vec![1.0, 0.0], None).unwrap();
        g.persist().unwrap();
        g.upsert("b", vec![0.0, 1.0], None).unwrap();
        g.persist().unwrap();
        g.delete("a").unwrap();
        g.persist().unwrap();
        let stats = g.segment_stats().unwrap();
        assert!(stats["segments"].as_u64().unwrap() >= 2);
        g.compact_segments().unwrap();
        let after = g.segment_stats().unwrap();
        assert_eq!(after["segments"].as_u64().unwrap(), 1);
        assert_eq!(after["deleted"].as_u64().unwrap(), 0);
    }
}

#[test]
fn linear_fusion_and_sparse_endpoint_path() {
    use dv_query::{FusionMethod, HybridOptions};
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    db.create_collection(CollectionConfig::new("lin", 4, DistanceMetric::L2).with_flat_index())
        .unwrap();
    let col = db.get_collection("lin").unwrap();
    {
        let mut g = col.write();
        g.upsert_with_text("d1", vec![1.0, 0.0, 0.0, 0.0], None, Some("alpha beta"))
            .unwrap();
        g.upsert_with_text("d2", vec![0.0, 1.0, 0.0, 0.0], None, Some("gamma delta"))
            .unwrap();
    }
    let sparse = col.read().query_sparse("alpha", 5, None).unwrap();
    assert_eq!(sparse[0].id.as_deref(), Some("d1"));

    let mut opts = HybridOptions::new(1, 16);
    opts.fusion = FusionMethod::Linear;
    opts.dense_weight = Some(0.0);
    let hits = col
        .read()
        .query_hybrid_opts(&[1.0, 0.0, 0.0, 0.0], "gamma", None, &opts)
        .unwrap();
    assert_eq!(hits[0].id.as_deref(), Some("d2"));
}

#[test]
fn hybrid_filter_and_prefetch_rrf_vs_linear() {
    use dv_metadata::Filter;
    use dv_query::{FusionMethod, HybridOptions};

    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    db.create_collection(CollectionConfig::new("hf", 4, DistanceMetric::L2).with_flat_index())
        .unwrap();
    let col = db.get_collection("hf").unwrap();
    {
        let mut g = col.write();
        g.upsert_with_text(
            "keep",
            vec![1.0, 0.0, 0.0, 0.0],
            Some(json!({"tag": "ok"})),
            Some("quantum fractal"),
        )
        .unwrap();
        g.upsert_with_text(
            "drop",
            vec![0.9, 0.1, 0.0, 0.0],
            Some(json!({"tag": "spam"})),
            Some("quantum fractal"),
        )
        .unwrap();
        g.upsert_with_text(
            "other",
            vec![0.0, 1.0, 0.0, 0.0],
            Some(json!({"tag": "ok"})),
            Some("unrelated filler"),
        )
        .unwrap();
    }

    let filter = Filter::from_json(&json!({"tag": {"$ne": "spam"}})).unwrap();
    let mut rrf = HybridOptions::new(2, 16);
    rrf.prefetch = Some(8);
    let hits = col
        .read()
        .query_hybrid_opts(
            &[1.0, 0.0, 0.0, 0.0],
            "quantum fractal",
            Some(&filter),
            &rrf,
        )
        .unwrap();
    let ids: Vec<_> = hits.iter().filter_map(|h| h.id.clone()).collect();
    assert!(
        !ids.contains(&"drop".to_string()),
        "filter should drop spam: {ids:?}"
    );
    assert!(
        ids.contains(&"keep".to_string()),
        "expected keep in {ids:?}"
    );

    let mut linear = HybridOptions::new(1, 16);
    linear.fusion = FusionMethod::Linear;
    linear.dense_weight = Some(0.0);
    linear.prefetch = Some(4);
    let hits = col
        .read()
        .query_hybrid_opts(
            &[1.0, 0.0, 0.0, 0.0],
            "quantum fractal",
            Some(&filter),
            &linear,
        )
        .unwrap();
    assert_eq!(hits[0].id.as_deref(), Some("keep"));
}
