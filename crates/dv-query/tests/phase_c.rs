//! Phase C integration: snapshots, namespaces, membership, replica policy.

use dv_query::{qualify_collection, Database, DEFAULT_NAMESPACE};
use dv_storage::{ClusterMembership, ClusterNode, NodeRole};
use dv_types::{CollectionConfig, DistanceMetric};
use tempfile::tempdir;

#[test]
fn namespace_qualifies_paths() {
    assert_eq!(qualify_collection(DEFAULT_NAMESPACE, "demo"), "demo");
    assert_eq!(qualify_collection("acme", "demo"), "acme/demo");
}

#[test]
fn snapshot_roundtrip() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    db.create_collection(CollectionConfig::new("snapcol", 2, DistanceMetric::L2).with_flat_index())
        .unwrap();
    {
        let col = db.get_collection("snapcol").unwrap();
        col.write().upsert("a", vec![1.0, 0.0], None).unwrap();
        col.write().persist().unwrap();
    }
    db.create_snapshot("s1").unwrap();
    assert!(db.list_snapshots().unwrap().contains(&"s1".into()));
    let meta = db.snapshot_meta("s1").unwrap();
    assert_eq!(meta["name"], "s1");
    assert!(meta["collections"]
        .as_array()
        .unwrap()
        .iter()
        .any(|c| c == "snapcol"));

    // Mutate then restore.
    {
        let col = db.get_collection("snapcol").unwrap();
        col.write().delete("a").unwrap();
        col.write().persist().unwrap();
        assert_eq!(col.read().len(), 0);
    }
    db.restore_snapshot("s1").unwrap();
    let col = db.get_collection("snapcol").unwrap();
    assert_eq!(col.read().len(), 1);
}

#[test]
fn partial_snapshot_preserves_unrelated_collections() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    db.create_collection(CollectionConfig::new("keep", 2, DistanceMetric::L2).with_flat_index())
        .unwrap();
    db.create_collection(
        CollectionConfig::new("snap_only", 2, DistanceMetric::L2).with_flat_index(),
    )
    .unwrap();
    {
        db.get_collection("keep")
            .unwrap()
            .write()
            .upsert("k", vec![1.0, 0.0], None)
            .unwrap();
        db.get_collection("snap_only")
            .unwrap()
            .write()
            .upsert("s", vec![0.0, 1.0], None)
            .unwrap();
        db.persist_all().unwrap();
    }
    db.create_snapshot_opts("partial", Some(&["snap_only".into()]))
        .unwrap();
    {
        db.get_collection("snap_only")
            .unwrap()
            .write()
            .delete("s")
            .unwrap();
        db.get_collection("keep")
            .unwrap()
            .write()
            .upsert("k2", vec![1.0, 1.0], None)
            .unwrap();
        db.persist_all().unwrap();
    }
    // Scoped restore: snap_only rolled back; keep untouched (still has k2).
    db.restore_snapshot_opts("partial", false).unwrap();
    assert_eq!(db.get_collection("snap_only").unwrap().read().len(), 1);
    assert_eq!(db.get_collection("keep").unwrap().read().len(), 2);
}

#[test]
fn membership_persists_and_heartbeat() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    db.set_membership(ClusterMembership {
        nodes: vec![ClusterNode {
            id: "n1".into(),
            advertise_url: "http://127.0.0.1:6333".into(),
            role: NodeRole::Data,
            last_heartbeat_ms: 0,
            healthy: true,
        }],
        generation: 1,
    })
    .unwrap();
    let m = db.membership().unwrap();
    assert_eq!(m.nodes.len(), 1);
    assert_eq!(m.nodes[0].id, "n1");

    db.heartbeat_node("n1", true).unwrap();
    let m = db.membership().unwrap();
    assert!(m.nodes[0].last_heartbeat_ms > 0);
    assert!(m.nodes[0].healthy);
}

#[test]
fn namespaced_collection_isolation() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let a = qualify_collection("acme", "docs");
    let b = qualify_collection("beta", "docs");
    db.create_collection(CollectionConfig::new(&a, 2, DistanceMetric::L2).with_flat_index())
        .unwrap();
    db.create_collection(CollectionConfig::new(&b, 2, DistanceMetric::L2).with_flat_index())
        .unwrap();
    db.get_collection(&a)
        .unwrap()
        .write()
        .upsert("x", vec![1.0, 0.0], None)
        .unwrap();
    assert_eq!(db.get_collection(&a).unwrap().read().len(), 1);
    assert_eq!(db.get_collection(&b).unwrap().read().len(), 0);
}

#[test]
fn wal_lag_samples_pending_records() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    db.create_collection(CollectionConfig::new("lag", 2, DistanceMetric::L2).with_flat_index())
        .unwrap();
    {
        let col = db.get_collection("lag").unwrap();
        col.write().upsert("a", vec![1.0, 0.0], None).unwrap();
    }
    let lag = db.sample_wal_lag().unwrap();
    assert!(lag >= 1);
    db.persist_all().unwrap();
    assert_eq!(db.sample_wal_lag().unwrap(), 0);
}
