use crate::column_format::ZColumnManifest;
use crate::column_segment::{ColumnCellRecord, ColumnSegment};
use crate::segment::VectorSegment;
use crate::wal::Wal;
use dv_types::{CollectionConfig, Result, TopolseaError, VectorId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CollectionManifest {
    config: CollectionConfig,
    next_id: u64,
    /// Last snapshot sequence; WAL records with seq > snapshot_seq must be replayed.
    #[serde(default)]
    snapshot_seq: u64,
}

/// On-disk layout:
/// `{root}/{collection}/manifest.json`
/// `{root}/{collection}/vectors.bin`
/// `{root}/{collection}/index.bin`
/// `{root}/{collection}/metadata.json`
#[derive(Debug)]
pub struct StorageEngine {
    root: PathBuf,
}

impl StorageEngine {
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn collection_dir(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    pub fn list_collections(&self) -> Result<Vec<String>> {
        let mut names = Vec::new();
        if !self.root.exists() {
            return Ok(names);
        }
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() && entry.path().join("manifest.json").exists() {
                if let Some(name) = entry.file_name().to_str() {
                    names.push(name.to_string());
                }
            }
        }
        names.sort();
        Ok(names)
    }

    pub fn create_collection(&self, config: CollectionConfig) -> Result<()> {
        let dir = self.collection_dir(&config.name);
        if dir.exists() {
            return Err(TopolseaError::CollectionExists(config.name.clone()));
        }
        fs::create_dir_all(&dir)?;
        let manifest = CollectionManifest {
            config: config.clone(),
            next_id: 0,
            snapshot_seq: 0,
        };
        atomic_write_json(dir.join("manifest.json"), &manifest)?;
        atomic_write_json(
            dir.join("metadata.json"),
            &HashMap::<String, serde_json::Value>::new(),
        )?;
        // Create empty WAL header.
        let _ = Wal::open(dir.join("wal.log"))?;
        Ok(())
    }

    pub fn load_config(&self, name: &str) -> Result<CollectionConfig> {
        let manifest: CollectionManifest =
            read_json(self.collection_dir(name).join("manifest.json"))?;
        Ok(manifest.config)
    }

    pub fn collection_exists(&self, name: &str) -> bool {
        self.collection_dir(name).join("manifest.json").exists()
    }

    pub fn delete_collection(&self, name: &str) -> Result<()> {
        let dir = self.collection_dir(name);
        if dir.exists() {
            fs::remove_dir_all(dir)?;
        }
        Ok(())
    }

    pub fn allocate_id(&self, name: &str) -> Result<VectorId> {
        let path = self.collection_dir(name).join("manifest.json");
        let mut manifest: CollectionManifest = read_json(&path)?;
        let id = VectorId(manifest.next_id);
        manifest.next_id += 1;
        atomic_write_json(path, &manifest)?;
        Ok(id)
    }

    /// Allocate an id in-memory without rewriting the manifest (WAL owns durability).
    pub fn peek_allocate_id(&self, name: &str) -> Result<VectorId> {
        let path = self.collection_dir(name).join("manifest.json");
        let manifest: CollectionManifest = read_json(&path)?;
        Ok(VectorId(manifest.next_id))
    }

    pub fn set_next_id(&self, name: &str, next_id: u64) -> Result<()> {
        let path = self.collection_dir(name).join("manifest.json");
        let mut manifest: CollectionManifest = read_json(&path)?;
        if next_id > manifest.next_id {
            manifest.next_id = next_id;
            atomic_write_json(path, &manifest)?;
        }
        Ok(())
    }

    pub fn snapshot_seq(&self, name: &str) -> Result<u64> {
        let manifest: CollectionManifest =
            read_json(self.collection_dir(name).join("manifest.json"))?;
        Ok(manifest.snapshot_seq)
    }

    pub fn set_snapshot_seq(&self, name: &str, seq: u64) -> Result<()> {
        let path = self.collection_dir(name).join("manifest.json");
        let mut manifest: CollectionManifest = read_json(&path)?;
        manifest.snapshot_seq = seq;
        atomic_write_json(path, &manifest)
    }

    pub fn open_wal(&self, name: &str) -> Result<Wal> {
        Wal::open(self.collection_dir(name).join("wal.log"))
    }

    pub fn wal_path(&self, name: &str) -> PathBuf {
        self.collection_dir(name).join("wal.log")
    }

    pub fn write_vectors(&self, name: &str, records: &[(VectorId, &[f32])]) -> Result<()> {
        let config = self.load_config(name)?;
        let seg = VectorSegment::new(
            self.collection_dir(name).join("vectors.bin"),
            config.dimension,
        );
        seg.write_all(records)
    }

    pub fn read_vectors(&self, name: &str) -> Result<Vec<(VectorId, Vec<f32>)>> {
        let config = self.load_config(name)?;
        // Prefer sealed segments when present (B7).
        let seg_dir = self.segments_dir(name);
        if seg_dir.join("manifest.json").exists() {
            let store = crate::SegmentStore::open(seg_dir, config.dimension)?;
            let sealed = store.read_all_mmap()?;
            if !sealed.is_empty() {
                return Ok(sealed);
            }
        }
        let path = self.collection_dir(name).join("vectors.bin");
        if !path.exists() {
            return Ok(Vec::new());
        }
        let seg = VectorSegment::new(path, config.dimension);
        seg.read_all()
    }

    pub fn segments_dir(&self, name: &str) -> PathBuf {
        self.collection_dir(name).join("segments")
    }

    /// Incremental seal: write only vectors not already sealed; mark removals.
    pub fn flush_vector_segments(&self, name: &str, current: &[(VectorId, &[f32])]) -> Result<()> {
        let config = self.load_config(name)?;
        let store = crate::SegmentStore::open(self.segments_dir(name), config.dimension)?;
        let sealed_ids = store.sealed_ids()?;
        let current_ids: std::collections::HashSet<u64> =
            current.iter().map(|(id, _)| id.raw()).collect();

        let new_records: Vec<(VectorId, &[f32])> = current
            .iter()
            .filter(|(id, _)| !sealed_ids.contains(&id.raw()))
            .map(|(id, v)| (*id, *v))
            .collect();
        store.seal_segment(&new_records)?;

        let deleted: Vec<VectorId> = sealed_ids
            .difference(&current_ids)
            .copied()
            .map(VectorId)
            .collect();
        store.mark_deleted(&deleted)?;

        // Drop legacy monolithic vectors.bin once sealed segments are authoritative.
        let legacy = self.collection_dir(name).join("vectors.bin");
        if store.manifest_path().exists() && legacy.exists() {
            let _ = fs::remove_file(legacy);
        }
        Ok(())
    }

    pub fn compact_vector_segments(&self, name: &str) -> Result<()> {
        let config = self.load_config(name)?;
        let store = crate::SegmentStore::open(self.segments_dir(name), config.dimension)?;
        store.compact()
    }

    pub fn segment_stats(&self, name: &str) -> Result<serde_json::Value> {
        let config = self.load_config(name)?;
        let store = crate::SegmentStore::open(self.segments_dir(name), config.dimension)?;
        let manifest = store.load_manifest()?;
        Ok(serde_json::json!({
            "segments": manifest.segments.len(),
            "deleted": manifest.deleted_ids.len(),
            "next_segment_id": manifest.next_segment_id,
            "dimension": manifest.dimension,
        }))
    }

    pub fn write_sparse_blob(&self, name: &str, data: &[u8]) -> Result<()> {
        atomic_write_bytes(self.collection_dir(name).join("sparse.bin"), data)
    }

    pub fn read_sparse_blob(&self, name: &str) -> Result<Vec<u8>> {
        let path = self.collection_dir(name).join("sparse.bin");
        if !path.exists() {
            return Ok(Vec::new());
        }
        Ok(fs::read(path)?)
    }

    pub fn write_index_blob(&self, name: &str, data: &[u8]) -> Result<()> {
        atomic_write_bytes(self.collection_dir(name).join("index.bin"), data)
    }

    pub fn read_index_blob(&self, name: &str) -> Result<Vec<u8>> {
        let path = self.collection_dir(name).join("index.bin");
        if !path.exists() {
            return Ok(Vec::new());
        }
        Ok(fs::read(path)?)
    }

    pub fn write_metadata_map(
        &self,
        name: &str,
        metadata: &HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        atomic_write_json(self.collection_dir(name).join("metadata.json"), metadata)
    }

    pub fn read_metadata_map(&self, name: &str) -> Result<HashMap<String, serde_json::Value>> {
        let path = self.collection_dir(name).join("metadata.json");
        if !path.exists() {
            return Ok(HashMap::new());
        }
        read_json(path)
    }

    pub fn root_path(&self) -> &Path {
        &self.root
    }

    pub fn columns_dir(&self, name: &str) -> PathBuf {
        self.collection_dir(name).join("columns")
    }

    pub fn write_zcolumn_manifest(&self, name: &str, manifest: &ZColumnManifest) -> Result<()> {
        let dir = self.columns_dir(name);
        fs::create_dir_all(&dir)?;
        atomic_write_json(dir.join("manifest.json"), manifest)
    }

    pub fn read_zcolumn_manifest(&self, name: &str) -> Result<ZColumnManifest> {
        read_json(self.columns_dir(name).join("manifest.json"))
    }

    pub fn write_column_layer(
        &self,
        name: &str,
        layer: u8,
        tier: crate::column_format::QuantTierTag,
        dimension: usize,
        records: &[ColumnCellRecord],
    ) -> Result<()> {
        let dir = self.columns_dir(name);
        fs::create_dir_all(&dir)?;
        let path = dir.join(format!("L{layer}.grid.bin"));
        let seg = ColumnSegment::new(path, dimension, layer, tier);
        seg.write_all(records)
    }

    pub fn read_column_layer(
        &self,
        name: &str,
        layer: u8,
        dimension: usize,
        tier: crate::column_format::QuantTierTag,
    ) -> Result<Vec<ColumnCellRecord>> {
        let path = self.columns_dir(name).join(format!("L{layer}.grid.bin"));
        if !path.exists() {
            return Ok(Vec::new());
        }
        let seg = ColumnSegment::new(path, dimension, layer, tier);
        seg.read_all()
    }

    pub fn at_root(root: PathBuf) -> Self {
        Self { root }
    }

    fn shards_dir(&self) -> PathBuf {
        self.root.join("__shards__")
    }

    fn shard_manifest_path(&self, logical_name: &str) -> PathBuf {
        self.shards_dir().join(format!("{logical_name}.json"))
    }

    pub fn shard_manifest_exists(&self, logical_name: &str) -> bool {
        self.shard_manifest_path(logical_name).exists()
    }

    pub fn write_shard_manifest(
        &self,
        manifest: &crate::shard_format::ShardManifest,
    ) -> Result<()> {
        let dir = self.shards_dir();
        fs::create_dir_all(&dir)?;
        atomic_write_json(self.shard_manifest_path(&manifest.logical_name), manifest)
    }

    pub fn read_shard_manifest(
        &self,
        logical_name: &str,
    ) -> Result<crate::shard_format::ShardManifest> {
        read_json(self.shard_manifest_path(logical_name))
    }

    pub fn list_shard_manifests(&self) -> Result<Vec<crate::shard_format::ShardManifest>> {
        let dir = self.shards_dir();
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            if entry.path().extension().and_then(|e| e.to_str()) == Some("json") {
                let manifest: crate::shard_format::ShardManifest = read_json(entry.path())?;
                out.push(manifest);
            }
        }
        out.sort_by(|a, b| a.logical_name.cmp(&b.logical_name));
        Ok(out)
    }

    pub fn delete_shard_manifest(&self, logical_name: &str) -> Result<()> {
        let path = self.shard_manifest_path(logical_name);
        if path.exists() {
            fs::remove_file(path)?;
        }
        let routing = self.shard_routing_path(logical_name);
        if routing.exists() {
            fs::remove_file(routing)?;
        }
        let cluster = self.shard_cluster_path(logical_name);
        if cluster.exists() {
            fs::remove_file(cluster)?;
        }
        Ok(())
    }

    fn shard_routing_path(&self, logical_name: &str) -> PathBuf {
        self.shards_dir()
            .join(format!("{logical_name}.routing.json"))
    }

    pub fn write_shard_routing(
        &self,
        logical_name: &str,
        index: &crate::shard_format::ShardRoutingIndex,
    ) -> Result<()> {
        fs::create_dir_all(self.shards_dir())?;
        atomic_write_json(self.shard_routing_path(logical_name), index)
    }

    pub fn read_shard_routing(
        &self,
        logical_name: &str,
    ) -> Result<crate::shard_format::ShardRoutingIndex> {
        let path = self.shard_routing_path(logical_name);
        if !path.exists() {
            return Ok(crate::shard_format::ShardRoutingIndex::default());
        }
        read_json(path)
    }

    fn shard_cluster_path(&self, logical_name: &str) -> PathBuf {
        self.shards_dir()
            .join(format!("{logical_name}.cluster.json"))
    }

    pub fn write_shard_cluster(
        &self,
        logical_name: &str,
        config: &crate::shard_format::ShardClusterConfig,
    ) -> Result<()> {
        fs::create_dir_all(self.shards_dir())?;
        atomic_write_json(self.shard_cluster_path(logical_name), config)
    }

    pub fn read_shard_cluster(
        &self,
        logical_name: &str,
    ) -> Result<crate::shard_format::ShardClusterConfig> {
        let path = self.shard_cluster_path(logical_name);
        if !path.exists() {
            return Ok(crate::shard_format::ShardClusterConfig::default());
        }
        read_json(path)
    }
}

fn atomic_write_json<T: Serialize>(path: impl AsRef<Path>, value: &T) -> Result<()> {
    let data = serde_json::to_vec_pretty(value)?;
    atomic_write_bytes(path, &data)
}

fn atomic_write_bytes(path: impl AsRef<Path>, data: &[u8]) -> Result<()> {
    let path = path.as_ref();
    let tmp = path.with_extension("tmp");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&tmp, data)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: impl AsRef<Path>) -> Result<T> {
    let data = fs::read(path)?;
    Ok(serde_json::from_slice(&data)?)
}
