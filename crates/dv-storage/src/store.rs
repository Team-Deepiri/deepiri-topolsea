use crate::column_format::ZColumnManifest;
use crate::column_segment::{ColumnCellRecord, ColumnSegment};
use crate::segment::VectorSegment;
use dv_types::{CollectionConfig, Result, TopolseaError, VectorId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CollectionManifest {
    config: CollectionConfig,
    next_id: u64,
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
        };
        write_json(dir.join("manifest.json"), &manifest)?;
        write_json(
            dir.join("metadata.json"),
            &HashMap::<String, serde_json::Value>::new(),
        )?;
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
        write_json(path, &manifest)?;
        Ok(id)
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
        let path = self.collection_dir(name).join("vectors.bin");
        if !path.exists() {
            return Ok(Vec::new());
        }
        let seg = VectorSegment::new(path, config.dimension);
        seg.read_all()
    }

    pub fn write_index_blob(&self, name: &str, data: &[u8]) -> Result<()> {
        fs::write(self.collection_dir(name).join("index.bin"), data)?;
        Ok(())
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
        write_json(self.collection_dir(name).join("metadata.json"), metadata)
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
        write_json(dir.join("manifest.json"), manifest)
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
        write_json(self.shard_manifest_path(&manifest.logical_name), manifest)
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
        write_json(self.shard_routing_path(logical_name), index)
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
}

fn write_json<T: Serialize>(path: impl AsRef<Path>, value: &T) -> Result<()> {
    let data = serde_json::to_vec_pretty(value)?;
    fs::write(path, data)?;
    Ok(())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: impl AsRef<Path>) -> Result<T> {
    let data = fs::read(path)?;
    Ok(serde_json::from_slice(&data)?)
}
