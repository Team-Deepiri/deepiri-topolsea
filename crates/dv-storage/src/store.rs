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

    pub fn at_root(root: PathBuf) -> Self {
        Self { root }
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
