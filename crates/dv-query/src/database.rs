use dv_storage::{ShardManifest, StorageEngine};
use dv_types::{CollectionConfig, DistanceMetric, IndexKind, Result, TopolseaError, VectorId};
use std::collections::HashMap;
use std::path::Path;

use super::collection::Collection;
use super::query::QueryResult;
use super::shard::{is_physical_shard_collection, merge_shard_results, FractalShardRouter};
use dv_metadata::Filter;
use serde_json::Value;

fn open_collection(storage: &StorageEngine, config: CollectionConfig) -> Result<Collection> {
    Collection::open(
        StorageEngine::at_root(storage.root_path().to_path_buf()),
        config,
    )
}

/// Top-level database handle managing multiple collections on disk.
pub struct Database {
    storage: StorageEngine,
    collections: HashMap<String, Collection>,
}

impl Database {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let storage = StorageEngine::new(path)?;
        Ok(Self {
            storage,
            collections: HashMap::new(),
        })
    }

    pub fn storage(&self) -> &StorageEngine {
        &self.storage
    }

    pub fn list_collections(&self) -> Result<Vec<String>> {
        Ok(self
            .storage
            .list_collections()?
            .into_iter()
            .filter(|n| !is_physical_shard_collection(n))
            .collect())
    }

    pub fn list_sharded_collections(&self) -> Result<Vec<ShardManifest>> {
        self.storage.list_shard_manifests()
    }

    pub fn create_collection(&mut self, config: CollectionConfig) -> Result<&mut Collection> {
        let name = config.name.clone();
        if self.collections.contains_key(&name) || self.storage.collection_exists(&name) {
            return Err(TopolseaError::CollectionExists(name));
        }
        self.storage.create_collection(config.clone())?;
        let col = open_collection(&self.storage, config)?;
        self.collections.insert(name.clone(), col);
        Ok(self.collections.get_mut(&name).unwrap())
    }

    /// Create a logically sharded collection backed by `num_shards` physical collections.
    pub fn create_sharded_collection(
        &mut self,
        name: &str,
        num_shards: usize,
        dimension: usize,
        metric: DistanceMetric,
        index_kind: IndexKind,
    ) -> Result<()> {
        if num_shards == 0 {
            return Err(TopolseaError::InvalidConfig(
                "num_shards must be >= 1".into(),
            ));
        }
        if self.storage.shard_manifest_exists(name) || self.storage.collection_exists(name) {
            return Err(TopolseaError::CollectionExists(name.to_string()));
        }

        let mut config = CollectionConfig::new(name, dimension, metric);
        config.index_kind = index_kind;
        if index_kind == IndexKind::ZColumn {
            config = config.with_zcolumn_index();
        } else if index_kind == IndexKind::Flat {
            config = config.with_flat_index();
        }

        let manifest = ShardManifest::new(name, num_shards, &config);
        self.storage.write_shard_manifest(&manifest)?;

        for shard_id in 0..num_shards {
            let physical = manifest.physical_name(shard_id);
            let mut shard_config = CollectionConfig::new(&physical, dimension, metric);
            shard_config.index_kind = index_kind;
            shard_config.zcolumn = config.zcolumn.clone();
            if index_kind == IndexKind::ZColumn {
                shard_config = shard_config.with_zcolumn_index();
            } else if index_kind == IndexKind::Flat {
                shard_config = shard_config.with_flat_index();
            }
            self.create_collection(shard_config)?;
        }
        Ok(())
    }

    pub fn is_sharded(&self, name: &str) -> bool {
        self.storage.shard_manifest_exists(name)
    }

    pub fn upsert_sharded(
        &mut self,
        logical_name: &str,
        external_id: &str,
        vector: Vec<f32>,
        metadata: Option<Value>,
    ) -> Result<VectorId> {
        let manifest = self.storage.read_shard_manifest(logical_name)?;
        let shard = FractalShardRouter::route_vector(&manifest, &vector);
        let physical = manifest.physical_name(shard);
        let col = self.get_collection(&physical)?;
        col.upsert(external_id, vector, metadata)
    }

    pub fn query_sharded(
        &mut self,
        logical_name: &str,
        query_vector: &[f32],
        top_k: usize,
        filter: Option<&Filter>,
        ef: usize,
    ) -> Result<Vec<QueryResult>> {
        let manifest = self.storage.read_shard_manifest(logical_name)?;
        let mut merged = Vec::new();
        for shard_id in 0..manifest.num_shards {
            let physical = manifest.physical_name(shard_id);
            let col = self.get_collection(&physical)?;
            let mut partial = col.query(query_vector, top_k, filter, ef)?;
            merged.append(&mut partial);
        }
        Ok(merge_shard_results(merged, top_k))
    }

    pub fn query_sharded_batch(
        &mut self,
        logical_name: &str,
        query_vectors: &[&[f32]],
        top_k: usize,
        filter: Option<&Filter>,
        ef: usize,
    ) -> Result<Vec<Vec<QueryResult>>> {
        query_vectors
            .iter()
            .map(|q| self.query_sharded(logical_name, q, top_k, filter, ef))
            .collect()
    }

    pub fn sharded_vector_count(&mut self, logical_name: &str) -> Result<usize> {
        let manifest = self.storage.read_shard_manifest(logical_name)?;
        let mut total = 0usize;
        for shard_id in 0..manifest.num_shards {
            let physical = manifest.physical_name(shard_id);
            total += self.get_collection(&physical)?.len();
        }
        Ok(total)
    }

    pub fn delete_sharded_collection(&mut self, logical_name: &str) -> Result<()> {
        let manifest = self.storage.read_shard_manifest(logical_name)?;
        for shard_id in 0..manifest.num_shards {
            let physical = manifest.physical_name(shard_id);
            self.collections.remove(&physical);
            self.storage.delete_collection(&physical)?;
        }
        self.storage.delete_shard_manifest(logical_name)?;
        Ok(())
    }

    pub fn get_or_create_collection(
        &mut self,
        name: &str,
        dimension: usize,
        metric: DistanceMetric,
    ) -> Result<&mut Collection> {
        self.get_or_create_collection_with_config(name, dimension, metric, IndexKind::Hnsw)
    }

    pub fn get_or_create_collection_with_config(
        &mut self,
        name: &str,
        dimension: usize,
        metric: DistanceMetric,
        index_kind: IndexKind,
    ) -> Result<&mut Collection> {
        if self.storage.shard_manifest_exists(name) {
            return Err(TopolseaError::InvalidConfig(format!(
                "'{name}' is a sharded logical collection — use upsert_sharded/query_sharded"
            )));
        }
        if !self.collections.contains_key(name) {
            if self.storage.collection_exists(name) {
                let config = self.storage.load_config(name)?;
                let col = open_collection(&self.storage, config)?;
                self.collections.insert(name.to_string(), col);
            } else {
                let mut config = CollectionConfig::new(name, dimension, metric);
                config.index_kind = index_kind;
                if index_kind == IndexKind::ZColumn {
                    config = config.with_zcolumn_index();
                } else if index_kind == IndexKind::Flat {
                    config = config.with_flat_index();
                }
                return self.create_collection(config);
            }
        }
        Ok(self.collections.get_mut(name).unwrap())
    }

    pub fn get_collection(&mut self, name: &str) -> Result<&mut Collection> {
        if self.storage.shard_manifest_exists(name) {
            return Err(TopolseaError::InvalidConfig(format!(
                "'{name}' is a sharded logical collection — use shard physical names or sharded APIs"
            )));
        }
        if !self.collections.contains_key(name) {
            if !self.storage.collection_exists(name) {
                return Err(TopolseaError::CollectionNotFound(name.to_string()));
            }
            let config = self.storage.load_config(name)?;
            let col = open_collection(&self.storage, config)?;
            self.collections.insert(name.to_string(), col);
        }
        Ok(self.collections.get_mut(name).unwrap())
    }

    pub fn delete_collection(&mut self, name: &str) -> Result<()> {
        if self.storage.shard_manifest_exists(name) {
            return self.delete_sharded_collection(name);
        }
        self.collections.remove(name);
        self.storage.delete_collection(name)
    }

    pub fn persist_all(&mut self) -> Result<()> {
        for col in self.collections.values_mut() {
            col.persist()?;
        }
        Ok(())
    }
}
