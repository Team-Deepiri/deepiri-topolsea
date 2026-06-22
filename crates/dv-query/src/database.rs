use dv_storage::StorageEngine;
use dv_types::{CollectionConfig, DistanceMetric, Result, TopolseaError};
use std::collections::HashMap;
use std::path::Path;

use super::collection::Collection;

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
        self.storage.list_collections()
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

    pub fn get_or_create_collection(
        &mut self,
        name: &str,
        dimension: usize,
        metric: DistanceMetric,
    ) -> Result<&mut Collection> {
        if !self.collections.contains_key(name) {
            if self.storage.collection_exists(name) {
                let config = self.storage.load_config(name)?;
                let col = open_collection(&self.storage, config)?;
                self.collections.insert(name.to_string(), col);
            } else {
                let config = CollectionConfig::new(name, dimension, metric);
                return self.create_collection(config);
            }
        }
        Ok(self.collections.get_mut(name).unwrap())
    }

    pub fn get_collection(&mut self, name: &str) -> Result<&mut Collection> {
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
