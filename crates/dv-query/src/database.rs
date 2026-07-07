use dv_storage::{ShardClusterConfig, ShardManifest, StorageEngine};
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
        self.storage
            .write_shard_routing(name, &dv_storage::ShardRoutingIndex::new(1))?;

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
        let id = {
            let col = self.get_collection(&physical)?;
            col.upsert(external_id, vector.clone(), metadata)?
        };

        if manifest.index_kind == IndexKind::ZColumn {
            let key = dv_index_zcolumn::column_key_for_vector(
                manifest.dimension,
                manifest.zcolumn.projection_seed,
                manifest.zcolumn.outer_grid,
                manifest.zcolumn.max_layers,
                manifest.zcolumn.pitch_ratio,
                &vector,
            );
            let mut routing = self.storage.read_shard_routing(logical_name)?;
            routing.record(key, shard as u8);
            self.storage.write_shard_routing(logical_name, &routing)?;
        }

        Ok(id)
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
        let routing = self.storage.read_shard_routing(logical_name)?;
        let cluster = self.storage.read_shard_cluster(logical_name)?;

        let target_shards = Self::resolve_target_shards(&manifest, &routing, query_vector);

        let remote_targets: Vec<_> = dv_shard_remote::endpoints_for_shards(&target_shards, &cluster.endpoints)
            .into_iter()
            .map(|(shard_id, endpoint)| dv_shard_remote::ShardFanoutRequest {
                shard_id,
                endpoint,
                request: dv_shard_remote::ShardQueryRequest {
                    vector: query_vector.to_vec(),
                    top_k,
                    ef,
                },
            })
            .collect();

        let mut merged = Vec::new();

        if !remote_targets.is_empty() {
            let remote = dv_shard_remote::fan_out_shard_queries(&remote_targets, 30_000)
                .map_err(|e| TopolseaError::InvalidConfig(e.to_string()))?;
            for partial in remote {
                for hit in partial.hits {
                    merged.push(QueryResult {
                        id: hit.id.clone(),
                        internal_id: hit.vector_id(),
                        distance: hit.distance,
                        score: hit.score,
                        metadata: None,
                    });
                }
            }
        }

        let remote_shard_ids: std::collections::HashSet<_> =
            remote_targets.iter().map(|t| t.shard_id).collect();

        for shard_id in target_shards {
            if remote_shard_ids.contains(&shard_id) {
                continue;
            }
            let physical = manifest.physical_name(shard_id);
            let col = self.get_collection(&physical)?;
            let mut partial = col.query(query_vector, top_k, filter, ef)?;
            merged.append(&mut partial);
        }

        Ok(merge_shard_results(merged, top_k))
    }

    fn resolve_target_shards(
        manifest: &ShardManifest,
        routing: &dv_storage::ShardRoutingIndex,
        query_vector: &[f32],
    ) -> Vec<usize> {
        if manifest.index_kind == IndexKind::ZColumn {
            let route = dv_index_zcolumn::ShardQueryRoute {
                dimension: manifest.dimension,
                projection_seed: manifest.zcolumn.projection_seed,
                outer_grid: manifest.zcolumn.outer_grid,
                max_layers: manifest.zcolumn.max_layers,
                pitch_ratio: manifest.zcolumn.pitch_ratio,
                num_shards: manifest.num_shards,
                beam_radius: routing.beam_radius,
            };
            let mut ids =
                dv_index_zcolumn::shard_ids_for_query(query_vector, &route, &routing.placements);
            if ids.is_empty() {
                ids = (0..manifest.num_shards).collect();
            }
            ids
        } else {
            (0..manifest.num_shards).collect()
        }
    }

    /// Register a remote HTTP endpoint for a physical shard (cross-node fan-out).
    pub fn set_shard_endpoint(
        &mut self,
        logical_name: &str,
        shard_id: usize,
        base_url: impl Into<String>,
    ) -> Result<()> {
        let manifest = self.storage.read_shard_manifest(logical_name)?;
        if shard_id >= manifest.num_shards {
            return Err(TopolseaError::InvalidConfig(format!(
                "shard_id {shard_id} out of range (num_shards={})",
                manifest.num_shards
            )));
        }
        let mut cluster = self.storage.read_shard_cluster(logical_name)?;
        cluster.endpoints.insert(shard_id, base_url.into());
        self.storage.write_shard_cluster(logical_name, &cluster)
    }

    pub fn shard_cluster_config(&self, logical_name: &str) -> Result<ShardClusterConfig> {
        self.storage.read_shard_cluster(logical_name)
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
