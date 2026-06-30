use dv_index_api::VectorIndex;
use dv_index_flat::FlatIndex;
use dv_index_hnsw::HnswIndex;
use dv_metadata::{empty_metadata, Filter, MetadataStore};
use dv_storage::StorageEngine;
use dv_types::{CollectionConfig, ExternalId, IndexKind, Result, TopolseaError, Vector, VectorId};
use serde_json::Value;
use std::collections::HashMap;

use crate::query::QueryResult;

enum IndexBackend {
    Flat(Box<FlatIndex>),
    Hnsw(Box<HnswIndex>),
}

impl IndexBackend {
    fn as_mut(&mut self) -> &mut dyn VectorIndex {
        match self {
            IndexBackend::Flat(i) => i.as_mut(),
            IndexBackend::Hnsw(i) => i.as_mut(),
        }
    }

    fn as_ref(&self) -> &dyn VectorIndex {
        match self {
            IndexBackend::Flat(i) => i.as_ref(),
            IndexBackend::Hnsw(i) => i.as_ref(),
        }
    }

    fn encode_bytes(&self) -> Result<Vec<u8>> {
        match self {
            IndexBackend::Flat(i) => i.to_bytes(),
            IndexBackend::Hnsw(i) => i.to_bytes(),
        }
    }

    fn from_bytes(kind: IndexKind, bytes: &[u8]) -> Result<Self> {
        match kind {
            IndexKind::Flat => Ok(IndexBackend::Flat(Box::new(FlatIndex::from_bytes(bytes)?))),
            IndexKind::Hnsw => Ok(IndexBackend::Hnsw(Box::new(HnswIndex::from_bytes(bytes)?))),
        }
    }

    fn ids(&self) -> Vec<VectorId> {
        match self {
            IndexBackend::Flat(f) => f.ids().collect(),
            IndexBackend::Hnsw(h) => h.ids().collect(),
        }
    }
}

/// A single named vector collection with index + metadata.
pub struct Collection {
    config: CollectionConfig,
    storage: StorageEngine,
    index: IndexBackend,
    metadata: MetadataStore,
    external_to_internal: HashMap<String, VectorId>,
    internal_to_external: HashMap<VectorId, ExternalId>,
}

impl Collection {
    pub fn open(storage: StorageEngine, config: CollectionConfig) -> Result<Self> {
        let index = if storage.read_index_blob(&config.name)?.is_empty() {
            Self::new_index(&config)
        } else {
            let bytes = storage.read_index_blob(&config.name)?;
            IndexBackend::from_bytes(config.index_kind, &bytes)?
        };

        let meta_map = storage.read_metadata_map(&config.name)?;
        let metadata = MetadataStore::load_from_persisted(meta_map);

        let mut col = Self {
            config,
            storage,
            index,
            metadata,
            external_to_internal: HashMap::new(),
            internal_to_external: HashMap::new(),
        };
        col.rebuild_id_maps();
        Ok(col)
    }

    fn new_index(config: &CollectionConfig) -> IndexBackend {
        match config.index_kind {
            IndexKind::Flat => {
                IndexBackend::Flat(Box::new(FlatIndex::new(config.dimension, config.metric)))
            }
            IndexKind::Hnsw => IndexBackend::Hnsw(Box::new(HnswIndex::new(
                config.dimension,
                config.metric,
                config.hnsw.clone(),
            ))),
        }
    }

    fn rebuild_id_maps(&mut self) {
        self.external_to_internal.clear();
        self.internal_to_external.clear();
        for (vid_str, ext) in self.metadata.id_mappings() {
            if let Ok(vid) = vid_str.parse::<u64>() {
                let id = VectorId(vid);
                self.external_to_internal.insert(ext.to_string(), id);
                self.internal_to_external.insert(id, ExternalId::new(ext));
            }
        }
    }

    pub fn name(&self) -> &str {
        &self.config.name
    }

    pub fn config(&self) -> &CollectionConfig {
        &self.config
    }

    pub fn len(&self) -> usize {
        self.index.as_ref().len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.as_ref().is_empty()
    }

    pub fn upsert(
        &mut self,
        external_id: &str,
        vector: Vec<f32>,
        metadata: Option<Value>,
    ) -> Result<VectorId> {
        let v = Vector::new(vector);
        v.validate_dimension(self.config.dimension)?;

        let internal_id = if let Some(id) = self.external_to_internal.get(external_id) {
            *id
        } else {
            let id = self.storage.allocate_id(self.name())?;
            self.external_to_internal
                .insert(external_id.to_string(), id);
            self.internal_to_external
                .insert(id, ExternalId::new(external_id));
            id
        };

        if self.index.as_ref().contains(internal_id) {
            self.index.as_mut().remove(internal_id)?;
        }
        self.index.as_mut().insert(internal_id, v)?;

        let meta = metadata.unwrap_or_else(empty_metadata);
        self.metadata.upsert(external_id, internal_id, meta);
        Ok(internal_id)
    }

    pub fn delete(&mut self, external_id: &str) -> Result<()> {
        let id = self
            .external_to_internal
            .remove(external_id)
            .ok_or_else(|| TopolseaError::NotFound(external_id.to_string()))?;
        self.internal_to_external.remove(&id);
        self.index.as_mut().remove(id)?;
        self.metadata.remove(external_id);
        Ok(())
    }

    pub fn query(
        &self,
        query_vector: &[f32],
        top_k: usize,
        filter: Option<&Filter>,
        ef: usize,
    ) -> Result<Vec<QueryResult>> {
        let fetch_k = if filter.is_some() {
            top_k.saturating_mul(10).max(top_k)
        } else {
            top_k
        };

        let hits = self.index.as_ref().search(query_vector, fetch_k, ef)?;
        let mut results = Vec::new();

        for hit in hits {
            let ext = self
                .internal_to_external
                .get(&hit.id)
                .map(|e| e.as_str().to_string());

            if let Some(ref external_id) = ext {
                if let Some(f) = filter {
                    let meta = self.metadata.get(external_id).unwrap_or(&Value::Null);
                    if !f.matches(meta) {
                        continue;
                    }
                }
            }

            results.push(QueryResult {
                id: ext.clone(),
                internal_id: hit.id,
                distance: hit.distance,
                score: hit.score,
                metadata: ext.and_then(|e| self.metadata.get(&e).cloned()),
            });

            if results.len() >= top_k {
                break;
            }
        }

        Ok(results)
    }

    pub fn persist(&mut self) -> Result<()> {
        let index_bytes = self.index.encode_bytes()?;
        self.storage.write_index_blob(self.name(), &index_bytes)?;

        let mut meta_map = self.metadata.to_persisted();
        let id_map: HashMap<String, String> = self
            .internal_to_external
            .iter()
            .map(|(k, v)| (k.to_string(), v.as_str().to_string()))
            .collect();
        meta_map.insert("__id_map__".to_string(), serde_json::to_value(&id_map)?);
        self.storage.write_metadata_map(self.name(), &meta_map)?;

        let records: Vec<(VectorId, Vec<f32>)> = self
            .index
            .ids()
            .into_iter()
            .filter_map(|id| {
                self.index
                    .as_ref()
                    .get_vector(id)
                    .ok()
                    .map(|v| (id, v.data))
            })
            .collect();
        let refs: Vec<_> = records.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        self.storage.write_vectors(self.name(), &refs)?;
        Ok(())
    }
}
