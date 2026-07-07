use dv_index_api::VectorIndex;
use dv_index_flat::FlatIndex;
use dv_index_hnsw::HnswIndex;
use dv_index_zcolumn::{ColumnStack, ZColumnIndex};
use dv_metadata::{empty_metadata, Filter, MetadataStore};
use dv_storage::{ColumnCellRecord, QuantTierTag, StorageEngine, ZColumnManifest};
use dv_types::{CollectionConfig, ExternalId, IndexKind, Result, TopolseaError, Vector, VectorId};
use serde_json::Value;
use std::collections::HashMap;

use crate::planner::{IndexPlanner, QueryPlannerInput};
use crate::query::{QueryExplainResult, QueryResult};

enum IndexBackend {
    Flat(Box<FlatIndex>),
    Hnsw(Box<HnswIndex>),
    ZColumn(Box<ZColumnIndex>),
}

impl IndexBackend {
    fn as_mut(&mut self) -> &mut dyn VectorIndex {
        match self {
            IndexBackend::Flat(i) => i.as_mut(),
            IndexBackend::Hnsw(i) => i.as_mut(),
            IndexBackend::ZColumn(i) => i.as_mut(),
        }
    }

    fn as_ref(&self) -> &dyn VectorIndex {
        match self {
            IndexBackend::Flat(i) => i.as_ref(),
            IndexBackend::Hnsw(i) => i.as_ref(),
            IndexBackend::ZColumn(i) => i.as_ref(),
        }
    }

    fn encode_bytes(&self) -> Result<Vec<u8>> {
        match self {
            IndexBackend::Flat(i) => i.to_bytes(),
            IndexBackend::Hnsw(i) => i.to_bytes(),
            IndexBackend::ZColumn(i) => i.to_bytes(),
        }
    }

    fn from_bytes(kind: IndexKind, bytes: &[u8]) -> Result<Self> {
        match kind {
            IndexKind::Flat => Ok(IndexBackend::Flat(Box::new(FlatIndex::from_bytes(bytes)?))),
            IndexKind::Hnsw => Ok(IndexBackend::Hnsw(Box::new(HnswIndex::from_bytes(bytes)?))),
            IndexKind::ZColumn => Ok(IndexBackend::ZColumn(Box::new(ZColumnIndex::from_bytes(
                bytes,
            )?))),
        }
    }

    fn ids(&self) -> Vec<VectorId> {
        match self {
            IndexBackend::Flat(f) => f.ids().collect(),
            IndexBackend::Hnsw(h) => h.ids().collect(),
            IndexBackend::ZColumn(z) => z.ids().collect(),
        }
    }

    fn rebalance_if_zcolumn(&mut self) {
        if let IndexBackend::ZColumn(z) = self {
            z.rebalance();
        }
    }

    fn record_zcolumn_access(&mut self, hit_ids: &[VectorId]) {
        if let IndexBackend::ZColumn(z) = self {
            z.record_access(hit_ids, 0);
        }
    }

    fn zcolumn_search_explain(
        &self,
        query: &[f32],
        top_k: usize,
        ef: usize,
    ) -> Option<Result<(Vec<dv_types::SearchHit>, dv_index_zcolumn::QueryExplain)>> {
        match self {
            IndexBackend::ZColumn(z) => Some(z.search_with_explain(query, top_k, ef)),
            _ => None,
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
        col.load_zcolumn_segments()?;
        col.rebuild_zcolumn_from_vectors()?;
        Ok(col)
    }

    fn rebuild_zcolumn_from_vectors(&mut self) -> Result<()> {
        if self.config.index_kind != IndexKind::ZColumn || !self.index.as_ref().is_empty() {
            return Ok(());
        }
        let vectors = self.storage.read_vectors(self.name())?;
        if vectors.is_empty() {
            return Ok(());
        }
        if let IndexBackend::ZColumn(z) = &mut self.index {
            z.rebuild_from_vectors(&vectors)?;
        }
        Ok(())
    }

    fn load_zcolumn_segments(&mut self) -> Result<()> {
        if self.config.index_kind != IndexKind::ZColumn {
            return Ok(());
        }
        let manifest_path = self.storage.columns_dir(self.name()).join("manifest.json");
        if !manifest_path.exists() {
            return Ok(());
        }
        let manifest = self.storage.read_zcolumn_manifest(self.name())?;
        let columns_empty = matches!(
            &self.index,
            IndexBackend::ZColumn(z) if z.columns().is_empty()
        );

        let mut layer_stacks: Vec<(u8, Vec<ColumnStack>)> = Vec::new();
        for (layer_idx, _) in manifest.layer_files.iter().enumerate() {
            let layer = layer_idx as u8;
            let tier = match layer {
                0 => QuantTierTag::U8,
                l if l + 1 >= manifest.max_layers => QuantTierTag::F32,
                _ => QuantTierTag::U16,
            };
            let quant_tier = match tier {
                QuantTierTag::U8 => dv_index_zcolumn::QuantTier::U8,
                QuantTierTag::U16 => dv_index_zcolumn::QuantTier::U16,
                QuantTierTag::F32 => dv_index_zcolumn::QuantTier::F32,
            };
            let records =
                self.storage
                    .read_column_layer(self.name(), layer, manifest.dimension, tier)?;
            let stacks: Vec<ColumnStack> = records
                .into_iter()
                .map(|rec| {
                    ColumnStack::from_persisted(
                        &rec.path_key,
                        rec.ids,
                        rec.payloads,
                        rec.centroid,
                        quant_tier,
                        manifest.dimension,
                    )
                })
                .collect();
            if !stacks.is_empty() {
                layer_stacks.push((layer, stacks));
            }
        }

        if !layer_stacks.is_empty() && columns_empty {
            if let IndexBackend::ZColumn(z) = &mut self.index {
                z.restore_from_segments(manifest.dimension, &layer_stacks);
            }
        }
        Ok(())
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
            IndexKind::ZColumn => IndexBackend::ZColumn(Box::new(ZColumnIndex::new(
                config.dimension,
                config.metric,
                config.zcolumn.clone(),
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
        &mut self,
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

        let hit_ids: Vec<VectorId> = results.iter().map(|r| r.internal_id).collect();
        self.index.record_zcolumn_access(&hit_ids);

        Ok(results)
    }

    pub fn query_explain(
        &mut self,
        query_vector: &[f32],
        top_k: usize,
        filter: Option<&Filter>,
        ef: usize,
    ) -> Result<(Vec<QueryResult>, QueryExplainResult)> {
        let plan = IndexPlanner::plan(&QueryPlannerInput {
            collection_size: self.len(),
            dimension: self.config.dimension,
            top_k,
            has_filter: filter.is_some(),
        });

        let mut explain = QueryExplainResult {
            index_kind: format!("{:?}", self.config.index_kind),
            planner_reason: Some(plan.reason),
            ..Default::default()
        };

        let fetch_k = if filter.is_some() {
            top_k.saturating_mul(10).max(top_k)
        } else {
            top_k
        };

        let hits =
            if let Some(result) = self.index.zcolumn_search_explain(query_vector, fetch_k, ef) {
                let (hits, zexplain) = result?;
                explain.entry_layer = Some(zexplain.entry_layer);
                explain.deepest_layer = Some(zexplain.deepest_layer_reached);
                explain.revert_count = zexplain.revert_count;
                explain.columns_scanned = zexplain.columns_scanned;
                explain.column_paths = zexplain.column_paths;
                explain.strategy = zexplain.strategy;
                hits
            } else {
                explain.strategy = "standard_index_search".into();
                self.index.as_ref().search(query_vector, fetch_k, ef)?
            };

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

        let hit_ids: Vec<VectorId> = results.iter().map(|r| r.internal_id).collect();
        self.index.record_zcolumn_access(&hit_ids);

        Ok((results, explain))
    }

    pub fn zcolumn_stats(&self) -> Option<serde_json::Value> {
        let IndexBackend::ZColumn(z) = &self.index else {
            return None;
        };
        let stats = z.search_stats();
        Some(serde_json::json!({
            "revert_count": stats.revert_count,
            "columns_scanned": stats.columns_scanned,
            "compaction_events": z.compaction_events(),
            "column_count": z.columns().len(),
            "vector_count": z.len(),
            "fractal_layers": z.grid().num_layers(),
        }))
    }

    pub fn persist(&mut self) -> Result<()> {
        self.index.rebalance_if_zcolumn();

        let index_bytes = self.index.encode_bytes()?;
        self.storage.write_index_blob(self.name(), &index_bytes)?;

        if self.config.index_kind == IndexKind::ZColumn {
            self.persist_zcolumn_segments()?;
        }

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

    fn persist_zcolumn_segments(&self) -> Result<()> {
        let IndexBackend::ZColumn(z) = &self.index else {
            return Ok(());
        };

        let mut layer_files = Vec::new();
        for layer in 0..z.grid().num_layers() {
            let layer_u8 = layer as u8;
            let tier = match layer {
                0 => QuantTierTag::U8,
                l if l + 1 >= z.grid().num_layers() => QuantTierTag::F32,
                _ => QuantTierTag::U16,
            };

            let records: Vec<ColumnCellRecord> = z
                .columns()
                .iter()
                .filter_map(|(key, col)| {
                    let cell = col.cell()?;
                    if cell.layer != layer_u8 {
                        return None;
                    }
                    Some(ColumnCellRecord {
                        path_key: key.clone(),
                        ids: col.ids.clone(),
                        payloads: col.quantized.clone(),
                        centroid: col.centroid.clone(),
                    })
                })
                .collect();

            self.storage.write_column_layer(
                self.name(),
                layer_u8,
                tier,
                self.config.dimension,
                &records,
            )?;
            layer_files.push(format!("L{layer_u8}.grid.bin"));
        }

        let manifest = ZColumnManifest {
            outer_grid: self.config.zcolumn.outer_grid,
            max_layers: self.config.zcolumn.max_layers,
            pitch_ratio: self.config.zcolumn.pitch_ratio,
            dimension: self.config.dimension,
            layer_files,
        };
        self.storage
            .write_zcolumn_manifest(self.name(), &manifest)?;
        Ok(())
    }
}
