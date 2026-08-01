use dv_index_api::VectorIndex;
use dv_index_flat::FlatIndex;
use dv_index_hnsw::HnswIndex;
use dv_index_ivf::IvfIndex;
use dv_index_zcolumn::{ColumnStack, ZColumnIndex};
use dv_metadata::{empty_metadata, Filter, InvertedIndex, MetadataStore};
use dv_sparse::Bm25Index;
use dv_storage::{ColumnCellRecord, StorageEngine, Wal, WalRecord, ZColumnManifest};
use dv_types::{
    CollectionConfig, ExternalId, IndexKind, QuantTier, Result, TopolseaError, Vector, VectorId,
};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Mutex;

use crate::hybrid::{reciprocal_rank_fusion, DEFAULT_RRF_K};
use crate::planner::{IndexPlanner, QueryPlannerInput};
use crate::query::{QueryExplainResult, QueryResult};

fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn tier_for_layer(layer: u8, max_layers: u8) -> QuantTier {
    match layer {
        0 => QuantTier::U8,
        l if l + 1 >= max_layers => QuantTier::F32,
        _ => QuantTier::U16,
    }
}

enum IndexBackend {
    Flat(Box<FlatIndex>),
    Hnsw(Box<HnswIndex>),
    ZColumn(Box<ZColumnIndex>),
    Ivf(Box<IvfIndex>),
}

impl IndexBackend {
    fn as_mut(&mut self) -> &mut dyn VectorIndex {
        match self {
            IndexBackend::Flat(i) => i.as_mut(),
            IndexBackend::Hnsw(i) => i.as_mut(),
            IndexBackend::ZColumn(i) => i.as_mut(),
            IndexBackend::Ivf(i) => i.as_mut(),
        }
    }

    fn as_ref(&self) -> &dyn VectorIndex {
        match self {
            IndexBackend::Flat(i) => i.as_ref(),
            IndexBackend::Hnsw(i) => i.as_ref(),
            IndexBackend::ZColumn(i) => i.as_ref(),
            IndexBackend::Ivf(i) => i.as_ref(),
        }
    }

    fn encode_bytes(&self) -> Result<Vec<u8>> {
        match self {
            IndexBackend::Flat(i) => i.to_bytes(),
            IndexBackend::Hnsw(i) => i.to_bytes(),
            IndexBackend::ZColumn(i) => i.to_bytes(),
            IndexBackend::Ivf(i) => i.to_bytes(),
        }
    }

    fn from_bytes(kind: IndexKind, bytes: &[u8]) -> Result<Self> {
        match kind {
            IndexKind::Flat => Ok(IndexBackend::Flat(Box::new(FlatIndex::from_bytes(bytes)?))),
            IndexKind::Hnsw => Ok(IndexBackend::Hnsw(Box::new(HnswIndex::from_bytes(bytes)?))),
            IndexKind::ZColumn => Ok(IndexBackend::ZColumn(Box::new(ZColumnIndex::from_bytes(
                bytes,
            )?))),
            IndexKind::Ivf => Ok(IndexBackend::Ivf(Box::new(IvfIndex::from_bytes(bytes)?))),
        }
    }

    fn ids(&self) -> Vec<VectorId> {
        match self {
            IndexBackend::Flat(f) => f.ids().collect(),
            IndexBackend::Hnsw(h) => h.ids().collect(),
            IndexBackend::ZColumn(z) => z.ids().collect(),
            IndexBackend::Ivf(i) => i.ids().collect(),
        }
    }

    fn rebalance_if_zcolumn(&mut self) {
        if let IndexBackend::ZColumn(z) = self {
            z.flush_access();
            z.rebalance();
        }
    }

    fn record_zcolumn_access(&self, hit_ids: &[VectorId]) {
        if let IndexBackend::ZColumn(z) = self {
            z.record_access(hit_ids, now_unix_ms());
        }
    }

    fn search_filtered(
        &self,
        query: &[f32],
        top_k: usize,
        ef: usize,
        eligible: Option<&dyn Fn(VectorId) -> bool>,
    ) -> Result<Vec<dv_types::SearchHit>> {
        match self {
            IndexBackend::Flat(i) => i.search_filtered(query, top_k, eligible),
            IndexBackend::Hnsw(i) => i.search_filtered(query, top_k, ef, eligible),
            IndexBackend::ZColumn(z) => {
                let (hits, _) = z.search_with_explain_filtered(query, top_k, ef, eligible)?;
                Ok(hits)
            }
            IndexBackend::Ivf(i) => i.search_filtered(query, top_k, None, eligible),
        }
    }

    fn load_segments(&mut self, storage: &StorageEngine, name: &str) -> Result<()> {
        let IndexBackend::ZColumn(z) = self else {
            return Ok(());
        };

        let manifest_path = storage.columns_dir(name).join("manifest.json");
        if !manifest_path.exists() {
            return Ok(());
        }

        let manifest = storage.read_zcolumn_manifest(name)?;
        if !z.columns().is_empty() {
            return Ok(());
        }

        let mut layer_stacks: Vec<(u8, Vec<ColumnStack>)> = Vec::new();
        for (layer_idx, _) in manifest.layer_files.iter().enumerate() {
            let layer = layer_idx as u8;
            let tier = tier_for_layer(layer, manifest.max_layers);
            let records = storage.read_column_layer(name, layer, manifest.dimension, tier)?;
            let stacks: Vec<ColumnStack> = records
                .into_iter()
                .map(|rec| {
                    ColumnStack::from_persisted(
                        &rec.path_key,
                        rec.ids,
                        rec.payloads,
                        rec.centroid,
                        tier,
                        manifest.dimension,
                    )
                })
                .collect();
            if !stacks.is_empty() {
                layer_stacks.push((layer, stacks));
            }
        }

        if !layer_stacks.is_empty() {
            z.restore_from_segments(manifest.dimension, &layer_stacks);
        }
        Ok(())
    }

    fn persist_segments(
        &self,
        storage: &StorageEngine,
        name: &str,
        config: &CollectionConfig,
    ) -> Result<()> {
        let IndexBackend::ZColumn(z) = self else {
            return Ok(());
        };

        let mut layer_files = Vec::new();
        for layer in 0..z.grid().num_layers() {
            let layer_u8 = layer as u8;
            let tier = tier_for_layer(layer_u8, z.grid().num_layers() as u8);

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

            storage.write_column_layer(name, layer_u8, tier, config.dimension, &records)?;
            layer_files.push(format!("L{layer_u8}.grid.bin"));
        }

        let manifest = ZColumnManifest {
            outer_grid: config.zcolumn.outer_grid,
            max_layers: config.zcolumn.max_layers,
            pitch_ratio: config.zcolumn.pitch_ratio,
            dimension: config.dimension,
            layer_files,
        };
        storage.write_zcolumn_manifest(name, &manifest)?;
        Ok(())
    }

    fn rebuild_zcolumn_from_vectors(
        &mut self,
        storage: &StorageEngine,
        name: &str,
        kind: IndexKind,
    ) -> Result<()> {
        if kind != IndexKind::ZColumn || !self.as_ref().is_empty() {
            return Ok(());
        }
        let vectors = storage.read_vectors(name)?;
        if vectors.is_empty() {
            return Ok(());
        }
        if let IndexBackend::ZColumn(z) = self {
            z.rebuild_from_vectors(&vectors)?;
        }
        Ok(())
    }

    fn zcolumn_search_explain(
        &self,
        query: &[f32],
        top_k: usize,
        ef: usize,
        eligible: Option<&dyn Fn(VectorId) -> bool>,
    ) -> Option<Result<(Vec<dv_types::SearchHit>, dv_index_zcolumn::QueryExplain)>> {
        match self {
            IndexBackend::ZColumn(z) => {
                Some(z.search_with_explain_filtered(query, top_k, ef, eligible))
            }
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
    sparse: Bm25Index,
    external_to_internal: HashMap<String, VectorId>,
    internal_to_external: HashMap<VectorId, ExternalId>,
    wal: Mutex<Wal>,
    /// Next internal id (advanced on upsert; persisted in manifest on snapshot).
    next_id: u64,
    snapshot_seq: u64,
}

impl Collection {
    pub fn open(storage: StorageEngine, config: CollectionConfig) -> Result<Self> {
        let name = config.name.clone();
        let wal = Wal::open(storage.wal_path(&name))?;
        let snapshot_seq = storage.snapshot_seq(&name).unwrap_or(0);

        let index = if storage.read_index_blob(&name)?.is_empty() {
            Self::new_index(&config)
        } else {
            let bytes = storage.read_index_blob(&name)?;
            IndexBackend::from_bytes(config.index_kind, &bytes)?
        };

        let meta_map = storage.read_metadata_map(&name)?;
        let metadata = MetadataStore::load_from_persisted(meta_map);

        let sparse = {
            let bytes = storage.read_sparse_blob(&name)?;
            if bytes.is_empty() {
                Bm25Index::new()
            } else {
                Bm25Index::from_bytes(&bytes).unwrap_or_else(|_| Bm25Index::new())
            }
        };

        // next_id from manifest — will bump during WAL replay.
        let mut next_id = {
            // allocate_id path stores next_id in manifest; read via peek.
            storage
                .peek_allocate_id(&name)
                .map(|id| id.raw())
                .unwrap_or(0)
        };

        let mut col = Self {
            config,
            storage,
            index,
            metadata,
            sparse,
            external_to_internal: HashMap::new(),
            internal_to_external: HashMap::new(),
            wal: Mutex::new(wal),
            next_id,
            snapshot_seq,
        };
        col.rebuild_id_maps();
        let name = col.config.name.clone();
        let kind = col.config.index_kind;
        col.index.load_segments(&col.storage, &name)?;
        col.index
            .rebuild_zcolumn_from_vectors(&col.storage, &name, kind)?;

        // Replay durable mutations after last snapshot.
        col.replay_wal()?;
        next_id = col.next_id;
        let _ = next_id;
        Ok(col)
    }

    fn replay_wal(&mut self) -> Result<()> {
        let records = {
            let wal = self
                .wal
                .lock()
                .map_err(|_| TopolseaError::Storage("WAL lock poisoned".into()))?;
            wal.read_after(self.snapshot_seq)?
        };
        for (_seq, record) in records {
            match record {
                WalRecord::Upsert {
                    external_id,
                    internal_id,
                    vector,
                    metadata,
                    text,
                } => {
                    let id = VectorId(internal_id);
                    if id.raw() >= self.next_id {
                        self.next_id = id.raw() + 1;
                    }
                    self.apply_upsert_memory(&external_id, id, vector, metadata, text.as_deref())?;
                }
                WalRecord::Delete {
                    external_id,
                    internal_id,
                } => {
                    let id = VectorId(internal_id);
                    self.apply_delete_memory(&external_id, id)?;
                }
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
            IndexKind::Ivf => IndexBackend::Ivf(Box::new(IvfIndex::new(
                config.dimension,
                config.metric,
                config.ivf.clone(),
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

    fn apply_upsert_memory(
        &mut self,
        external_id: &str,
        internal_id: VectorId,
        vector: Vec<f32>,
        metadata: Value,
        text: Option<&str>,
    ) -> Result<()> {
        let v = Vector::new(vector);
        v.validate_dimension(self.config.dimension)?;

        self.external_to_internal
            .insert(external_id.to_string(), internal_id);
        self.internal_to_external
            .insert(internal_id, ExternalId::new(external_id));

        if self.index.as_ref().contains(internal_id) {
            self.index.as_mut().remove(internal_id)?;
        }
        self.index.as_mut().insert(internal_id, v)?;
        self.metadata.upsert(external_id, internal_id, metadata);
        if let Some(doc) = text {
            self.sparse.upsert(internal_id, doc);
        }
        Ok(())
    }

    fn apply_delete_memory(&mut self, external_id: &str, id: VectorId) -> Result<()> {
        self.external_to_internal.remove(external_id);
        self.internal_to_external.remove(&id);
        if self.index.as_ref().contains(id) {
            let _ = self.index.as_mut().remove(id);
        }
        self.metadata.remove(external_id);
        self.sparse.remove(id);
        Ok(())
    }

    pub fn upsert(
        &mut self,
        external_id: &str,
        vector: Vec<f32>,
        metadata: Option<Value>,
    ) -> Result<VectorId> {
        self.upsert_with_text(external_id, vector, metadata, None)
    }

    /// Upsert dense vector plus optional document text for BM25 / hybrid search.
    pub fn upsert_with_text(
        &mut self,
        external_id: &str,
        vector: Vec<f32>,
        metadata: Option<Value>,
        text: Option<&str>,
    ) -> Result<VectorId> {
        let v = Vector::new(vector.clone());
        v.validate_dimension(self.config.dimension)?;

        let internal_id = if let Some(id) = self.external_to_internal.get(external_id) {
            *id
        } else {
            let id = VectorId(self.next_id);
            self.next_id += 1;
            id
        };

        let meta = metadata.unwrap_or_else(empty_metadata);
        let record = WalRecord::Upsert {
            external_id: external_id.to_string(),
            internal_id: internal_id.raw(),
            vector: vector.clone(),
            metadata: meta.clone(),
            text: text.map(|s| s.to_string()),
        };
        {
            let mut wal = self
                .wal
                .lock()
                .map_err(|_| TopolseaError::Storage("WAL lock poisoned".into()))?;
            wal.append(&record)?;
        }

        self.apply_upsert_memory(external_id, internal_id, vector, meta, text)?;
        Ok(internal_id)
    }

    pub fn delete(&mut self, external_id: &str) -> Result<()> {
        let id = self
            .external_to_internal
            .get(external_id)
            .copied()
            .ok_or_else(|| TopolseaError::NotFound(external_id.to_string()))?;

        let record = WalRecord::Delete {
            external_id: external_id.to_string(),
            internal_id: id.raw(),
        };
        {
            let mut wal = self
                .wal
                .lock()
                .map_err(|_| TopolseaError::Storage("WAL lock poisoned".into()))?;
            wal.append(&record)?;
        }

        self.apply_delete_memory(external_id, id)?;
        Ok(())
    }

    pub fn query(
        &self,
        query_vector: &[f32],
        top_k: usize,
        filter: Option<&Filter>,
        ef: usize,
    ) -> Result<Vec<QueryResult>> {
        let eligible_bm = filter.and_then(|f| self.metadata.eligible_ids(f));
        let use_payload_ann = eligible_bm.is_some();

        // Tiny eligible sets: exact scan beats filtered HNSW connectivity.
        if let Some(ref bm) = eligible_bm {
            if bm.len() as usize <= top_k.saturating_mul(8).max(64) {
                return self.query_exact_eligible(query_vector, top_k, filter, bm);
            }
        }

        let fetch_k = if filter.is_some() && !use_payload_ann {
            top_k.saturating_mul(10).max(top_k)
        } else {
            top_k
        };

        let hits = if let Some(ref bm) = eligible_bm {
            let pred = |id: VectorId| InvertedIndex::contains_id(bm, id);
            self.index
                .search_filtered(query_vector, fetch_k, ef, Some(&pred))?
        } else {
            self.index
                .search_filtered(query_vector, fetch_k, ef, None)?
        };
        self.finish_query(hits, top_k, filter)
    }

    fn query_exact_eligible(
        &self,
        query_vector: &[f32],
        top_k: usize,
        filter: Option<&Filter>,
        bm: &roaring::RoaringBitmap,
    ) -> Result<Vec<QueryResult>> {
        use dv_metrics::distance;
        use dv_topk::{Candidate, TopKHeap};

        let mut heap = TopKHeap::new(top_k.max(1));
        for uid in bm.iter() {
            let id = VectorId(uid as u64);
            let Ok(vec) = self.index.as_ref().get_vector(id) else {
                continue;
            };
            let dist = distance(self.config.metric, query_vector, &vec.data);
            heap.push(Candidate { id, distance: dist });
        }
        let hits: Vec<_> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|c| dv_types::SearchHit::new(c.id, c.distance))
            .collect();
        self.finish_query(hits, top_k, filter)
    }

    fn finish_query(
        &self,
        hits: Vec<dv_types::SearchHit>,
        top_k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<QueryResult>> {
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

    /// Batch ANN query — one round-trip per query vector, shared filter/ef.
    pub fn query_batch(
        &self,
        query_vectors: &[&[f32]],
        top_k: usize,
        filter: Option<&Filter>,
        ef: usize,
    ) -> Result<Vec<Vec<QueryResult>>> {
        query_vectors
            .iter()
            .map(|q| self.query(q, top_k, filter, ef))
            .collect()
    }

    /// Hybrid dense + BM25 search fused with Reciprocal Rank Fusion.
    pub fn query_hybrid(
        &self,
        query_vector: &[f32],
        text_query: &str,
        top_k: usize,
        filter: Option<&Filter>,
        ef: usize,
        rrf_k: Option<f32>,
    ) -> Result<Vec<QueryResult>> {
        let fetch = top_k.saturating_mul(5).max(top_k);
        let dense = self.query(query_vector, fetch, filter, ef)?;
        let dense_list: Vec<(VectorId, f32)> =
            dense.iter().map(|r| (r.internal_id, r.score)).collect();

        let mut sparse_hits = self.sparse.search(text_query, fetch);
        if let Some(f) = filter {
            sparse_hits.retain(|(id, _)| {
                let Some(ext) = self.internal_to_external.get(id) else {
                    return false;
                };
                let meta = self.metadata.get(ext.as_str()).unwrap_or(&Value::Null);
                f.matches(meta)
            });
        }

        let fused = reciprocal_rank_fusion(
            &[dense_list, sparse_hits],
            top_k,
            rrf_k.unwrap_or(DEFAULT_RRF_K),
        );

        let mut results = Vec::with_capacity(fused.len());
        for (id, rrf_score) in fused {
            let ext = self
                .internal_to_external
                .get(&id)
                .map(|e| e.as_str().to_string());
            let distance = self
                .index
                .as_ref()
                .get_vector(id)
                .ok()
                .map(|v| dv_metrics::distance(self.config.metric, query_vector, &v.data))
                .unwrap_or(0.0);
            results.push(QueryResult {
                id: ext.clone(),
                internal_id: id,
                distance,
                score: rrf_score,
                metadata: ext.and_then(|e| self.metadata.get(&e).cloned()),
            });
        }
        Ok(results)
    }

    /// Sparse-only BM25 search (document text previously upserted).
    pub fn query_sparse(&self, text_query: &str, top_k: usize) -> Result<Vec<QueryResult>> {
        let hits = self.sparse.search(text_query, top_k);
        let mut results = Vec::with_capacity(hits.len());
        for (id, score) in hits {
            let ext = self
                .internal_to_external
                .get(&id)
                .map(|e| e.as_str().to_string());
            results.push(QueryResult {
                id: ext.clone(),
                internal_id: id,
                distance: 0.0,
                score,
                metadata: ext.and_then(|e| self.metadata.get(&e).cloned()),
            });
        }
        Ok(results)
    }

    pub fn query_explain(
        &self,
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

        let eligible_bm = filter.and_then(|f| self.metadata.eligible_ids(f));
        let use_payload_ann = eligible_bm.is_some();

        if let Some(ref bm) = eligible_bm {
            if bm.len() as usize <= top_k.saturating_mul(8).max(64) {
                let results = self.query_exact_eligible(query_vector, top_k, filter, bm)?;
                explain.strategy = "payload_exact_eligible".into();
                return Ok((results, explain));
            }
        }

        let fetch_k = if filter.is_some() && !use_payload_ann {
            top_k.saturating_mul(10).max(top_k)
        } else {
            top_k
        };

        let hits = if let Some(ref bm) = eligible_bm {
            let pred = |id: VectorId| InvertedIndex::contains_id(bm, id);
            if let Some(result) =
                self.index
                    .zcolumn_search_explain(query_vector, fetch_k, ef, Some(&pred))
            {
                let (hits, zexplain) = result?;
                explain.entry_layer = Some(zexplain.entry_layer);
                explain.deepest_layer = Some(zexplain.deepest_layer_reached);
                explain.revert_count = zexplain.revert_count;
                explain.columns_scanned = zexplain.columns_scanned;
                explain.column_paths = zexplain.column_paths;
                explain.strategy = zexplain.strategy;
                hits
            } else {
                explain.strategy = "payload_aware_index_search".into();
                self.index
                    .search_filtered(query_vector, fetch_k, ef, Some(&pred))?
            }
        } else if let Some(result) =
            self.index
                .zcolumn_search_explain(query_vector, fetch_k, ef, None)
        {
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
            self.index
                .search_filtered(query_vector, fetch_k, ef, None)?
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

    /// Snapshot to disk and truncate the WAL. Durability of upserts comes from the WAL;
    /// `persist` is the compaction / recovery checkpoint.
    pub fn persist(&mut self) -> Result<()> {
        self.index.rebalance_if_zcolumn();

        let index_bytes = self.index.encode_bytes()?;
        self.storage.write_index_blob(self.name(), &index_bytes)?;

        if self.config.index_kind == IndexKind::ZColumn {
            self.index
                .persist_segments(&self.storage, self.name(), &self.config)?;
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
        // B7: incremental sealed segments (no full-corpus rewrite of prior segs).
        self.storage.flush_vector_segments(self.name(), &refs)?;

        let sparse_bytes = self.sparse.to_bytes()?;
        self.storage.write_sparse_blob(self.name(), &sparse_bytes)?;

        // Advance snapshot watermark and truncate WAL.
        let wal_seq = {
            let wal = self
                .wal
                .lock()
                .map_err(|_| TopolseaError::Storage("WAL lock poisoned".into()))?;
            wal.next_seq().saturating_sub(1)
        };
        self.storage.set_snapshot_seq(self.name(), wal_seq)?;
        self.storage.set_next_id(self.name(), self.next_id)?;
        {
            let mut wal = self
                .wal
                .lock()
                .map_err(|_| TopolseaError::Storage("WAL lock poisoned".into()))?;
            wal.truncate()?;
        }
        self.snapshot_seq = 0;
        Ok(())
    }
}
