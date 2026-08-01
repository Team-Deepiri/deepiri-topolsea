use crate::filter::{Filter, FilterOp};
use crate::inverted::InvertedIndex;
use dv_types::VectorId;
use roaring::RoaringBitmap;
use serde_json::{Map, Value};
use std::collections::HashMap;

#[derive(Debug, Default, Clone)]
pub struct MetadataStore {
    /// external_id -> metadata document
    by_external: HashMap<String, Value>,
    /// internal id string -> external id
    id_map: HashMap<String, String>,
    /// Payload inverted index (not persisted; rebuilt on load).
    inverted: InvertedIndex,
}

impl MetadataStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn upsert(&mut self, external_id: &str, internal_id: VectorId, metadata: Value) {
        self.id_map
            .insert(internal_id.to_string(), external_id.to_string());
        self.inverted.upsert(internal_id, &metadata);
        self.by_external.insert(external_id.to_string(), metadata);
    }

    pub fn remove(&mut self, external_id: &str) {
        if let Some(old) = self.by_external.remove(external_id) {
            // Find internal id to remove from inverted index.
            let internal = self
                .id_map
                .iter()
                .find(|(_, v)| v.as_str() == external_id)
                .and_then(|(k, _)| k.parse::<u64>().ok())
                .map(VectorId);
            if let Some(id) = internal {
                self.inverted.remove(id);
            }
            let _ = old;
        }
        self.id_map.retain(|_, v| v != external_id);
    }

    pub fn get(&self, external_id: &str) -> Option<&Value> {
        self.by_external.get(external_id)
    }

    pub fn external_id_for(&self, internal_id: VectorId) -> Option<&str> {
        self.id_map
            .get(&internal_id.to_string())
            .map(|s| s.as_str())
    }

    pub fn external_id_for_reverse(&self, external_id: &str) -> Option<&str> {
        self.id_map
            .iter()
            .find(|(_, v)| v.as_str() == external_id)
            .map(|(k, _)| k.as_str())
    }

    pub fn to_persisted(&self) -> HashMap<String, Value> {
        self.by_external.clone()
    }

    pub fn load_from_persisted(mut map: HashMap<String, Value>) -> Self {
        let id_map: HashMap<String, String> = map
            .remove("__id_map__")
            .and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or_default();
        let mut store = Self {
            by_external: map,
            id_map,
            inverted: InvertedIndex::new(),
        };
        store.rebuild_inverted();
        store
    }

    fn rebuild_inverted(&mut self) {
        self.inverted.clear();
        for (vid_str, ext) in &self.id_map {
            if let Ok(vid) = vid_str.parse::<u64>() {
                if let Some(meta) = self.by_external.get(ext) {
                    self.inverted.upsert(VectorId(vid), meta);
                }
            }
        }
    }

    pub fn external_ids(&self) -> impl Iterator<Item = &str> {
        self.by_external.keys().map(|s| s.as_str())
    }

    pub fn id_mappings(&self) -> impl Iterator<Item = (&str, &str)> {
        self.id_map.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    pub fn len(&self) -> usize {
        self.by_external.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_external.is_empty()
    }

    pub fn inverted(&self) -> &InvertedIndex {
        &self.inverted
    }

    /// Resolve filter to an eligible id set when the filter is index-backed.
    /// Returns `None` when the filter cannot be fully evaluated via the inverted index
    /// (caller should fall back to post-filtering).
    pub fn eligible_ids(&self, filter: &Filter) -> Option<RoaringBitmap> {
        match filter {
            Filter::Eq { field, value } => self.inverted.ids_eq(field, value),
            Filter::Cmp {
                field,
                op: FilterOp::Eq,
                value,
            } => self.inverted.ids_eq(field, value),
            Filter::Cmp {
                field,
                op: FilterOp::In,
                value,
            } => {
                let arr = value.as_array()?;
                self.inverted.ids_in(field, arr)
            }
            Filter::Cmp {
                field,
                op: FilterOp::Ne,
                value,
            } => {
                let all = self.inverted.ids_field(field)?;
                let eq = self
                    .inverted
                    .ids_eq(field, value)
                    .unwrap_or_else(RoaringBitmap::new);
                Some(all - eq)
            }
            Filter::And(items) => {
                let mut iter = items.iter();
                let mut acc = self.eligible_ids(iter.next()?)?;
                for item in iter {
                    acc &= self.eligible_ids(item)?;
                }
                Some(acc)
            }
            Filter::Or(items) => {
                let mut acc = RoaringBitmap::new();
                for item in items {
                    acc |= self.eligible_ids(item)?;
                }
                Some(acc)
            }
            // Range ops need a sorted numeric index; fall back to post-filter for now.
            Filter::Cmp {
                op: FilterOp::Gt | FilterOp::Gte | FilterOp::Lt | FilterOp::Lte,
                ..
            } => None,
        }
    }
}

pub fn empty_metadata() -> Value {
    Value::Object(Map::new())
}
