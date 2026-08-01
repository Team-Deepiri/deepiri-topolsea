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
    /// external id -> internal id (O(1) remove / reverse lookup)
    ext_to_id: HashMap<String, VectorId>,
    /// Payload inverted index (not persisted; rebuilt on load).
    inverted: InvertedIndex,
}

impl MetadataStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn upsert(&mut self, external_id: &str, internal_id: VectorId, metadata: Value) {
        // Drop stale reverse entry if this external id previously pointed elsewhere.
        if let Some(old_id) = self.ext_to_id.insert(external_id.to_string(), internal_id) {
            if old_id != internal_id {
                self.id_map.remove(&old_id.to_string());
                self.inverted.remove(old_id);
            }
        }
        // Drop stale forward entry if this internal id previously mapped to another external.
        if let Some(old_ext) = self
            .id_map
            .insert(internal_id.to_string(), external_id.to_string())
        {
            if old_ext != external_id {
                self.ext_to_id.remove(&old_ext);
            }
        }
        self.inverted.upsert(internal_id, &metadata);
        self.by_external.insert(external_id.to_string(), metadata);
    }

    pub fn remove(&mut self, external_id: &str) {
        self.by_external.remove(external_id);
        if let Some(id) = self.ext_to_id.remove(external_id) {
            self.id_map.remove(&id.to_string());
            self.inverted.remove(id);
        }
    }

    pub fn get(&self, external_id: &str) -> Option<&Value> {
        self.by_external.get(external_id)
    }

    pub fn external_id_for(&self, internal_id: VectorId) -> Option<&str> {
        self.id_map
            .get(&internal_id.to_string())
            .map(|s| s.as_str())
    }

    /// Resolve external id → internal id string (O(1)).
    pub fn external_id_for_reverse(&self, external_id: &str) -> Option<&str> {
        let id = self.ext_to_id.get(external_id)?;
        self.id_map
            .get_key_value(&id.to_string())
            .map(|(k, _)| k.as_str())
    }

    pub fn internal_id_for(&self, external_id: &str) -> Option<VectorId> {
        self.ext_to_id.get(external_id).copied()
    }

    pub fn to_persisted(&self) -> HashMap<String, Value> {
        self.by_external.clone()
    }

    pub fn load_from_persisted(mut map: HashMap<String, Value>) -> Self {
        let id_map: HashMap<String, String> = map
            .remove("__id_map__")
            .and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or_default();
        let mut ext_to_id = HashMap::with_capacity(id_map.len());
        for (vid_str, ext) in &id_map {
            if let Ok(vid) = vid_str.parse::<u64>() {
                ext_to_id.insert(ext.clone(), VectorId(vid));
            }
        }
        let mut store = Self {
            by_external: map,
            id_map,
            ext_to_id,
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
                let eq = self.inverted.ids_eq(field, value).unwrap_or_default();
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn remove_is_o1_and_keeps_maps_consistent() {
        let mut store = MetadataStore::new();
        store.upsert("a", VectorId(1), json!({"tag": "x"}));
        store.upsert("b", VectorId(2), json!({"tag": "y"}));
        store.upsert("c", VectorId(3), json!({"tag": "z"}));

        assert_eq!(store.internal_id_for("b"), Some(VectorId(2)));
        store.remove("b");
        assert!(store.get("b").is_none());
        assert!(store.internal_id_for("b").is_none());
        assert_eq!(store.external_id_for(VectorId(2)), None);
        assert_eq!(store.len(), 2);

        // Remaining ids still reverse-lookup correctly.
        assert_eq!(store.internal_id_for("a"), Some(VectorId(1)));
        assert_eq!(store.external_id_for_reverse("c"), Some("3"));
    }

    #[test]
    fn and_or_eligible_ids() {
        let mut store = MetadataStore::new();
        store.upsert("a", VectorId(1), json!({"tag": "x", "n": 1}));
        store.upsert("b", VectorId(2), json!({"tag": "y", "n": 10}));
        store.upsert("c", VectorId(3), json!({"tag": "x", "n": 10}));

        let and = Filter::from_json(&json!({
            "$and": [{"tag": "x"}, {"n": {"$in": [10]}}]
        }))
        .unwrap();
        let ids = store.eligible_ids(&and).unwrap();
        assert!(ids.contains(3));
        assert_eq!(ids.len(), 1);

        let or = Filter::from_json(&json!({
            "$or": [{"tag": "y"}, {"tag": "missing"}]
        }))
        .unwrap();
        let ids = store.eligible_ids(&or).unwrap();
        assert!(ids.contains(2));
        assert_eq!(ids.len(), 1);

        let multi = Filter::from_json(&json!({"tag": "x", "n": {"$in": [1, 10]}})).unwrap();
        let ids = store.eligible_ids(&multi).unwrap();
        assert_eq!(ids.len(), 2);
    }
}
