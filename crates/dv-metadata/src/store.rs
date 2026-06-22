use dv_types::VectorId;
use serde_json::{Map, Value};
use std::collections::HashMap;

#[derive(Debug, Default, Clone)]
pub struct MetadataStore {
    /// external_id -> metadata document
    by_external: HashMap<String, Value>,
    /// internal id string -> external id
    id_map: HashMap<String, String>,
}

impl MetadataStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn upsert(&mut self, external_id: &str, internal_id: VectorId, metadata: Value) {
        self.id_map
            .insert(internal_id.to_string(), external_id.to_string());
        self.by_external.insert(external_id.to_string(), metadata);
    }

    pub fn remove(&mut self, external_id: &str) {
        self.by_external.remove(external_id);
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
        let id_map = map
            .remove("__id_map__")
            .and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or_default();
        Self {
            by_external: map,
            id_map,
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
}

pub fn empty_metadata() -> Value {
    Value::Object(Map::new())
}
