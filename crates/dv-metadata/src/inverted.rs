//! Payload inverted index: field → value → roaring bitmap of VectorId (as u32).

use dv_types::VectorId;
use roaring::RoaringBitmap;
use serde_json::Value;
use std::collections::HashMap;

/// In-memory inverted index over JSON metadata fields.
#[derive(Debug, Default, Clone)]
pub struct InvertedIndex {
    /// field → (canonical value key → ids)
    fields: HashMap<String, HashMap<String, RoaringBitmap>>,
}

impl InvertedIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.fields.clear();
    }

    pub fn upsert(&mut self, id: VectorId, metadata: &Value) {
        self.remove(id);
        let Some(obj) = metadata.as_object() else {
            return;
        };
        let uid = id_as_u32(id);
        for (field, value) in obj {
            let key = value_key(value);
            self.fields
                .entry(field.clone())
                .or_default()
                .entry(key)
                .or_default()
                .insert(uid);
        }
    }

    pub fn remove(&mut self, id: VectorId) {
        let uid = id_as_u32(id);
        for values in self.fields.values_mut() {
            for bitmap in values.values_mut() {
                bitmap.remove(uid);
            }
        }
    }

    /// Exact equality look-up.
    pub fn ids_eq(&self, field: &str, value: &Value) -> Option<RoaringBitmap> {
        let key = value_key(value);
        self.fields.get(field).and_then(|m| m.get(&key)).cloned()
    }

    /// Union of ids whose field value is in the given set.
    pub fn ids_in(&self, field: &str, values: &[Value]) -> Option<RoaringBitmap> {
        let map = self.fields.get(field)?;
        let mut out = RoaringBitmap::new();
        for v in values {
            if let Some(bm) = map.get(&value_key(v)) {
                out |= bm;
            }
        }
        Some(out)
    }

    /// All ids indexed under a field (for `$ne` = universe_field − value).
    pub fn ids_field(&self, field: &str) -> Option<RoaringBitmap> {
        let map = self.fields.get(field)?;
        let mut out = RoaringBitmap::new();
        for bm in map.values() {
            out |= bm;
        }
        Some(out)
    }

    pub fn contains_id(bitmap: &RoaringBitmap, id: VectorId) -> bool {
        bitmap.contains(id_as_u32(id))
    }
}

fn id_as_u32(id: VectorId) -> u32 {
    id.raw() as u32
}

fn value_key(value: &Value) -> String {
    match value {
        Value::Null => "null".into(),
        Value::Bool(b) => format!("b:{b}"),
        Value::Number(n) => format!("n:{n}"),
        Value::String(s) => format!("s:{s}"),
        other => format!("j:{}", other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn eq_and_remove() {
        let mut inv = InvertedIndex::new();
        inv.upsert(VectorId(1), &json!({"tag": "a", "n": 1}));
        inv.upsert(VectorId(2), &json!({"tag": "b", "n": 1}));
        let a = inv.ids_eq("tag", &json!("a")).unwrap();
        assert!(InvertedIndex::contains_id(&a, VectorId(1)));
        assert!(!InvertedIndex::contains_id(&a, VectorId(2)));
        inv.remove(VectorId(1));
        let a2 = inv.ids_eq("tag", &json!("a")).unwrap();
        assert!(!InvertedIndex::contains_id(&a2, VectorId(1)));
    }
}
