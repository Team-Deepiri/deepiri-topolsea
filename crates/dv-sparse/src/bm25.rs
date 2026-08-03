use crate::tokenize::tokenize;
use dv_types::VectorId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bm25Params {
    pub k1: f32,
    pub b: f32,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Posting {
    /// term frequency in the document
    tf: u32,
}

/// In-memory BM25 inverted index keyed by `VectorId`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Bm25Index {
    params: Bm25Params,
    /// term → (doc_id → posting)
    postings: HashMap<String, HashMap<u64, Posting>>,
    /// doc_id → token count
    doc_len: HashMap<u64, u32>,
    /// doc_id → raw text (for rebuild / persist)
    texts: HashMap<u64, String>,
    total_len: u64,
}

impl Bm25Index {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_params(params: Bm25Params) -> Self {
        Self {
            params,
            ..Self::default()
        }
    }

    pub fn len(&self) -> usize {
        self.doc_len.len()
    }

    pub fn is_empty(&self) -> bool {
        self.doc_len.is_empty()
    }

    pub fn upsert(&mut self, id: VectorId, text: &str) {
        self.remove(id);
        let tokens = tokenize(text);
        if tokens.is_empty() {
            self.texts.insert(id.raw(), text.to_string());
            self.doc_len.insert(id.raw(), 0);
            return;
        }
        let mut tf: HashMap<String, u32> = HashMap::new();
        for t in &tokens {
            *tf.entry(t.clone()).or_default() += 1;
        }
        let len = tokens.len() as u32;
        self.doc_len.insert(id.raw(), len);
        self.total_len += len as u64;
        self.texts.insert(id.raw(), text.to_string());
        for (term, count) in tf {
            self.postings
                .entry(term)
                .or_default()
                .insert(id.raw(), Posting { tf: count });
        }
    }

    pub fn remove(&mut self, id: VectorId) {
        if let Some(len) = self.doc_len.remove(&id.raw()) {
            self.total_len = self.total_len.saturating_sub(len as u64);
        }
        self.texts.remove(&id.raw());
        for map in self.postings.values_mut() {
            map.remove(&id.raw());
        }
        self.postings.retain(|_, m| !m.is_empty());
    }

    pub fn text(&self, id: VectorId) -> Option<&str> {
        self.texts.get(&id.raw()).map(|s| s.as_str())
    }

    fn avg_dl(&self) -> f32 {
        let n = self.doc_len.len().max(1) as f32;
        self.total_len as f32 / n
    }

    fn idf(&self, df: usize) -> f32 {
        let n = self.doc_len.len() as f32;
        let df = df as f32;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    /// Rank documents for a free-text query; higher score is better.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<(VectorId, f32)> {
        if top_k == 0 || self.is_empty() {
            return Vec::new();
        }
        let q_tokens = tokenize(query);
        if q_tokens.is_empty() {
            return Vec::new();
        }
        let avgdl = self.avg_dl().max(1e-6);
        let mut scores: HashMap<u64, f32> = HashMap::new();
        let mut seen_terms = std::collections::HashSet::new();
        for term in &q_tokens {
            if !seen_terms.insert(term.clone()) {
                continue; // query term once
            }
            let Some(postings) = self.postings.get(term) else {
                continue;
            };
            let idf = self.idf(postings.len());
            for (&doc, posting) in postings {
                let dl = *self.doc_len.get(&doc).unwrap_or(&1) as f32;
                let tf = posting.tf as f32;
                let denom =
                    tf + self.params.k1 * (1.0 - self.params.b + self.params.b * dl / avgdl);
                let contrib = idf * (tf * (self.params.k1 + 1.0) / denom);
                *scores.entry(doc).or_default() += contrib;
            }
        }
        let mut ranked: Vec<_> = scores
            .into_iter()
            .map(|(id, s)| (VectorId(id), s))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(top_k);
        ranked
    }

    pub fn to_bytes(&self) -> dv_types::Result<Vec<u8>> {
        Ok(serde_json::to_vec(self)?)
    }

    pub fn from_bytes(bytes: &[u8]) -> dv_types::Result<Self> {
        Ok(serde_json::from_slice(bytes)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_relevant_doc_first() {
        let mut idx = Bm25Index::new();
        idx.upsert(VectorId(0), "the cat sat on the mat");
        idx.upsert(VectorId(1), "dogs and puppies play outside");
        idx.upsert(VectorId(2), "a cat and another cat");
        let hits = idx.search("cat", 2);
        assert_eq!(hits[0].0, VectorId(2));
    }
}
