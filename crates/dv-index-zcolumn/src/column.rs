use crate::grid::{CellCoord, ColumnPath};
use crate::ledger::AccessLedger;
use crate::quant::{self, QuantTier};
use dv_types::VectorId;
use serde::{Deserialize, Serialize};

/// A vertical stack of vectors at one fractal cell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnStack {
    pub path: ColumnPath,
    pub ids: Vec<VectorId>,
    pub centroid: Vec<f32>,
    pub quant_tier: QuantTier,
    pub quantized: Vec<Vec<u8>>,
    pub ledger: AccessLedger,
}

impl ColumnStack {
    pub fn new(path: ColumnPath, dimension: usize, quant_tier: QuantTier) -> Self {
        Self {
            path,
            ids: Vec::new(),
            centroid: vec![0.0; dimension],
            quant_tier,
            quantized: Vec::new(),
            ledger: AccessLedger::default(),
        }
    }

    pub fn from_persisted(
        path_key: &str,
        ids: Vec<VectorId>,
        quantized: Vec<Vec<u8>>,
        centroid: Vec<f32>,
        tier: QuantTier,
        dimension: usize,
    ) -> Self {
        let parts: Vec<_> = path_key.split(':').collect();
        let cell = CellCoord::new(
            parts.first().and_then(|s| s.parse().ok()).unwrap_or(0),
            parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0),
            parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0),
        );
        Self {
            path: ColumnPath::from_cell(cell),
            ids,
            centroid: if centroid.is_empty() {
                vec![0.0; dimension]
            } else {
                centroid
            },
            quant_tier: tier,
            quantized,
            ledger: AccessLedger::default(),
        }
    }

    pub fn height(&self) -> u32 {
        self.ids.len() as u32
    }

    pub fn push(&mut self, id: VectorId, vector: &[f32]) {
        let n = self.ids.len() as f32;
        if self.ids.is_empty() {
            self.centroid = vector.to_vec();
        } else {
            for (c, &v) in self.centroid.iter_mut().zip(vector.iter()) {
                *c = (*c * n + v) / (n + 1.0);
            }
        }
        self.quantized.push(quant::encode(vector, self.quant_tier));
        self.ids.push(id);
    }

    pub fn remove_id(&mut self, id: VectorId) -> bool {
        if let Some(pos) = self.ids.iter().position(|&x| x == id) {
            self.ids.remove(pos);
            self.quantized.remove(pos);
            true
        } else {
            false
        }
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn cell(&self) -> Option<&CellCoord> {
        self.path.leaf()
    }
}
