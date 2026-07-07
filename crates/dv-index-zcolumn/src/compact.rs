use crate::column::ColumnStack;
use crate::grid::{CellCoord, ColumnPath, FractalGrid};
use crate::quant::QuantTier;
use dv_types::VectorId;
use std::collections::HashMap;

const HOT_THRESHOLD: f32 = 0.5;
const COLD_THRESHOLD: f32 = 0.05;

/// Self-compacting engine: center collapse, hot promote, cold demote.
#[derive(Debug)]
pub struct CompactionEngine {
    pub events: u64,
}

impl CompactionEngine {
    pub fn new() -> Self {
        Self { events: 0 }
    }

    fn column_key(key: (u8, u16, u16)) -> String {
        format!("{}:{}:{}", key.0, key.1, key.2)
    }

    fn parse_key(s: &str) -> Option<(u8, u16, u16)> {
        let parts: Vec<_> = s.split(':').collect();
        if parts.len() != 3 {
            return None;
        }
        Some((
            parts[0].parse().ok()?,
            parts[1].parse().ok()?,
            parts[2].parse().ok()?,
        ))
    }

    /// Run compaction: collapse empty inner cells, promote hot, demote cold.
    pub fn collapse_and_promote(
        &mut self,
        grid: &mut FractalGrid,
        columns: &mut HashMap<String, ColumnStack>,
        vectors: &HashMap<VectorId, Vec<f32>>,
        dimension: usize,
        max_layers: u8,
    ) {
        self.collapse_empty_inner(grid, columns);
        self.promote_hot(columns, vectors, dimension, max_layers);
        self.demote_cold(columns, vectors, dimension, max_layers);
    }

    fn collapse_empty_inner(
        &mut self,
        grid: &FractalGrid,
        columns: &mut HashMap<String, ColumnStack>,
    ) {
        let max_layer = grid.num_layers().saturating_sub(1) as u8;
        if max_layer == 0 {
            return;
        }

        let inner_cells: Vec<_> = columns
            .iter()
            .filter(|(key, col)| {
                Self::parse_key(key)
                    .map(|(layer, _, _)| layer == max_layer && col.is_empty())
                    .unwrap_or(false)
            })
            .map(|(k, _)| k.clone())
            .collect();

        for key in inner_cells {
            columns.remove(&key);
            self.events += 1;
        }

        let remaining_inner: usize = columns
            .keys()
            .filter(|key| {
                Self::parse_key(key)
                    .map(|(layer, _, _)| layer == max_layer)
                    .unwrap_or(false)
            })
            .count();

        if remaining_inner == 0 && grid.num_layers() > 1 {
            self.events += 1;
        }
    }

    fn promote_hot(
        &mut self,
        columns: &mut HashMap<String, ColumnStack>,
        vectors: &HashMap<VectorId, Vec<f32>>,
        dimension: usize,
        max_layers: u8,
    ) {
        let hot_ids: Vec<(VectorId, CellCoord)> = columns
            .iter()
            .filter_map(|(key, col)| {
                let (layer, x, y) = Self::parse_key(key)?;
                if col.ledger.is_hot(HOT_THRESHOLD) && layer > 0 {
                    let id = *col.ids.last()?;
                    Some((id, CellCoord::new(layer - 1, x, y)))
                } else {
                    None
                }
            })
            .collect();

        for (id, target_cell) in hot_ids {
            if let Some(vec) = vectors.get(&id) {
                let key = Self::column_key(target_cell.key());
                let tier = QuantTier::for_layer(target_cell.layer, max_layers);
                let col = columns.entry(key).or_insert_with(|| {
                    ColumnStack::new(ColumnPath::from_cell(target_cell), dimension, tier)
                });
                if !col.ids.contains(&id) {
                    col.push(id, vec);
                    self.events += 1;
                }
            }
        }
    }

    fn demote_cold(
        &mut self,
        columns: &mut HashMap<String, ColumnStack>,
        vectors: &HashMap<VectorId, Vec<f32>>,
        dimension: usize,
        max_layers: u8,
    ) {
        let cold_moves: Vec<(VectorId, String, CellCoord)> = columns
            .iter()
            .filter_map(|(key, col)| {
                let (layer, x, y) = Self::parse_key(key)?;
                if col.ledger.is_cold(COLD_THRESHOLD) && layer + 1 < max_layers {
                    let id = *col.ids.first()?;
                    let target = CellCoord::new(layer + 1, x, y);
                    Some((id, key.clone(), target))
                } else {
                    None
                }
            })
            .collect();

        for (id, src_key, target_cell) in cold_moves {
            if let Some(vec) = vectors.get(&id) {
                let dst_key = Self::column_key(target_cell.key());
                let tier = QuantTier::for_layer(target_cell.layer, max_layers);
                if let Some(src) = columns.get_mut(&src_key) {
                    src.remove_id(id);
                }
                let col = columns.entry(dst_key).or_insert_with(|| {
                    ColumnStack::new(ColumnPath::from_cell(target_cell), dimension, tier)
                });
                col.push(id, vec);
                self.events += 1;
            }
        }
    }
}

impl Default for CompactionEngine {
    fn default() -> Self {
        Self::new()
    }
}
