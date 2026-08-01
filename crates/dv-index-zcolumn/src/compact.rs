use crate::column::ColumnStack;
use crate::grid::{CellCoord, ColumnPath, FractalGrid};
use dv_types::{QuantTier, VectorId};
use std::collections::HashMap;
use std::str::FromStr;

const HOT_THRESHOLD: f32 = 0.5;
const COLD_THRESHOLD: f32 = 0.05;

/// Self-compacting engine: center collapse, hot promote (move), cold demote, height split.
#[derive(Debug)]
pub struct CompactionEngine {
    pub events: u64,
}

impl CompactionEngine {
    pub fn new() -> Self {
        Self { events: 0 }
    }

    /// Run compaction: collapse empty inner cells, promote hot, demote cold, split tall columns.
    pub fn collapse_and_promote(
        &mut self,
        grid: &mut FractalGrid,
        columns: &mut HashMap<String, ColumnStack>,
        vectors: &HashMap<VectorId, Vec<f32>>,
        dimension: usize,
        max_layers: u8,
    ) {
        self.collapse_and_promote_with_ratio(grid, columns, vectors, dimension, max_layers, 4.0);
    }

    pub fn collapse_and_promote_with_ratio(
        &mut self,
        grid: &mut FractalGrid,
        columns: &mut HashMap<String, ColumnStack>,
        vectors: &HashMap<VectorId, Vec<f32>>,
        dimension: usize,
        max_layers: u8,
        max_height_ratio: f32,
    ) {
        self.collapse_empty_inner(grid, columns);
        self.promote_hot(columns, vectors, dimension, max_layers);
        self.demote_cold(columns, vectors, dimension, max_layers);
        self.split_tall_columns(
            grid,
            columns,
            vectors,
            dimension,
            max_layers,
            max_height_ratio,
        );
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
                CellCoord::from_str(key)
                    .map(|cell| cell.layer == max_layer && col.is_empty())
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
                CellCoord::from_str(key)
                    .map(|cell| cell.layer == max_layer)
                    .unwrap_or(false)
            })
            .count();

        if remaining_inner == 0 && grid.num_layers() > 1 {
            self.events += 1;
        }
    }

    /// Promote hot ids by **moving** them (remove from source) — M4 move-not-copy.
    fn promote_hot(
        &mut self,
        columns: &mut HashMap<String, ColumnStack>,
        vectors: &HashMap<VectorId, Vec<f32>>,
        dimension: usize,
        max_layers: u8,
    ) {
        let hot_moves: Vec<(VectorId, String, CellCoord)> = columns
            .iter()
            .filter_map(|(key, col)| {
                let cell = CellCoord::from_str(key).ok()?;
                if col.ledger.is_hot(HOT_THRESHOLD) && cell.layer > 0 {
                    let id = *col.ids.last()?;
                    let target = CellCoord::new(cell.layer - 1, cell.x, cell.y);
                    Some((id, key.clone(), target))
                } else {
                    None
                }
            })
            .collect();

        for (id, src_key, target_cell) in hot_moves {
            if let Some(vec) = vectors.get(&id) {
                if let Some(src) = columns.get_mut(&src_key) {
                    src.remove_id(id);
                    src.rebuild_centroid(vectors, dimension);
                }
                let dst_key = target_cell.to_string();
                let tier = QuantTier::for_layer(target_cell.layer, max_layers);
                let col = columns.entry(dst_key).or_insert_with(|| {
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
                let cell = CellCoord::from_str(key).ok()?;
                if col.ledger.is_cold(COLD_THRESHOLD) && cell.layer + 1 < max_layers {
                    let id = *col.ids.first()?;
                    let target = CellCoord::new(cell.layer + 1, cell.x, cell.y);
                    Some((id, key.clone(), target))
                } else {
                    None
                }
            })
            .collect();

        for (id, src_key, target_cell) in cold_moves {
            if let Some(vec) = vectors.get(&id) {
                let dst_key = target_cell.to_string();
                let tier = QuantTier::for_layer(target_cell.layer, max_layers);
                if let Some(src) = columns.get_mut(&src_key) {
                    src.remove_id(id);
                    src.rebuild_centroid(vectors, dimension);
                }
                let col = columns.entry(dst_key).or_insert_with(|| {
                    ColumnStack::new(ColumnPath::from_cell(target_cell), dimension, tier)
                });
                col.push(id, vec);
                self.events += 1;
            }
        }
    }

    /// Split columns whose height exceeds `ratio × mean` (M4 height-balance).
    fn split_tall_columns(
        &mut self,
        grid: &FractalGrid,
        columns: &mut HashMap<String, ColumnStack>,
        vectors: &HashMap<VectorId, Vec<f32>>,
        dimension: usize,
        max_layers: u8,
        ratio: f32,
    ) {
        let heights: Vec<u32> = columns
            .values()
            .map(|c| c.height())
            .filter(|h| *h > 0)
            .collect();
        if heights.is_empty() {
            return;
        }
        let mean = heights.iter().map(|&h| h as f64).sum::<f64>() / heights.len() as f64;
        let threshold = ((mean * ratio.max(1.0) as f64).ceil() as u32).max(8);

        let tall: Vec<(String, CellCoord)> = columns
            .iter()
            .filter_map(|(k, c)| {
                let cell = CellCoord::from_str(k).ok()?;
                if c.height() > threshold && cell.layer + 1 < max_layers {
                    Some((k.clone(), cell))
                } else {
                    None
                }
            })
            .collect();

        for (src_key, cell) in tall {
            let Some(src) = columns.get(&src_key) else {
                continue;
            };
            let ids: Vec<VectorId> = src.ids.clone();
            if ids.len() < 2 {
                continue;
            }
            let mid = ids.len() / 2;
            let move_ids: Vec<VectorId> = ids[mid..].to_vec();
            // Prefer a child cell when available; else nudge x.
            let target = grid
                .child_cell(&cell)
                .unwrap_or_else(|| CellCoord::new(cell.layer, cell.x.saturating_add(1), cell.y));
            let dst_key = target.to_string();
            let tier = QuantTier::for_layer(target.layer, max_layers);

            for id in move_ids {
                let Some(vec) = vectors.get(&id) else {
                    continue;
                };
                if let Some(src) = columns.get_mut(&src_key) {
                    src.remove_id(id);
                }
                let col = columns.entry(dst_key.clone()).or_insert_with(|| {
                    ColumnStack::new(ColumnPath::from_cell(target), dimension, tier)
                });
                if !col.ids.contains(&id) {
                    col.push(id, vec);
                    self.events += 1;
                }
            }
            if let Some(src) = columns.get_mut(&src_key) {
                src.rebuild_centroid(vectors, dimension);
            }
            if let Some(dst) = columns.get_mut(&dst_key) {
                dst.rebuild_centroid(vectors, dimension);
            }
        }
    }
}

impl Default for CompactionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::AccessLedger;
    use dv_types::VectorId;

    #[test]
    fn promote_hot_moves_not_copies() {
        let mut columns = HashMap::new();
        let cell = CellCoord::new(1, 0, 0);
        let mut stack = ColumnStack::new(ColumnPath::from_cell(cell), 2, QuantTier::U8);
        let id = VectorId(7);
        let v = vec![1.0f32, 0.0];
        stack.push(id, &v);
        stack.ledger = AccessLedger::default();
        // Force hot.
        for _ in 0..20 {
            stack.ledger.record_hit(1_000);
        }
        columns.insert(cell.to_string(), stack);

        let mut vectors = HashMap::new();
        vectors.insert(id, v.clone());

        let mut engine = CompactionEngine::new();
        let mut grid = FractalGrid::new((4, 4), 3, 0.5);
        engine.collapse_and_promote(&mut grid, &mut columns, &vectors, 2, 3);

        let src = columns.get(&cell.to_string());
        let promoted = columns.get(&CellCoord::new(0, 0, 0).to_string());
        assert!(src.map(|c| !c.ids.contains(&id)).unwrap_or(true));
        assert!(promoted.map(|c| c.ids.contains(&id)).unwrap_or(false));
    }
}
