use crate::column::ColumnStack;
use crate::grid::{CellCoord, FractalGrid};
use crate::predictor::LayerPredictor;
use crate::quant;
use dv_metrics::distance;
use dv_topk::{Candidate, TopKHeap};
use dv_types::{DistanceMetric, VectorId};
use std::collections::{HashSet, VecDeque};

/// Search statistics for benchmarking revert behavior.
#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    pub revert_count: u64,
    pub columns_scanned: u64,
}

/// Beam search with callback-reverse backtracking on miss.
pub struct RevertBeamSearch<'a> {
    grid: &'a FractalGrid,
    columns: &'a [ColumnStack],
    vectors: &'a std::collections::HashMap<VectorId, Vec<f32>>,
    metric: DistanceMetric,
    dimension: usize,
    predictor: LayerPredictor,
    stats: &'a mut SearchStats,
}

impl<'a> RevertBeamSearch<'a> {
    pub fn new(
        grid: &'a FractalGrid,
        columns: &'a [ColumnStack],
        vectors: &'a std::collections::HashMap<VectorId, Vec<f32>>,
        metric: DistanceMetric,
        dimension: usize,
        stats: &'a mut SearchStats,
    ) -> Self {
        Self {
            grid,
            columns,
            vectors,
            metric,
            dimension,
            predictor: LayerPredictor::default_predictor(),
            stats,
        }
    }

    pub fn run(&mut self, query: &[f32], top_k: usize, ef: usize) -> Vec<(VectorId, f32)> {
        if self.columns.is_empty() || top_k == 0 {
            return Vec::new();
        }

        let col_refs: Vec<&ColumnStack> = self.columns.iter().collect();
        let start_layer = self
            .predictor
            .entry_layer(query, self.grid, &col_refs, self.metric);

        let beam_width = ef.max(top_k).max(1);
        let mut heap = TopKHeap::new(top_k);
        let mut visited: HashSet<VectorId> = HashSet::new();
        let mut revert_stack: VecDeque<CellCoord> = VecDeque::new();

        let mut frontier: Vec<(CellCoord, f32)> = self.score_layer_columns(start_layer, query);
        frontier.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        frontier.truncate(beam_width);

        let mut depth = start_layer;
        let max_depth = self.grid.num_layers().saturating_sub(1) as u8;

        loop {
            let mut found_any = false;

            for (cell, _) in &frontier {
                self.stats.columns_scanned += 1;
                if let Some(col) = self.column_at(*cell) {
                    for (i, &id) in col.ids.iter().enumerate() {
                        if !visited.insert(id) {
                            continue;
                        }
                        let dist = if let Some(v) = self.vectors.get(&id) {
                            distance(self.metric, query, v)
                        } else if i < col.quantized.len() {
                            quant::quantized_distance(
                                self.metric,
                                query,
                                &col.quantized[i],
                                col.quant_tier,
                                self.dimension,
                            )
                        } else {
                            continue;
                        };
                        heap.push(Candidate { id, distance: dist });
                        found_any = true;
                    }
                }
            }

            if depth < max_depth && !frontier.is_empty() {
                let mut child_frontier = Vec::new();
                for (cell, score) in &frontier {
                    if let Some(child) = self.grid.child_cell(cell) {
                        if self.column_at(child).is_some() {
                            child_frontier.push((child, *score));
                        }
                    }
                }
                if !child_frontier.is_empty() {
                    revert_stack.push_back(frontier[0].0);
                    depth += 1;
                    frontier = child_frontier;
                    continue;
                }
            }

            if found_any && heap.len() >= top_k {
                break;
            }

            // Callback reverse: ascend and try sibling columns
            if let Some(parent_cell) = revert_stack.pop_back() {
                self.stats.revert_count += 1;
                depth = parent_cell.layer;
                let siblings = self.sibling_columns(parent_cell, query);
                if siblings.is_empty() {
                    break;
                }
                frontier = siblings;
                frontier.truncate(beam_width);
            } else if depth > 0 {
                self.stats.revert_count += 1;
                depth -= 1;
                frontier = self.score_layer_columns(depth, query);
                frontier.truncate(beam_width);
            } else {
                break;
            }
        }

        // Fallback: scan all columns if heap is underfilled
        if heap.len() < top_k {
            for col in self.columns {
                for &id in &col.ids {
                    if visited.insert(id) {
                        if let Some(v) = self.vectors.get(&id) {
                            let dist = distance(self.metric, query, v);
                            heap.push(Candidate { id, distance: dist });
                        }
                    }
                }
            }
        }

        heap.into_sorted_vec()
            .into_iter()
            .map(|c| (c.id, c.distance))
            .collect()
    }

    fn column_at(&self, cell: CellCoord) -> Option<&ColumnStack> {
        self.columns.iter().find(|c| c.cell() == Some(&cell))
    }

    fn score_layer_columns(&self, layer: u8, query: &[f32]) -> Vec<(CellCoord, f32)> {
        self.columns
            .iter()
            .filter_map(|c| {
                let cell = c.cell()?;
                if cell.layer != layer || c.centroid.is_empty() {
                    return None;
                }
                let dist = distance(self.metric, query, &c.centroid);
                Some((*cell, dist))
            })
            .collect()
    }

    fn sibling_columns(&self, parent: CellCoord, query: &[f32]) -> Vec<(CellCoord, f32)> {
        let layer = parent.layer;
        self.columns
            .iter()
            .filter_map(|c| {
                let cell = c.cell()?;
                if cell.layer != layer || *cell == parent || c.centroid.is_empty() {
                    return None;
                }
                let dist = distance(self.metric, query, &c.centroid);
                Some((*cell, dist))
            })
            .collect()
    }
}
