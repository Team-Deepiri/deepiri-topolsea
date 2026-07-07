use crate::column::ColumnStack;
use crate::explain::QueryExplain;
use crate::grid::{CellCoord, FractalGrid};
use crate::predictor::LayerPredictor;
use dv_metrics::{distance, scan_column_distances};
use dv_topk::{Candidate, TopKHeap};
use dv_types::{DistanceMetric, VectorId};
use std::collections::{HashMap, HashSet, VecDeque};

/// Search statistics for benchmarking revert behavior.
#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    pub revert_count: u64,
    pub columns_scanned: u64,
}

/// Beam search with callback-reverse backtracking on miss.
pub struct RevertBeamSearch<'a> {
    grid: &'a FractalGrid,
    columns: &'a HashMap<String, ColumnStack>,
    vectors: &'a HashMap<VectorId, Vec<f32>>,
    metric: DistanceMetric,
    dimension: usize,
    predictor: LayerPredictor,
    stats: &'a mut SearchStats,
}

impl<'a> RevertBeamSearch<'a> {
    pub fn new(
        grid: &'a FractalGrid,
        columns: &'a HashMap<String, ColumnStack>,
        vectors: &'a HashMap<VectorId, Vec<f32>>,
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
        self.run_with_explain(query, top_k, ef).0
    }

    pub fn run_with_explain(
        &mut self,
        query: &[f32],
        top_k: usize,
        ef: usize,
    ) -> (Vec<(VectorId, f32)>, QueryExplain) {
        let mut explain = QueryExplain::new("predictive_revert_beam");
        if self.columns.is_empty() || top_k == 0 {
            return (Vec::new(), explain);
        }

        let col_refs: Vec<&ColumnStack> = self.columns.values().collect();
        let start_layer = self
            .predictor
            .entry_layer(query, self.grid, &col_refs, self.metric);
        explain.entry_layer = start_layer;

        let beam_width = ef.max(top_k).max(1);
        let mut heap = TopKHeap::new(top_k);
        let mut visited: HashSet<VectorId> = HashSet::new();
        let mut revert_stack: VecDeque<CellCoord> = VecDeque::new();
        let mut visited_cells: HashSet<(u8, u16, u16)> = HashSet::new();

        let mut frontier: Vec<(CellCoord, f32)> = self.score_layer_columns(start_layer, query);
        frontier.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        frontier.truncate(beam_width);

        let mut depth = start_layer;
        let max_depth = self.grid.num_layers().saturating_sub(1) as u8;
        explain.deepest_layer_reached = depth;

        loop {
            let mut found_any = false;

            for (cell, _) in &frontier {
                self.stats.columns_scanned += 1;
                visited_cells.insert(cell.key());
                if let Some(col) = self.column_at(*cell) {
                    found_any |= self.scan_column(col, query, &mut visited, &mut heap);
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
                    explain.deepest_layer_reached = depth;
                    frontier = child_frontier;
                    continue;
                }
            }

            if found_any && heap.len() >= top_k {
                break;
            }

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

        if heap.len() < top_k {
            explain.used_fallback_scan = true;
            for col in self.columns.values() {
                self.scan_column(col, query, &mut visited, &mut heap);
            }
        }

        explain.revert_count = self.stats.revert_count;
        explain.columns_scanned = self.stats.columns_scanned;
        explain.candidate_pool = visited.len();
        explain.column_paths = visited_cells
            .iter()
            .map(|(l, x, y)| format!("{l}:{x}:{y}"))
            .collect();

        let results = heap
            .into_sorted_vec()
            .into_iter()
            .map(|c| (c.id, c.distance))
            .collect();
        (results, explain)
    }

    /// Batch-scan an entire column stack (quantized SIMD path + FP32 override when resident).
    fn scan_column(
        &self,
        col: &ColumnStack,
        query: &[f32],
        visited: &mut HashSet<VectorId>,
        heap: &mut TopKHeap,
    ) -> bool {
        let mut found = false;
        let quantized_dists = scan_column_distances(
            self.metric,
            query,
            col.quant_tier,
            &col.quantized,
            self.dimension,
        );
        for (i, &id) in col.ids.iter().enumerate() {
            if !visited.insert(id) {
                continue;
            }
            let dist = if let Some(v) = self.vectors.get(&id) {
                distance(self.metric, query, v)
            } else if i < quantized_dists.len() {
                quantized_dists[i]
            } else {
                continue;
            };
            heap.push(Candidate { id, distance: dist });
            found = true;
        }
        found
    }

    fn column_at(&self, cell: CellCoord) -> Option<&ColumnStack> {
        self.columns.get(&cell.to_string())
    }

    fn score_layer_columns(&self, layer: u8, query: &[f32]) -> Vec<(CellCoord, f32)> {
        self.columns
            .values()
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
            .values()
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
