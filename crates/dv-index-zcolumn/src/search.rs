use crate::centroid_graph::CentroidGraph;
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

/// Tunable search parameters (from `ZColumnConfig` + query planner).
#[derive(Debug, Clone, Copy)]
pub struct SearchParams {
    pub coarse_pool: usize,
    pub recall_k: usize,
    pub ef: usize,
    pub query_xy: (f32, f32),
    pub fallback_beam_radius: u16,
    pub max_fallback_rings: u16,
    /// Cap on centroid-ranked column fallback (never full corpus).
    pub max_fallback_columns: usize,
    /// Hard V_touch budget (M2). `0` = unlimited.
    pub touch_budget: usize,
    /// Prefer centroid-graph expansion (M-graph).
    pub use_centroid_graph: bool,
    /// Only fire fallback when needed (M1).
    pub conditional_fallback: bool,
    pub fallback_score_gap: f32,
}

#[derive(Debug, Clone, Copy)]
struct FallbackCtx<'a> {
    query_xy: (f32, f32),
    query: &'a [f32],
    beam_radius: u16,
    max_rings: u16,
    max_columns: usize,
}

/// Beam search with callback-reverse backtracking on miss.
pub struct RevertBeamSearch<'a> {
    grid: &'a FractalGrid,
    columns: &'a HashMap<String, ColumnStack>,
    vectors: &'a HashMap<VectorId, Vec<f32>>,
    metric: DistanceMetric,
    dimension: usize,
    predictor: &'a mut LayerPredictor,
    stats: &'a mut SearchStats,
    /// When set, only score ids that pass this predicate (payload-aware ANN).
    eligible: Option<&'a dyn Fn(VectorId) -> bool>,
    /// Optional prebuilt centroid graph (M-graph).
    graph: Option<&'a CentroidGraph>,
}

impl<'a> RevertBeamSearch<'a> {
    pub fn new(
        grid: &'a FractalGrid,
        columns: &'a HashMap<String, ColumnStack>,
        vectors: &'a HashMap<VectorId, Vec<f32>>,
        metric: DistanceMetric,
        dimension: usize,
        stats: &'a mut SearchStats,
        predictor: &'a mut LayerPredictor,
    ) -> Self {
        Self {
            grid,
            columns,
            vectors,
            metric,
            dimension,
            predictor,
            stats,
            eligible: None,
            graph: None,
        }
    }

    pub fn with_eligible(mut self, eligible: &'a dyn Fn(VectorId) -> bool) -> Self {
        self.eligible = Some(eligible);
        self
    }

    pub fn with_graph(mut self, graph: &'a CentroidGraph) -> Self {
        self.graph = Some(graph);
        self
    }

    pub fn run(&mut self, query: &[f32], params: SearchParams) -> Vec<(VectorId, f32)> {
        self.run_with_explain(query, params).0
    }

    pub fn run_with_explain(
        &mut self,
        query: &[f32],
        params: SearchParams,
    ) -> (Vec<(VectorId, f32)>, QueryExplain) {
        let SearchParams {
            coarse_pool,
            recall_k,
            ef,
            query_xy,
            fallback_beam_radius,
            max_fallback_rings,
            max_fallback_columns,
            touch_budget,
            use_centroid_graph,
            conditional_fallback,
            fallback_score_gap,
        } = params;

        let mut explain = QueryExplain::new("predictive_revert_beam");
        if self.columns.is_empty() || recall_k == 0 {
            return (Vec::new(), explain);
        }

        let col_refs: Vec<&ColumnStack> = self.columns.values().collect();
        let start_layer = self
            .predictor
            .entry_layer(query, self.grid, &col_refs, self.metric);
        explain.entry_layer = start_layer;

        let beam_width = ef.max(recall_k).max(1);
        let heap_cap = coarse_pool.max(recall_k);
        let mut heap = TopKHeap::new(heap_cap);
        let mut visited: HashSet<VectorId> = HashSet::new();
        let mut revert_stack: VecDeque<CellCoord> = VecDeque::new();
        let mut visited_cells: HashSet<(u8, u16, u16)> = HashSet::new();

        // M-graph path: expand via centroid neighbor edges.
        if use_centroid_graph {
            if let Some(graph) = self.graph {
                if !graph.is_empty() {
                    explain.strategy = "centroid_graph_beam".into();
                    self.graph_beam(
                        graph,
                        query,
                        beam_width,
                        touch_budget,
                        &mut visited,
                        &mut visited_cells,
                        &mut heap,
                        &mut explain,
                    );
                }
            }
        }

        // Fractal layer beam (also used when graph is empty / disabled).
        if visited.is_empty() || heap.len() < recall_k {
            if explain.strategy == "centroid_graph_beam" {
                explain.strategy = "centroid_graph_plus_fractal".into();
            }
            let mut frontier: Vec<(CellCoord, f32)> = self.score_layer_columns(start_layer, query);
            frontier.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            frontier.truncate(beam_width);

            let mut depth = start_layer;
            let max_depth = self.grid.num_layers().saturating_sub(1) as u8;
            explain.deepest_layer_reached = depth;

            loop {
                if self.budget_exhausted(touch_budget, &visited) {
                    explain.hit_touch_budget = true;
                    break;
                }
                let mut found_any = false;

                for (cell, _) in &frontier {
                    if self.budget_exhausted(touch_budget, &visited) {
                        explain.hit_touch_budget = true;
                        break;
                    }
                    self.stats.columns_scanned += 1;
                    visited_cells.insert(cell.key());
                    if let Some(col) = self.column_at(*cell) {
                        // M3: quantized coarse filter during beam; FP32 at hybrid rerank.
                        found_any |= self.scan_column_budgeted(
                            col,
                            query,
                            &mut visited,
                            &mut heap,
                            true,
                            touch_budget,
                        );
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

                if found_any && heap.len() >= recall_k {
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
        }

        // M1: conditional fallback — only if heap short or score gap large.
        explain.used_fallback_scan = false;
        let need_fallback = if !conditional_fallback {
            true
        } else {
            self.needs_fallback(&heap, recall_k, fallback_score_gap)
        };
        if need_fallback && !self.budget_exhausted(touch_budget, &visited) {
            let fb = FallbackCtx {
                query_xy,
                query,
                beam_radius: fallback_beam_radius,
                max_rings: max_fallback_rings,
                max_columns: max_fallback_columns,
            };
            if fallback_beam_radius > 0 && max_fallback_rings > 0 {
                explain.used_fallback_scan = true;
                self.neighborhood_fallback(
                    fb,
                    touch_budget,
                    &mut visited,
                    &mut visited_cells,
                    &mut heap,
                    &mut explain,
                );
            }
            if max_fallback_columns > 0 && !self.budget_exhausted(touch_budget, &visited) {
                explain.used_fallback_scan = true;
                self.ranked_column_fallback(
                    fb,
                    touch_budget,
                    &mut visited,
                    &mut visited_cells,
                    &mut heap,
                    &mut explain,
                );
            }
        }

        explain.revert_count = self.stats.revert_count;
        explain.columns_scanned = self.stats.columns_scanned;
        explain.candidate_pool = visited.len();
        explain.touch_budget = touch_budget;
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

    fn budget_exhausted(&self, touch_budget: usize, visited: &HashSet<VectorId>) -> bool {
        touch_budget > 0 && visited.len() >= touch_budget
    }

    fn needs_fallback(&self, heap: &TopKHeap, recall_k: usize, _gap: f32) -> bool {
        // M1: fire only when the beam failed to fill top-k (score-gap reserved for future).
        heap.len() < recall_k
    }

    #[allow(clippy::too_many_arguments)]
    fn graph_beam(
        &mut self,
        graph: &CentroidGraph,
        query: &[f32],
        beam_width: usize,
        touch_budget: usize,
        visited: &mut HashSet<VectorId>,
        visited_cells: &mut HashSet<(u8, u16, u16)>,
        heap: &mut TopKHeap,
        explain: &mut QueryExplain,
    ) {
        let seeds = graph.nearest_seeds(self.columns, self.metric, query, beam_width);
        let mut frontier: VecDeque<String> = seeds.into_iter().map(|(k, _)| k).collect();
        let mut seen_keys: HashSet<String> = HashSet::new();

        while let Some(key) = frontier.pop_front() {
            if self.budget_exhausted(touch_budget, visited) {
                explain.hit_touch_budget = true;
                break;
            }
            if !seen_keys.insert(key.clone()) {
                continue;
            }
            if let Some(col) = self.columns.get(&key) {
                if let Some(cell) = col.cell() {
                    visited_cells.insert(cell.key());
                }
                self.stats.columns_scanned += 1;
                self.scan_column_budgeted(col, query, visited, heap, true, touch_budget);
            }
            for (nbr, _) in graph.neighbors_of(&key) {
                if !seen_keys.contains(nbr) {
                    frontier.push_back(nbr.clone());
                }
            }
            // Bound graph expansion to ~beam_width * degree hops worth of columns.
            if seen_keys.len() >= beam_width.saturating_mul(4).max(beam_width) {
                break;
            }
        }
    }

    /// Expand fractal rings around the query projection — never full corpus.
    fn neighborhood_fallback(
        &mut self,
        ctx: FallbackCtx<'_>,
        touch_budget: usize,
        visited: &mut HashSet<VectorId>,
        visited_cells: &mut HashSet<(u8, u16, u16)>,
        heap: &mut TopKHeap,
        explain: &mut QueryExplain,
    ) {
        let (px, py) = ctx.query_xy;
        let mut ring = ctx.beam_radius.max(1);
        let limit = ctx.max_rings.max(ring);
        while ring <= limit {
            if self.budget_exhausted(touch_budget, visited) {
                explain.hit_touch_budget = true;
                return;
            }
            for cell in self.grid.cells_in_neighborhood(px, py, ring) {
                if self.budget_exhausted(touch_budget, visited) {
                    explain.hit_touch_budget = true;
                    return;
                }
                if !visited_cells.insert(cell.key()) {
                    continue;
                }
                self.stats.columns_scanned += 1;
                if let Some(col) = self.column_at(cell) {
                    self.scan_column_budgeted(col, ctx.query, visited, heap, true, touch_budget);
                }
            }
            ring += 1;
        }
    }

    /// Centroid-ranked column cap — bounded sweep, not O(corpus).
    fn ranked_column_fallback(
        &mut self,
        ctx: FallbackCtx<'_>,
        touch_budget: usize,
        visited: &mut HashSet<VectorId>,
        visited_cells: &mut HashSet<(u8, u16, u16)>,
        heap: &mut TopKHeap,
        explain: &mut QueryExplain,
    ) {
        let mut ranked: Vec<(f32, CellCoord)> = self
            .columns
            .values()
            .filter_map(|c| {
                let cell = c.cell()?;
                if c.centroid.is_empty() {
                    return None;
                }
                let dist = distance(self.metric, ctx.query, &c.centroid);
                Some((dist, *cell))
            })
            .collect();
        ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut scanned_extra = 0usize;
        for (_, cell) in ranked {
            if scanned_extra >= ctx.max_columns {
                break;
            }
            if self.budget_exhausted(touch_budget, visited) {
                explain.hit_touch_budget = true;
                break;
            }
            if !visited_cells.insert(cell.key()) {
                continue;
            }
            self.stats.columns_scanned += 1;
            scanned_extra += 1;
            if let Some(col) = self.column_at(cell) {
                self.scan_column_budgeted(col, ctx.query, visited, heap, true, touch_budget);
            }
        }
    }

    /// Batch-scan a column. Coarse path uses quantized SIMD only; FP32 rerank happens later.
    /// Returns false early if `touch_budget` is hit mid-column (M2).
    fn scan_column_budgeted(
        &self,
        col: &ColumnStack,
        query: &[f32],
        visited: &mut HashSet<VectorId>,
        heap: &mut TopKHeap,
        coarse_only: bool,
        touch_budget: usize,
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
            if self.budget_exhausted(touch_budget, visited) {
                break;
            }
            if let Some(pred) = self.eligible {
                if !pred(id) {
                    continue;
                }
            }
            if !visited.insert(id) {
                continue;
            }
            let dist = if coarse_only {
                if i < quantized_dists.len() {
                    quantized_dists[i]
                } else {
                    continue;
                }
            } else if let Some(v) = self.vectors.get(&id) {
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
