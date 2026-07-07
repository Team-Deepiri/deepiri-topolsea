use crate::column::ColumnStack;
use crate::explain::QueryExplain;
use crate::grid::FractalGrid;
use dv_metrics::distance;
use dv_types::DistanceMetric;
use serde::{Deserialize, Serialize};

const FEATURES: usize = 4;
const LEARNING_RATE: f32 = 0.05;

/// Online-learned layer entry model (replaces static heuristics after warm-up).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictorState {
    pub layer_bias: Vec<f32>,
    pub feature_weights: [f32; FEATURES],
    pub layer_hit_ema: Vec<f32>,
    pub specificity_threshold: f32,
    pub queries_trained: u64,
}

impl Default for PredictorState {
    fn default() -> Self {
        Self {
            layer_bias: vec![0.0],
            feature_weights: [0.55, 0.45, -0.15, 0.25],
            layer_hit_ema: vec![0.5],
            specificity_threshold: 0.3,
            queries_trained: 0,
        }
    }
}

impl PredictorState {
    pub fn ensure_layers(&mut self, max_layer: u8) {
        let n = max_layer as usize + 1;
        if self.layer_bias.len() < n {
            self.layer_bias.resize(n, 0.0);
            self.layer_hit_ema.resize(n, 0.5);
        }
    }
}

/// Learned entry-layer predictor with per-query online updates.
pub struct LayerPredictor {
    state: PredictorState,
}

impl LayerPredictor {
    pub fn new(specificity_threshold: f32) -> Self {
        let state = PredictorState {
            specificity_threshold,
            ..PredictorState::default()
        };
        Self { state }
    }

    pub fn default_predictor() -> Self {
        Self::new(0.3)
    }

    pub fn with_state(state: PredictorState) -> Self {
        Self { state }
    }

    pub fn state(&self) -> &PredictorState {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut PredictorState {
        &mut self.state
    }

    fn query_specificity(query: &[f32]) -> f32 {
        if query.is_empty() {
            return 0.0;
        }
        let mean = query.iter().sum::<f32>() / query.len() as f32;
        let variance = query.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / query.len() as f32;
        let norm = query.iter().map(|v| v * v).sum::<f32>().sqrt();
        let spread_penalty = 1.0 / (1.0 + variance * 4.0);
        let norm_factor = (norm / (norm + 1.0)).clamp(0.0, 1.0);
        (spread_penalty * 0.6 + norm_factor * 0.4).clamp(0.0, 1.0)
    }

    fn layer_features(
        query: &[f32],
        layer: u8,
        max_layer: u8,
        min_centroid_dist: f32,
        hit_ema: f32,
    ) -> [f32; FEATURES] {
        let specificity = Self::query_specificity(query);
        let centroid_score = if min_centroid_dist.is_finite() {
            1.0 / (1.0 + min_centroid_dist)
        } else {
            0.0
        };
        let depth = if max_layer > 0 {
            layer as f32 / max_layer as f32
        } else {
            0.0
        };
        [specificity, centroid_score, depth, hit_ema]
    }

    fn score_layer(features: &[f32; FEATURES], bias: f32, weights: &[f32; FEATURES]) -> f32 {
        let mut logit = bias;
        for (f, w) in features.iter().zip(weights.iter()) {
            logit += f * w;
        }
        (1.0 / (1.0 + (-logit).exp())).clamp(0.0, 1.0)
    }

    /// Returns the starting layer index (0 = outermost).
    pub fn entry_layer(
        &mut self,
        query: &[f32],
        grid: &FractalGrid,
        columns: &[&ColumnStack],
        metric: DistanceMetric,
    ) -> u8 {
        if columns.is_empty() {
            return 0;
        }

        let max_layer = grid.num_layers().saturating_sub(1) as u8;
        self.state.ensure_layers(max_layer);

        let mut best_layer = 0u8;
        let mut best_score = f32::MIN;

        for layer in (0..=max_layer).rev() {
            let layer_cols: Vec<_> = columns
                .iter()
                .filter(|c| c.cell().map(|cell| cell.layer) == Some(layer))
                .collect();
            if layer_cols.is_empty() {
                continue;
            }
            let min_dist = layer_cols
                .iter()
                .map(|c| {
                    if c.centroid.is_empty() {
                        f32::MAX
                    } else {
                        distance(metric, query, &c.centroid)
                    }
                })
                .fold(f32::MAX, f32::min);

            let hit_ema = self.state.layer_hit_ema.get(layer as usize).copied().unwrap_or(0.5);
            let features = Self::layer_features(query, layer, max_layer, min_dist, hit_ema);
            let bias = self.state.layer_bias.get(layer as usize).copied().unwrap_or(0.0);
            let score = Self::score_layer(&features, bias, &self.state.feature_weights);

            if score > best_score {
                best_score = score;
                best_layer = layer;
            }
        }

        if best_score >= self.state.specificity_threshold {
            best_layer
        } else {
            0
        }
    }

    /// Online SGD update from a completed query.
    pub fn observe(
        &mut self,
        query: &[f32],
        grid: &FractalGrid,
        columns: &[&ColumnStack],
        metric: DistanceMetric,
        explain: &QueryExplain,
    ) {
        let max_layer = grid.num_layers().saturating_sub(1) as u8;
        self.state.ensure_layers(max_layer);
        self.state.queries_trained += 1;

        let entry = explain.entry_layer;
        let target_layer = if explain.revert_count > 0 || explain.used_fallback_scan {
            0u8
        } else {
            explain.deepest_layer_reached.min(entry)
        };

        let ema_alpha = 0.1;
        for layer in 0..=max_layer {
            let idx = layer as usize;
            let hit = if layer == target_layer { 1.0 } else { 0.0 };
            let prev = self.state.layer_hit_ema[idx];
            self.state.layer_hit_ema[idx] = prev * (1.0 - ema_alpha) + hit * ema_alpha;
        }

        let layer_cols: Vec<_> = columns
            .iter()
            .filter(|c| c.cell().map(|cell| cell.layer) == Some(entry))
            .collect();
        let min_dist = if layer_cols.is_empty() {
            f32::MAX
        } else {
            layer_cols
                .iter()
                .map(|c| distance(metric, query, &c.centroid))
                .fold(f32::MAX, f32::min)
        };
        let hit_ema = self.state.layer_hit_ema[entry as usize];
        let features = Self::layer_features(query, entry, max_layer, min_dist, hit_ema);

        let target = if target_layer == entry { 1.0 } else { 0.0 };
        let bias = self.state.layer_bias[entry as usize];
        let pred = Self::score_layer(&features, bias, &self.state.feature_weights);
        let error = pred - target;

        self.state.layer_bias[entry as usize] -= LEARNING_RATE * error;
        for (w, f) in self.state.feature_weights.iter_mut().zip(features.iter()) {
            *w -= LEARNING_RATE * error * f;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{ColumnPath, FractalGrid};
    use dv_types::QuantTier;

    #[test]
    fn generic_query_starts_outer() {
        let grid = FractalGrid::new((4, 4), 3, 0.5);
        let mut col = ColumnStack::new(
            ColumnPath::from_cell(crate::grid::CellCoord::new(0, 1, 1)),
            2,
            QuantTier::F32,
        );
        col.push(dv_types::VectorId(1), &[1.0, 0.0]);
        let mut predictor = LayerPredictor::default_predictor();
        let layer = predictor.entry_layer(&[0.0, 1.0], &grid, &[&col], DistanceMetric::L2);
        assert_eq!(layer, 0);
    }

    #[test]
    fn peaked_query_can_tunnel_inward() {
        let grid = FractalGrid::new((4, 4), 3, 0.5);
        let mut col = ColumnStack::new(
            ColumnPath::from_cell(crate::grid::CellCoord::new(2, 1, 1)),
            4,
            QuantTier::F32,
        );
        col.push(dv_types::VectorId(1), &[1.0, 0.0, 0.0, 0.0]);
        let mut predictor = LayerPredictor::new(0.2);
        let layer =
            predictor.entry_layer(&[1.0, 0.0, 0.0, 0.0], &grid, &[&col], DistanceMetric::L2);
        assert!(layer >= 1);
    }

    #[test]
    fn observe_updates_weights() {
        let grid = FractalGrid::new((4, 4), 3, 0.5);
        let mut col = ColumnStack::new(
            ColumnPath::from_cell(crate::grid::CellCoord::new(1, 1, 1)),
            2,
            QuantTier::F32,
        );
        col.push(dv_types::VectorId(1), &[1.0, 0.0]);
        let mut predictor = LayerPredictor::default_predictor();
        let before = predictor.state().feature_weights;
        let mut explain = QueryExplain::new("test");
        explain.entry_layer = 2;
        explain.deepest_layer_reached = 2;
        explain.revert_count = 3;
        predictor.observe(
            &[1.0, 0.0],
            &grid,
            &[&col],
            DistanceMetric::L2,
            &explain,
        );
        assert_ne!(predictor.state().feature_weights, before);
        assert_eq!(predictor.state().queries_trained, 1);
    }
}
