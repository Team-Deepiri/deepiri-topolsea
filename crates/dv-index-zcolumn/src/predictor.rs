use crate::column::ColumnStack;
use crate::grid::FractalGrid;
use dv_metrics::distance;
use dv_types::DistanceMetric;

/// Predicts which fractal layer to start search at (predictive revert entry point).
pub struct LayerPredictor {
    specificity_threshold: f32,
}

impl LayerPredictor {
    pub fn new(specificity_threshold: f32) -> Self {
        Self {
            specificity_threshold,
        }
    }

    pub fn default_predictor() -> Self {
        Self::new(0.3)
    }

    /// Query specificity: low variance / high norm → tunnel inward; diffuse → start outer.
    fn query_specificity(query: &[f32]) -> f32 {
        if query.is_empty() {
            return 0.0;
        }
        let mean = query.iter().sum::<f32>() / query.len() as f32;
        let variance = query.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / query.len() as f32;
        let norm = query.iter().map(|v| v * v).sum::<f32>().sqrt();
        // Peaked queries (low variance, moderate norm) score higher.
        let spread_penalty = 1.0 / (1.0 + variance * 4.0);
        let norm_factor = (norm / (norm + 1.0)).clamp(0.0, 1.0);
        (spread_penalty * 0.6 + norm_factor * 0.4).clamp(0.0, 1.0)
    }

    /// Returns the starting layer index (0 = outermost).
    /// Generic queries start outer; specific queries tunnel inward.
    pub fn entry_layer(
        &self,
        query: &[f32],
        grid: &FractalGrid,
        columns: &[&ColumnStack],
        metric: DistanceMetric,
    ) -> u8 {
        if columns.is_empty() {
            return 0;
        }

        let max_layer = grid.num_layers().saturating_sub(1) as u8;
        let specificity = Self::query_specificity(query);

        let mut best_layer = 0u8;
        let mut best_dist = f32::MAX;

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

            if min_dist < best_dist {
                best_dist = min_dist;
                best_layer = layer;
            }
        }

        // Blend centroid proximity with query-shape specificity (M6 heuristic).
        let centroid_score = if best_dist.is_finite() {
            1.0 / (1.0 + best_dist)
        } else {
            0.0
        };
        let combined = specificity * 0.55 + centroid_score * 0.45;

        if combined >= self.specificity_threshold {
            best_layer
        } else {
            0
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
            ColumnPath::from_cell(crate::grid::CellCoord::new(2, 1, 1)),
            2,
            QuantTier::F32,
        );
        col.push(dv_types::VectorId(1), &[1.0, 0.0]);
        let predictor = LayerPredictor::default_predictor();
        // Orthogonal diffuse query should not tunnel to inner layer.
        let layer =
            predictor.entry_layer(&[0.0, 1.0], &grid, &[&col], DistanceMetric::L2);
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
        let predictor = LayerPredictor::new(0.2);
        let layer =
            predictor.entry_layer(&[1.0, 0.0, 0.0, 0.0], &grid, &[&col], DistanceMetric::L2);
        assert!(layer >= 1);
    }
}
