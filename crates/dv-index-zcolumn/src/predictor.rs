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

        if best_dist <= self.specificity_threshold {
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
        let grid = FractalGrid::new((4, 4), 2, 0.5);
        let col = ColumnStack::new(
            ColumnPath::from_cell(crate::grid::CellCoord::new(1, 0, 0)),
            2,
            QuantTier::F32,
        );
        let predictor = LayerPredictor::default_predictor();
        let layer = predictor.entry_layer(&[10.0, 10.0], &grid, &[&col], DistanceMetric::L2);
        assert_eq!(layer, 0);
    }
}
