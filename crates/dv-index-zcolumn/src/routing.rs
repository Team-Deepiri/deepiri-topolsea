use crate::grid::{CellCoord, FractalGrid};
use crate::projection::RoutingProjection;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Fractal column key for a vector (layer:x:y).
pub fn column_key_for_vector(
    dimension: usize,
    projection_seed: u64,
    outer_grid: (u16, u16),
    max_layers: u8,
    pitch_ratio: f32,
    vector: &[f32],
) -> String {
    let projection = RoutingProjection::new(dimension, projection_seed);
    let grid = FractalGrid::new(outer_grid, max_layers, pitch_ratio);
    let (px, py) = projection.project(vector);
    let cell = grid.deepest_cell(px, py).unwrap_or(CellCoord::new(0, 0, 0));
    cell.to_string()
}

/// Map a fractal column key to a shard id (M4 partition primitive).
pub fn shard_id_for_column_key(column_key: &str, num_shards: usize) -> usize {
    if num_shards <= 1 {
        return 0;
    }
    let mut hasher = DefaultHasher::new();
    column_key.hash(&mut hasher);
    (hasher.finish() as usize) % num_shards
}

/// Route a vector to a shard using fractal column addressing.
pub fn shard_id_for_vector(
    dimension: usize,
    projection_seed: u64,
    outer_grid: (u16, u16),
    max_layers: u8,
    pitch_ratio: f32,
    vector: &[f32],
    num_shards: usize,
) -> usize {
    let key = column_key_for_vector(
        dimension,
        projection_seed,
        outer_grid,
        max_layers,
        pitch_ratio,
        vector,
    );
    shard_id_for_column_key(&key, num_shards)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_is_deterministic() {
        let v = vec![0.1, 0.2, 0.3, 0.4];
        let a = shard_id_for_vector(4, 42, (8, 8), 3, 0.5, &v, 8);
        let b = shard_id_for_vector(4, 42, (8, 8), 3, 0.5, &v, 8);
        assert_eq!(a, b);
    }

    #[test]
    fn shards_spread_across_keys() {
        let mut seen = std::collections::HashSet::new();
        for i in 0..64u64 {
            let v: Vec<f32> = (0..8).map(|d| (i as f32 * 0.1 + d as f32).sin()).collect();
            let shard = shard_id_for_vector(8, 42, (8, 8), 3, 0.5, &v, 8);
            seen.insert(shard);
        }
        assert!(seen.len() > 1);
    }
}
