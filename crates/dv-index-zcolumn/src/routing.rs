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

/// Parameters for fractal query → shard routing.
#[derive(Debug, Clone, Copy)]
pub struct ShardQueryRoute {
    pub dimension: usize,
    pub projection_seed: u64,
    pub outer_grid: (u16, u16),
    pub max_layers: u8,
    pub pitch_ratio: f32,
    pub num_shards: usize,
    pub beam_radius: u16,
}

/// Shards to probe for a query — primary shard + beam neighborhood (not all shards).
pub fn shard_ids_for_query(
    query: &[f32],
    route: &ShardQueryRoute,
    placements: &std::collections::HashMap<String, u8>,
) -> Vec<usize> {
    use std::collections::HashSet;
    if route.beam_radius == 0 {
        return vec![shard_id_for_vector(
            route.dimension,
            route.projection_seed,
            route.outer_grid,
            route.max_layers,
            route.pitch_ratio,
            query,
            route.num_shards,
        )];
    }

    let projection = RoutingProjection::new(route.dimension, route.projection_seed);
    let grid = FractalGrid::new(route.outer_grid, route.max_layers, route.pitch_ratio);
    let (px, py) = projection.project(query);

    let mut shards = HashSet::new();
    shards.insert(shard_id_for_vector(
        route.dimension,
        route.projection_seed,
        route.outer_grid,
        route.max_layers,
        route.pitch_ratio,
        query,
        route.num_shards,
    ));

    for cell in grid.cells_in_neighborhood(px, py, route.beam_radius) {
        let key = cell.to_string();
        if let Some(&shard) = placements.get(&key) {
            shards.insert(shard as usize);
        } else {
            shards.insert(shard_id_for_column_key(&key, route.num_shards));
        }
    }

    let mut out: Vec<_> = shards.into_iter().collect();
    out.sort_unstable();
    out
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
    fn query_beam_routing() {
        let config = dv_types::ZColumnConfig::default();
        let mut placements = std::collections::HashMap::new();
        for i in 0..64u64 {
            let v: Vec<f32> = (0..8).map(|d| (i as f32 * 0.07 + d as f32).sin()).collect();
            let key = column_key_for_vector(
                8,
                config.projection_seed,
                config.outer_grid,
                config.max_layers,
                config.pitch_ratio,
                &v,
            );
            let shard = shard_id_for_column_key(&key, 16) as u8;
            placements.insert(key, shard);
        }
        let query = vec![0.2; 8];
        let route = ShardQueryRoute {
            dimension: 8,
            projection_seed: config.projection_seed,
            outer_grid: config.outer_grid,
            max_layers: config.max_layers,
            pitch_ratio: config.pitch_ratio,
            num_shards: 16,
            beam_radius: 0,
        };
        let primary = shard_ids_for_query(&query, &route, &placements);
        assert_eq!(primary.len(), 1);

        let beam_route = ShardQueryRoute {
            beam_radius: 1,
            ..route
        };
        let beam = shard_ids_for_query(&query, &beam_route, &placements);
        assert!(!beam.is_empty());
        assert!(beam.len() < 16);
    }
}
