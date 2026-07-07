use dv_index_zcolumn::shard_id_for_vector;
use dv_storage::{parse_physical_shard_name, ShardManifest};
use dv_types::IndexKind;

use crate::query::QueryResult;

/// Fractal shard routing — column key is the partition primitive (M4).
pub struct FractalShardRouter;

impl FractalShardRouter {
    pub fn route_vector(manifest: &ShardManifest, vector: &[f32]) -> usize {
        if manifest.index_kind == IndexKind::ZColumn {
            shard_id_for_vector(
                manifest.dimension,
                manifest.zcolumn.projection_seed,
                manifest.zcolumn.outer_grid,
                manifest.zcolumn.max_layers,
                manifest.zcolumn.pitch_ratio,
                vector,
                manifest.num_shards,
            )
        } else {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            for x in vector {
                x.to_bits().hash(&mut hasher);
            }
            (hasher.finish() as usize) % manifest.num_shards.max(1)
        }
    }

    pub fn physical_name(manifest: &ShardManifest, shard_id: usize) -> String {
        manifest.physical_name(shard_id)
    }
}

/// Merge per-shard top-k results into a global top-k by distance.
pub fn merge_shard_results(mut partial: Vec<QueryResult>, top_k: usize) -> Vec<QueryResult> {
    partial.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    partial.truncate(top_k);
    partial
}

pub fn is_physical_shard_collection(name: &str) -> bool {
    parse_physical_shard_name(name).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use dv_types::{CollectionConfig, DistanceMetric};

    #[test]
    fn routes_deterministically() {
        let config = CollectionConfig::new("docs", 8, DistanceMetric::Cosine).with_zcolumn_index();
        let manifest = ShardManifest::new("docs", 4, &config);
        let v = vec![0.1; 8];
        let a = FractalShardRouter::route_vector(&manifest, &v);
        let b = FractalShardRouter::route_vector(&manifest, &v);
        assert_eq!(a, b);
        assert!(a < 4);
    }
}
