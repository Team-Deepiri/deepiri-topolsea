use dv_types::{CollectionConfig, DistanceMetric, IndexKind, ZColumnConfig};
use serde::{Deserialize, Serialize};

/// Logical sharded collection — physical shards are `{logical_name}__s{N}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardManifest {
    pub logical_name: String,
    pub num_shards: usize,
    pub dimension: usize,
    pub metric: DistanceMetric,
    pub index_kind: IndexKind,
    #[serde(default)]
    pub zcolumn: ZColumnConfig,
}

impl ShardManifest {
    pub fn new(
        logical_name: impl Into<String>,
        num_shards: usize,
        config: &CollectionConfig,
    ) -> Self {
        Self {
            logical_name: logical_name.into(),
            num_shards,
            dimension: config.dimension,
            metric: config.metric,
            index_kind: config.index_kind,
            zcolumn: config.zcolumn.clone(),
        }
    }

    pub fn physical_name(&self, shard_id: usize) -> String {
        format!("{}__s{shard_id}", self.logical_name)
    }
}

/// Parse `logical__s3` → (`logical`, 3).
pub fn parse_physical_shard_name(name: &str) -> Option<(String, usize)> {
    let (logical, idx_str) = name.rsplit_once("__s")?;
    let idx = idx_str.parse().ok()?;
    Some((logical.to_string(), idx))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_physical_shard() {
        let (logical, idx) = parse_physical_shard_name("docs__s2").unwrap();
        assert_eq!(logical, "docs");
        assert_eq!(idx, 2);
    }
}
