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

/// Live map of fractal column keys → owning shard (enables query-beam routing).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShardRoutingIndex {
    pub placements: std::collections::HashMap<String, u8>,
    pub beam_radius: u16,
}

/// Remote shard node endpoints for cross-node fan-out (`shard_id` → base URL).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShardClusterConfig {
    /// Primary endpoint per shard.
    pub endpoints: std::collections::HashMap<usize, String>,
    /// Backup replica endpoints per shard (tried on primary failure) — C10.
    #[serde(default)]
    pub replicas: std::collections::HashMap<usize, Vec<String>>,
    /// When true, sharded upsert fails if any replica ack fails (sync durability).
    #[serde(default)]
    pub require_replica_ack: bool,
    /// Timeout for replica sync RPCs (ms).
    #[serde(default = "default_replica_timeout_ms")]
    pub replica_timeout_ms: u64,
}

fn default_replica_timeout_ms() -> u64 {
    10_000
}

impl ShardClusterConfig {
    pub fn endpoints_for_shard(&self, shard_id: usize) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(p) = self.endpoints.get(&shard_id) {
            out.push(p.clone());
        }
        if let Some(reps) = self.replicas.get(&shard_id) {
            for r in reps {
                if !out.iter().any(|x| x == r) {
                    out.push(r.clone());
                }
            }
        }
        out
    }

    pub fn add_replica(&mut self, shard_id: usize, url: impl Into<String>) {
        self.replicas.entry(shard_id).or_default().push(url.into());
    }
}

/// Cluster membership registry (C10) — nodes that can host shard replicas.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClusterMembership {
    pub nodes: Vec<ClusterNode>,
    pub generation: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterNode {
    pub id: String,
    pub advertise_url: String,
    #[serde(default)]
    pub role: NodeRole,
    /// Last heartbeat unix millis (0 = never).
    #[serde(default)]
    pub last_heartbeat_ms: u64,
    #[serde(default = "default_node_healthy")]
    pub healthy: bool,
}

fn default_node_healthy() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NodeRole {
    #[default]
    Data,
    Coordinator,
}

impl ShardRoutingIndex {
    pub fn new(beam_radius: u16) -> Self {
        Self {
            placements: std::collections::HashMap::new(),
            beam_radius,
        }
    }

    pub fn record(&mut self, column_key: impl Into<String>, shard_id: u8) {
        self.placements.insert(column_key.into(), shard_id);
    }
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
