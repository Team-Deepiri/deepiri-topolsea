use crate::client::ShardQueryClient;
use crate::protocol::{ShardQueryHit, ShardQueryRequest, ShardRemoteError};
use rayon::prelude::*;
use std::collections::HashMap;

/// One shard target for fan-out.
#[derive(Debug, Clone)]
pub struct ShardFanoutRequest {
    pub shard_id: usize,
    pub endpoint: String,
    pub request: ShardQueryRequest,
}

/// Merged partial result from a single shard.
#[derive(Debug, Clone)]
pub struct ShardFanoutResult {
    pub shard_id: usize,
    pub hits: Vec<ShardQueryHit>,
}

/// Fan out queries to remote shard nodes in parallel.
pub fn fan_out_shard_queries(
    targets: &[ShardFanoutRequest],
    timeout_ms: u64,
) -> Result<Vec<ShardFanoutResult>, ShardRemoteError> {
    let client = ShardQueryClient::new(timeout_ms);
    targets
        .par_iter()
        .map(|target| {
            let response = client.query(&target.endpoint, &target.request)?;
            Ok(ShardFanoutResult {
                shard_id: target.shard_id,
                hits: response.hits,
            })
        })
        .collect()
}

/// Merge remote hits into top-k by distance (lower is better).
pub fn merge_remote_hits(hits: Vec<ShardQueryHit>, top_k: usize) -> Vec<ShardQueryHit> {
    let mut all = hits;
    all.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    all.truncate(top_k);
    all
}

/// Resolve endpoint URLs for shard ids from cluster map.
pub fn endpoints_for_shards(
    shard_ids: &[usize],
    endpoints: &HashMap<usize, String>,
) -> Vec<(usize, String)> {
    shard_ids
        .iter()
        .filter_map(|id| endpoints.get(id).map(|url| (*id, url.clone())))
        .collect()
}
