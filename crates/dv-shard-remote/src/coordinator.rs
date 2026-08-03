use crate::circuit::CircuitBreakerRegistry;
use crate::client::ShardQueryClient;
use crate::protocol::{ShardQueryHit, ShardQueryRequest, ShardRemoteError};
use rayon::prelude::*;
use std::collections::HashMap;

/// One shard target for fan-out.
#[derive(Debug, Clone)]
pub struct ShardFanoutRequest {
    pub shard_id: usize,
    /// Ordered failover list: primary first, then replicas.
    pub endpoints: Vec<String>,
    pub request: ShardQueryRequest,
}

/// Merged partial result from a single shard.
#[derive(Debug, Clone)]
pub struct ShardFanoutResult {
    pub shard_id: usize,
    pub hits: Vec<ShardQueryHit>,
}

/// Fan out queries to remote shard nodes in parallel (retries + circuit + failover).
pub fn fan_out_shard_queries(
    targets: &[ShardFanoutRequest],
    timeout_ms: u64,
) -> Result<Vec<ShardFanoutResult>, ShardRemoteError> {
    fan_out_shard_queries_with_breaker(targets, timeout_ms, CircuitBreakerRegistry::default())
}

pub fn fan_out_shard_queries_with_breaker(
    targets: &[ShardFanoutRequest],
    timeout_ms: u64,
    breaker: CircuitBreakerRegistry,
) -> Result<Vec<ShardFanoutResult>, ShardRemoteError> {
    let client = ShardQueryClient::new(timeout_ms)
        .with_retries(2)
        .with_breaker(breaker);
    targets
        .par_iter()
        .map(|target| {
            let response = client.query_with_failover(&target.endpoints, &target.request)?;
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

/// Resolve endpoint URLs for shard ids from cluster map (legacy single-endpoint).
pub fn endpoints_for_shards(
    shard_ids: &[usize],
    endpoints: &HashMap<usize, String>,
) -> Vec<(usize, String)> {
    shard_ids
        .iter()
        .filter_map(|id| endpoints.get(id).map(|url| (*id, url.clone())))
        .collect()
}

/// Build fan-out targets from primary map + optional replica lists.
pub fn fanout_targets_from_cluster(
    shard_ids: &[usize],
    primaries: &HashMap<usize, String>,
    replicas: &HashMap<usize, Vec<String>>,
    request: &ShardQueryRequest,
) -> Vec<ShardFanoutRequest> {
    shard_ids
        .iter()
        .filter_map(|id| {
            let mut eps = Vec::new();
            if let Some(p) = primaries.get(id) {
                eps.push(p.clone());
            }
            if let Some(reps) = replicas.get(id) {
                for r in reps {
                    if !eps.iter().any(|x| x == r) {
                        eps.push(r.clone());
                    }
                }
            }
            if eps.is_empty() {
                None
            } else {
                Some(ShardFanoutRequest {
                    shard_id: *id,
                    endpoints: eps,
                    request: request.clone(),
                })
            }
        })
        .collect()
}
