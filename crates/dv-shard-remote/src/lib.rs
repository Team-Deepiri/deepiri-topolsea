//! HTTP/gRPC-style shard query fan-out for cross-node Z-Column routing.

mod circuit;
mod client;
mod coordinator;
mod protocol;

pub use circuit::CircuitBreakerRegistry;
pub use client::ShardQueryClient;
pub use coordinator::{
    endpoints_for_shards, fan_out_shard_queries, fan_out_shard_queries_with_breaker,
    fanout_targets_from_cluster, merge_remote_hits, ShardFanoutRequest, ShardFanoutResult,
};
pub use protocol::{
    ReplicateDeleteRequest, ReplicateDeleteResponse, ReplicateUpsertRequest,
    ReplicateUpsertResponse, ShardHealthResponse, ShardQueryHit, ShardQueryRequest,
    ShardQueryResponse, ShardRemoteError, QUERY_PATH, REPLICATE_DELETE_PATH, REPLICATE_UPSERT_PATH,
    SHARD_HEALTH_PATH,
};
