//! HTTP/gRPC-style shard query fan-out for cross-node Z-Column routing.

mod client;
mod coordinator;
mod protocol;

pub use client::ShardQueryClient;
pub use coordinator::{
    endpoints_for_shards, fan_out_shard_queries, merge_remote_hits, ShardFanoutRequest,
    ShardFanoutResult,
};
pub use protocol::{
    ShardQueryHit, ShardQueryRequest, ShardQueryResponse, ShardRemoteError, QUERY_PATH,
};
