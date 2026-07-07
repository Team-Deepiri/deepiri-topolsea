use crate::protocol::{ShardQueryRequest, ShardQueryResponse, ShardRemoteError, QUERY_PATH};

/// HTTP client for remote shard queries.
pub struct ShardQueryClient {
    timeout_ms: u64,
}

impl Default for ShardQueryClient {
    fn default() -> Self {
        Self { timeout_ms: 30_000 }
    }
}

impl ShardQueryClient {
    pub fn new(timeout_ms: u64) -> Self {
        Self { timeout_ms }
    }

    pub fn query(
        &self,
        base_url: &str,
        request: &ShardQueryRequest,
    ) -> Result<ShardQueryResponse, ShardRemoteError> {
        let url = format!("{}{}", base_url.trim_end_matches('/'), QUERY_PATH);
        let agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_millis(self.timeout_ms))
            .build();
        let response = agent
            .post(&url)
            .set("Content-Type", "application/json")
            .send_json(request)
            .map_err(|e| ShardRemoteError::Transport(e.to_string()))?;

        let status = response.status();
        let body = response
            .into_string()
            .map_err(|e| ShardRemoteError::Transport(e.to_string()))?;

        if status != 200 {
            return Err(ShardRemoteError::Http { status, body });
        }

        serde_json::from_str(&body).map_err(|e| ShardRemoteError::Serde(e.to_string()))
    }
}
