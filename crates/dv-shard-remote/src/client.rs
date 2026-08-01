use crate::circuit::CircuitBreakerRegistry;
use crate::protocol::{
    ReplicateDeleteRequest, ReplicateDeleteResponse, ReplicateUpsertRequest,
    ReplicateUpsertResponse, ShardHealthResponse, ShardQueryRequest, ShardQueryResponse,
    ShardRemoteError, QUERY_PATH, REPLICATE_DELETE_PATH, REPLICATE_UPSERT_PATH, SHARD_HEALTH_PATH,
};
use std::thread;
use std::time::Duration;

/// HTTP client for remote shard queries with retries + circuit breaker (C11).
pub struct ShardQueryClient {
    timeout_ms: u64,
    max_retries: u32,
    breaker: CircuitBreakerRegistry,
}

impl Default for ShardQueryClient {
    fn default() -> Self {
        Self::new(30_000)
    }
}

impl ShardQueryClient {
    pub fn new(timeout_ms: u64) -> Self {
        Self {
            timeout_ms,
            max_retries: 2,
            breaker: CircuitBreakerRegistry::default(),
        }
    }

    pub fn with_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    pub fn with_breaker(mut self, breaker: CircuitBreakerRegistry) -> Self {
        self.breaker = breaker;
        self
    }

    pub fn breaker(&self) -> &CircuitBreakerRegistry {
        &self.breaker
    }

    pub fn query(
        &self,
        base_url: &str,
        request: &ShardQueryRequest,
    ) -> Result<ShardQueryResponse, ShardRemoteError> {
        let url = format!("{}{}", base_url.trim_end_matches('/'), QUERY_PATH);
        self.post_json(&url, base_url, request)
    }

    pub fn replicate_upsert(
        &self,
        base_url: &str,
        request: &ReplicateUpsertRequest,
    ) -> Result<ReplicateUpsertResponse, ShardRemoteError> {
        let url = format!(
            "{}{}",
            base_url.trim_end_matches('/'),
            REPLICATE_UPSERT_PATH
        );
        self.post_json(&url, base_url, request)
    }

    pub fn replicate_delete(
        &self,
        base_url: &str,
        request: &ReplicateDeleteRequest,
    ) -> Result<ReplicateDeleteResponse, ShardRemoteError> {
        let url = format!(
            "{}{}",
            base_url.trim_end_matches('/'),
            REPLICATE_DELETE_PATH
        );
        self.post_json(&url, base_url, request)
    }

    pub fn health(&self, base_url: &str) -> Result<ShardHealthResponse, ShardRemoteError> {
        let url = format!("{}{}", base_url.trim_end_matches('/'), SHARD_HEALTH_PATH);
        if !self.breaker.allow(base_url) {
            return Err(ShardRemoteError::CircuitOpen(base_url.into()));
        }
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_millis(self.timeout_ms.min(5_000)))
            .build();
        let response = agent
            .get(&url)
            .call()
            .map_err(|e| ShardRemoteError::Transport(e.to_string()))?;
        let status = response.status();
        let body = response
            .into_string()
            .map_err(|e| ShardRemoteError::Transport(e.to_string()))?;
        if status != 200 {
            self.breaker.on_failure(base_url);
            return Err(ShardRemoteError::Http { status, body });
        }
        self.breaker.on_success(base_url);
        serde_json::from_str(&body).map_err(|e| ShardRemoteError::Serde(e.to_string()))
    }

    /// Try primary then failover endpoints (circuit-aware).
    pub fn query_with_failover(
        &self,
        endpoints: &[String],
        request: &ShardQueryRequest,
    ) -> Result<ShardQueryResponse, ShardRemoteError> {
        let mut last_err = ShardRemoteError::Transport("no endpoints".into());
        for ep in endpoints {
            if !self.breaker.allow(ep) {
                last_err = ShardRemoteError::CircuitOpen(ep.clone());
                continue;
            }
            match self.query(ep, request) {
                Ok(r) => {
                    self.breaker.on_success(ep);
                    return Ok(r);
                }
                Err(e) => {
                    self.breaker.on_failure(ep);
                    last_err = e;
                }
            }
        }
        Err(last_err)
    }

    fn post_json<Req: serde::Serialize, Resp: serde::de::DeserializeOwned>(
        &self,
        url: &str,
        endpoint_key: &str,
        body: &Req,
    ) -> Result<Resp, ShardRemoteError> {
        if !self.breaker.allow(endpoint_key) {
            return Err(ShardRemoteError::CircuitOpen(endpoint_key.into()));
        }

        let mut attempt = 0u32;
        loop {
            match self.post_once(url, body) {
                Ok(v) => {
                    self.breaker.on_success(endpoint_key);
                    return Ok(v);
                }
                Err(e) => {
                    let retryable = matches!(&e, ShardRemoteError::Transport(_))
                        || matches!(&e, ShardRemoteError::Http { status, .. } if *status >= 500);
                    if attempt >= self.max_retries || !retryable {
                        self.breaker.on_failure(endpoint_key);
                        return Err(e);
                    }
                    attempt += 1;
                    let backoff = Duration::from_millis(25 * (1u64 << attempt.min(4)));
                    thread::sleep(backoff);
                }
            }
        }
    }

    fn post_once<Req: serde::Serialize, Resp: serde::de::DeserializeOwned>(
        &self,
        url: &str,
        body: &Req,
    ) -> Result<Resp, ShardRemoteError> {
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_millis(self.timeout_ms))
            .build();
        let response = agent
            .post(url)
            .set("Content-Type", "application/json")
            .send_json(body)
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
