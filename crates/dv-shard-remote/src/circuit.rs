//! Circuit breaker for remote shard endpoints.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Debug)]
struct Breaker {
    state: State,
    failures: u32,
    opened_at: Option<Instant>,
}

impl Default for Breaker {
    fn default() -> Self {
        Self {
            state: State::Closed,
            failures: 0,
            opened_at: None,
        }
    }
}

/// Per-endpoint circuit breaker (failure threshold → cool-down → half-open probe).
#[derive(Debug, Clone)]
pub struct CircuitBreakerRegistry {
    inner: Arc<Mutex<HashMap<String, Breaker>>>,
    failure_threshold: u32,
    open_ms: u64,
}

impl Default for CircuitBreakerRegistry {
    fn default() -> Self {
        Self::new(5, 5_000)
    }
}

impl CircuitBreakerRegistry {
    pub fn new(failure_threshold: u32, open_ms: u64) -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            failure_threshold: failure_threshold.max(1),
            open_ms,
        }
    }

    pub fn allow(&self, endpoint: &str) -> bool {
        let mut map = self.inner.lock().expect("circuit lock");
        let b = map.entry(endpoint.to_string()).or_default();
        match b.state {
            State::Closed | State::HalfOpen => true,
            State::Open => {
                if let Some(at) = b.opened_at {
                    if at.elapsed() >= Duration::from_millis(self.open_ms) {
                        b.state = State::HalfOpen;
                        tracing::info!(endpoint, "circuit half-open");
                        true
                    } else {
                        false
                    }
                } else {
                    true
                }
            }
        }
    }

    pub fn on_success(&self, endpoint: &str) {
        let mut map = self.inner.lock().expect("circuit lock");
        let b = map.entry(endpoint.to_string()).or_default();
        if b.state != State::Closed {
            tracing::info!(endpoint, from = ?b.state, "circuit closed");
        }
        b.failures = 0;
        b.state = State::Closed;
        b.opened_at = None;
    }

    pub fn on_failure(&self, endpoint: &str) {
        let mut map = self.inner.lock().expect("circuit lock");
        let b = map.entry(endpoint.to_string()).or_default();
        b.failures = b.failures.saturating_add(1);
        if b.failures >= self.failure_threshold || b.state == State::HalfOpen {
            if b.state != State::Open {
                tracing::warn!(endpoint, failures = b.failures, "circuit open");
            }
            b.state = State::Open;
            b.opened_at = Some(Instant::now());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opens_after_threshold() {
        let reg = CircuitBreakerRegistry::new(2, 60_000);
        assert!(reg.allow("http://a"));
        reg.on_failure("http://a");
        assert!(reg.allow("http://a"));
        reg.on_failure("http://a");
        assert!(!reg.allow("http://a"));
        reg.on_success("http://a"); // shouldn't reset if we can't call — simulate half-open success path
    }
}
