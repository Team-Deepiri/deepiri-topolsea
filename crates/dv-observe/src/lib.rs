//! Observability: Prometheus-style metrics for the Topolsea service.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Process-wide service metrics (Phase C12).
#[derive(Debug, Default)]
pub struct ServiceMetrics {
    pub http_requests_total: AtomicU64,
    pub http_errors_total: AtomicU64,
    pub search_total: AtomicU64,
    pub upsert_total: AtomicU64,
    pub replicate_total: AtomicU64,
    pub snapshot_total: AtomicU64,
    pub shard_fanout_total: AtomicU64,
    pub shard_fanout_errors: AtomicU64,
    /// Sum of request latency micros (for average); paired with http_requests_total.
    pub http_latency_micros_sum: AtomicU64,
    /// Rough WAL lag samples (records pending snapshot), last observed.
    pub wal_lag_last: AtomicU64,
    histograms: Mutex<HashMap<String, Vec<u64>>>,
}

impl ServiceMetrics {
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn record_http(&self, route: &str, status: u16, started: Instant) {
        self.http_requests_total.fetch_add(1, Ordering::Relaxed);
        let micros = started.elapsed().as_micros() as u64;
        self.http_latency_micros_sum
            .fetch_add(micros, Ordering::Relaxed);
        if status >= 400 {
            self.http_errors_total.fetch_add(1, Ordering::Relaxed);
        }
        let mut h = self.histograms.lock();
        h.entry(route.to_string()).or_default().push(micros);
        // Cap samples to keep memory bounded.
        let bucket = h.get_mut(route).unwrap();
        if bucket.len() > 2048 {
            bucket.drain(0..1024);
        }
    }

    pub fn set_wal_lag(&self, lag: u64) {
        self.wal_lag_last.store(lag, Ordering::Relaxed);
    }

    pub fn percentile_micros(&self, route: &str, pct: f64) -> u64 {
        let h = self.histograms.lock();
        let Some(samples) = h.get(route) else {
            return 0;
        };
        if samples.is_empty() {
            return 0;
        }
        let mut sorted = samples.clone();
        sorted.sort_unstable();
        let idx = ((pct.clamp(0.0, 1.0) * (sorted.len() as f64 - 1.0)).round() as usize)
            .min(sorted.len() - 1);
        sorted[idx]
    }

    /// Prometheus text exposition format.
    pub fn render_prometheus(&self) -> String {
        let mut out = String::new();
        macro_rules! gauge {
            ($name:expr, $help:expr, $val:expr) => {{
                out.push_str(&format!("# HELP {} {}\n", $name, $help));
                out.push_str(&format!("# TYPE {} gauge\n", $name));
                out.push_str(&format!("{} {}\n", $name, $val));
            }};
        }
        macro_rules! counter {
            ($name:expr, $help:expr, $val:expr) => {{
                out.push_str(&format!("# HELP {} {}\n", $name, $help));
                out.push_str(&format!("# TYPE {} counter\n", $name));
                out.push_str(&format!("{} {}\n", $name, $val));
            }};
        }
        counter!(
            "topolsea_http_requests_total",
            "Total HTTP requests",
            self.http_requests_total.load(Ordering::Relaxed)
        );
        counter!(
            "topolsea_http_errors_total",
            "Total HTTP 4xx/5xx responses",
            self.http_errors_total.load(Ordering::Relaxed)
        );
        counter!(
            "topolsea_search_total",
            "Search / hybrid / sparse queries",
            self.search_total.load(Ordering::Relaxed)
        );
        counter!(
            "topolsea_upsert_total",
            "Upsert operations",
            self.upsert_total.load(Ordering::Relaxed)
        );
        counter!(
            "topolsea_replicate_total",
            "Replication apply operations",
            self.replicate_total.load(Ordering::Relaxed)
        );
        counter!(
            "topolsea_snapshot_total",
            "Snapshot create operations",
            self.snapshot_total.load(Ordering::Relaxed)
        );
        counter!(
            "topolsea_shard_fanout_total",
            "Remote shard fan-out calls",
            self.shard_fanout_total.load(Ordering::Relaxed)
        );
        counter!(
            "topolsea_shard_fanout_errors_total",
            "Remote shard fan-out failures",
            self.shard_fanout_errors.load(Ordering::Relaxed)
        );
        gauge!(
            "topolsea_http_latency_p99_micros",
            "Approx p99 latency across recent samples (search route if present)",
            self.percentile_micros("search", 0.99)
                .max(self.percentile_micros("http", 0.99))
        );
        gauge!(
            "topolsea_wal_lag_last",
            "Last observed WAL records pending snapshot",
            self.wal_lag_last.load(Ordering::Relaxed)
        );
        let reqs = self.http_requests_total.load(Ordering::Relaxed).max(1);
        let avg = self.http_latency_micros_sum.load(Ordering::Relaxed) / reqs;
        gauge!(
            "topolsea_http_latency_avg_micros",
            "Average HTTP latency microseconds",
            avg
        );
        out
    }
}
