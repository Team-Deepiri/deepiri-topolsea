//! Blocking helpers to run the axum app (CLI / shard-compat / tests).

use crate::routes::router;
use crate::state::AppState;
use dv_query::{Database, SharedDatabase};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Configuration for a background HTTP server (product + shard fan-out).
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub data_dir: PathBuf,
    pub bind_addr: String,
    pub api_key: Option<String>,
    pub shard_collection: Option<String>,
    pub flush_secs: u64,
}

/// Handle for a server started on a background thread (tokio runtime).
pub struct BackgroundServer {
    shutdown: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    port: u16,
}

impl BackgroundServer {
    pub fn start(config: ServerConfig) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let listener = std::net::TcpListener::bind(&config.bind_addr)?;
        listener.set_nonblocking(true)?;
        let port = listener.local_addr()?.port();
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_flag = shutdown.clone();

        let handle = thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("tokio runtime");
            rt.block_on(async move {
                let db = Database::open(&config.data_dir)
                    .expect("open db")
                    .into_shared();
                if config.flush_secs > 0 {
                    let flush_db = db.clone();
                    let period = Duration::from_secs(config.flush_secs);
                    tokio::spawn(async move {
                        let mut ticker = tokio::time::interval(period);
                        loop {
                            ticker.tick().await;
                            let _ = flush_db.write().persist_all();
                        }
                    });
                }
                let mut state = AppState::new(db, config.api_key);
                if let Some(name) = config.shard_collection {
                    state = state.with_shard_collection(name);
                }
                let app = router(state);
                listener.set_nonblocking(true).ok();
                let listener = tokio::net::TcpListener::from_std(listener).expect("tokio listener");
                let flag = shutdown_flag.clone();
                axum::serve(listener, app)
                    .with_graceful_shutdown(async move {
                        while !flag.load(Ordering::Relaxed) {
                            tokio::time::sleep(Duration::from_millis(20)).await;
                        }
                    })
                    .await
                    .ok();
            });
        });

        // Wait briefly for bind
        thread::sleep(Duration::from_millis(30));
        Ok(Self {
            shutdown,
            handle: Some(handle),
            port,
        })
    }

    pub fn start_shared(
        db: SharedDatabase,
        bind_addr: &str,
        shard_collection: Option<String>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let listener = std::net::TcpListener::bind(bind_addr)?;
        let port = listener.local_addr()?.port();
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_flag = shutdown.clone();

        let handle = thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("tokio runtime");
            rt.block_on(async move {
                let mut state = AppState::new(db, None);
                if let Some(name) = shard_collection {
                    state = state.with_shard_collection(name);
                }
                let app = router(state);
                listener.set_nonblocking(true).ok();
                let listener = tokio::net::TcpListener::from_std(listener).expect("tokio listener");
                let flag = shutdown_flag.clone();
                axum::serve(listener, app)
                    .with_graceful_shutdown(async move {
                        while !flag.load(Ordering::Relaxed) {
                            tokio::time::sleep(Duration::from_millis(20)).await;
                        }
                    })
                    .await
                    .ok();
            });
        });
        thread::sleep(Duration::from_millis(30));
        Ok(Self {
            shutdown,
            handle: Some(handle),
            port,
        })
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    pub fn shutdown(mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        let _ = std::net::TcpStream::connect(format!("127.0.0.1:{}", self.port));
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for BackgroundServer {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }
}
