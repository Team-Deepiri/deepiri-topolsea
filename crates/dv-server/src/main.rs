//! Deepiri Topolsea HTTP service (Phase A3).

use clap::Parser;
use dv_query::Database;
use dv_server::{router, AppState};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(
    name = "topolsea-server",
    about = "Topolsea REST vector database server"
)]
struct Args {
    /// Data directory for collections
    #[arg(long, default_value = "./data")]
    data_dir: PathBuf,

    /// Bind address
    #[arg(long, default_value = "127.0.0.1:6333")]
    bind: String,

    /// Optional API key (sent as `x-api-key` or `Authorization: Bearer …`)
    #[arg(long, env = "TOPOLSEA_API_KEY")]
    api_key: Option<String>,

    /// JSON map of tenant API key → namespace, e.g. '{"k1":"acme"}' (C13)
    #[arg(long, env = "TOPOLSEA_TENANT_KEYS")]
    tenant_keys: Option<String>,

    /// Background snapshot interval in seconds (0 disables auto-flush)
    #[arg(long, default_value = "30")]
    flush_secs: u64,

    /// Optional physical collection for `/topolsea/v1/shard/query` (shard fan-out)
    #[arg(long)]
    shard_collection: Option<String>,

    /// TLS certificate PEM path (requires --tls-key)
    #[arg(long)]
    tls_cert: Option<PathBuf>,

    /// TLS private key PEM path (requires --tls-cert)
    #[arg(long)]
    tls_key: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let args = Args::parse();
    if args.tls_cert.is_some() != args.tls_key.is_some() {
        return Err("both --tls-cert and --tls-key are required for TLS".into());
    }

    let db = Database::open(&args.data_dir)?.into_shared();
    let mut state = AppState::new(db.clone(), args.api_key);
    if let Some(raw) = args.tenant_keys {
        let map: std::collections::HashMap<String, String> = serde_json::from_str(&raw)?;
        state = state.with_tenant_keys(map);
    }
    if let Some(name) = args.shard_collection {
        state = state.with_shard_collection(name);
    }

    if args.flush_secs > 0 {
        let flush_db = db.clone();
        let flush_metrics = state.metrics.clone();
        let period = Duration::from_secs(args.flush_secs);
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(period);
            loop {
                ticker.tick().await;
                let lag = flush_db.read().sample_wal_lag().unwrap_or(0);
                flush_metrics.set_wal_lag(lag);
                if let Err(e) = flush_db.write().persist_all() {
                    tracing::warn!("auto-flush failed: {e}");
                } else {
                    flush_metrics.set_wal_lag(0);
                    tracing::debug!("auto-flush snapshot complete");
                }
            }
        });
        tracing::info!("auto-flush every {}s", args.flush_secs);
    }

    let app = router(state);
    let addr: SocketAddr = args.bind.parse()?;

    if let (Some(cert), Some(key)) = (args.tls_cert, args.tls_key) {
        let config = axum_server::tls_rustls::RustlsConfig::from_pem_file(cert, key).await?;
        tracing::info!("listening on https://{addr}");
        axum_server::bind_rustls(addr, config)
            .serve(app.into_make_service())
            .await?;
    } else {
        tracing::info!("listening on http://{addr}");
        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;
    }
    Ok(())
}
