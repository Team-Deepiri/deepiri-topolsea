//! Deepiri Topolsea HTTP service (Phase A3).

use clap::Parser;
use dv_query::Database;
use dv_server::{router, AppState};
use std::net::SocketAddr;
use std::path::PathBuf;
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
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let args = Args::parse();
    let db = Database::open(&args.data_dir)?.into_shared();
    let state = AppState::new(db, args.api_key);

    let app = router(state);
    let addr: SocketAddr = args.bind.parse()?;
    tracing::info!("listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
