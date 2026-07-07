use dv_shard_remote::{ShardQueryRequest, ShardQueryResponse, QUERY_PATH};
use dv_types::TopolseaError;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Configuration for a shard query HTTP server.
#[derive(Debug, Clone)]
pub struct ShardServerConfig {
    pub data_dir: PathBuf,
    pub collection: String,
    pub bind_addr: String,
}

/// Lightweight HTTP server exposing shard query on a single physical collection.
pub struct ShardQueryServer {
    shutdown: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    port: u16,
}

impl ShardQueryServer {
    pub fn start(config: ShardServerConfig) -> Result<Self, TopolseaError> {
        let listener = TcpListener::bind(&config.bind_addr).map_err(|e| {
            TopolseaError::Io(std::io::Error::new(
                std::io::ErrorKind::AddrInUse,
                e.to_string(),
            ))
        })?;
        listener.set_nonblocking(true).map_err(TopolseaError::Io)?;
        let port = listener.local_addr().map_err(TopolseaError::Io)?.port();
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_flag = shutdown.clone();
        let data_dir = config.data_dir;
        let collection = config.collection;

        let handle = thread::spawn(move || {
            while !shutdown_flag.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let data_dir = data_dir.clone();
                        let collection = collection.clone();
                        thread::spawn(move || {
                            let _ = handle_connection(&data_dir, &collection, stream);
                        });
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
        });

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
        if let Ok(stream) = TcpStream::connect(format!("127.0.0.1:{}", self.port)) {
            let _ = stream.shutdown(Shutdown::Both);
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for ShardQueryServer {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }
}

fn handle_connection(
    data_dir: &std::path::Path,
    collection: &str,
    stream: TcpStream,
) -> Result<(), TopolseaError> {
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .map_err(TopolseaError::Io)?;
    let mut reader = BufReader::new(stream.try_clone().map_err(TopolseaError::Io)?);
    let mut writer = stream;

    let mut request_line = String::new();
    reader
        .read_line(&mut request_line)
        .map_err(TopolseaError::Io)?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("");

    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).map_err(TopolseaError::Io)?;
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        if let Some(val) = trimmed.strip_prefix("Content-Length:") {
            content_length = val.trim().parse().unwrap_or(0);
        }
    }

    if method != "POST" || path != QUERY_PATH {
        write_http_response(&mut writer, 404, r#"{"error":"not found"}"#)?;
        return Ok(());
    }

    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body).map_err(TopolseaError::Io)?;
    let query_req: ShardQueryRequest =
        serde_json::from_slice(&body).map_err(TopolseaError::Serde)?;

    match execute_local_query(data_dir, collection, &query_req) {
        Ok(response) => {
            let json = serde_json::to_string(&response).map_err(TopolseaError::Serde)?;
            write_http_response(&mut writer, 200, &json)?;
        }
        Err(e) => {
            let json = serde_json::json!({ "error": e.to_string() }).to_string();
            write_http_response(&mut writer, 500, &json)?;
        }
    }
    Ok(())
}

fn write_http_response(
    stream: &mut TcpStream,
    status: u16,
    body: &str,
) -> Result<(), TopolseaError> {
    let status_text = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Error",
    };
    let response = format!(
        "HTTP/1.1 {status} {status_text}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(response.as_bytes())
        .map_err(TopolseaError::Io)?;
    stream.flush().map_err(TopolseaError::Io)?;
    Ok(())
}

fn execute_local_query(
    data_dir: &std::path::Path,
    collection: &str,
    req: &ShardQueryRequest,
) -> Result<ShardQueryResponse, TopolseaError> {
    let mut db = crate::Database::open(data_dir)?;
    let col = db.get_collection(collection)?;
    let results = col.query(&req.vector, req.top_k, None, req.ef)?;
    let hits = results
        .into_iter()
        .map(|r| dv_shard_remote::ShardQueryHit {
            id: r.id,
            internal_id: r.internal_id.0,
            distance: r.distance,
            score: r.score,
        })
        .collect();
    Ok(ShardQueryResponse { hits })
}
