use crate::auth::authorize;
use crate::state::AppState;
use axum::body::Body;
use axum::extract::{Path, Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use dv_metadata::Filter;
use dv_observe::trace::{new_request_id, traceparent};
use dv_query::{qualify_collection, strip_namespace, FusionMethod, HybridOptions};
use dv_shard_remote::{
    ReplicateDeleteRequest, ReplicateDeleteResponse, ReplicateUpsertRequest,
    ReplicateUpsertResponse, ShardHealthResponse, ShardQueryRequest, ShardQueryResponse,
    QUERY_PATH, REPLICATE_DELETE_PATH, REPLICATE_UPSERT_PATH, SHARD_HEALTH_PATH,
};
use dv_storage::{ClusterMembership, ClusterNode};
use dv_types::{CollectionConfig, DistanceMetric, IndexKind, IvfConfig};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::str::FromStr;
use std::sync::atomic::Ordering;
use std::time::Instant;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/livez", get(livez))
        .route("/readyz", get(readyz))
        .route(
            "/v1/collections",
            get(list_collections).post(create_collection),
        )
        .route(
            "/v1/collections/:name",
            get(get_collection).delete(delete_collection),
        )
        .route("/v1/collections/:name/upsert", put(upsert))
        .route(
            "/v1/collections/:name/points",
            put(upsert).delete(delete_points),
        )
        .route("/v1/collections/:name/search", post(search))
        .route("/v1/collections/:name/hybrid", post(hybrid_search))
        .route("/v1/collections/:name/sparse", post(sparse_search))
        .route("/v1/collections/:name/explain", post(explain))
        .route("/v1/collections/:name/persist", post(persist_collection))
        .route("/v1/collections/:name/compact", post(compact_collection))
        .route(
            "/v1/ns/:ns/collections",
            get(list_collections_ns).post(create_collection_ns),
        )
        .route(
            "/v1/ns/:ns/collections/:name",
            get(get_collection_ns).delete(delete_collection_ns),
        )
        .route("/v1/ns/:ns/collections/:name/upsert", put(upsert_ns))
        .route("/v1/ns/:ns/collections/:name/search", post(search_ns))
        .route("/v1/persist", post(persist_all))
        .route("/metrics", get(metrics_endpoint))
        .route(
            "/v1/cluster/membership",
            get(get_membership).put(put_membership),
        )
        .route("/v1/cluster/heartbeat", post(cluster_heartbeat))
        .route("/v1/snapshots", get(list_snapshots).post(create_snapshot))
        .route(
            "/v1/snapshots/:name",
            get(get_snapshot_meta).delete(delete_snapshot),
        )
        .route("/v1/snapshots/:name/restore", post(restore_snapshot))
        .route("/v1/shards/:logical/replicas", post(add_replica))
        .route(
            "/v1/shards/:logical/replica-policy",
            put(set_replica_policy),
        )
        .route(QUERY_PATH, post(shard_query))
        .route(REPLICATE_UPSERT_PATH, post(replicate_upsert))
        .route(REPLICATE_DELETE_PATH, post(replicate_delete))
        .route(SHARD_HEALTH_PATH, get(shard_health))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            observe_middleware,
        ))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn observe_middleware(
    State(state): State<AppState>,
    mut req: Request<Body>,
    next: Next,
) -> Response {
    let started = Instant::now();
    let path = req.uri().path().to_string();
    let request_id = req
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(new_request_id);
    tracing::info!(request_id = %request_id, path = %path, "http_request");
    req.headers_mut().insert(
        "x-request-id",
        HeaderValue::from_str(&request_id).unwrap_or_else(|_| HeaderValue::from_static("unknown")),
    );
    let mut response = next.run(req).await;
    let status = response.status().as_u16();
    state
        .metrics
        .record_http(&route_label(&path), status, started);
    response.headers_mut().insert(
        "x-request-id",
        HeaderValue::from_str(&request_id).unwrap_or_else(|_| HeaderValue::from_static("unknown")),
    );
    response.headers_mut().insert(
        "traceparent",
        HeaderValue::from_str(&traceparent(&request_id))
            .unwrap_or_else(|_| HeaderValue::from_static("00-0-0-01")),
    );
    response
}

fn route_label(path: &str) -> String {
    if path.starts_with("/v1/collections/") && path.ends_with("/search") {
        "search".into()
    } else if path.starts_with("/v1/collections/") && path.contains("/upsert") {
        "upsert".into()
    } else if path == "/metrics" {
        "metrics".into()
    } else if path.starts_with("/topolsea/v1/shard") {
        "shard".into()
    } else if path.starts_with("/v1/snapshots") {
        "snapshot".into()
    } else {
        "http".into()
    }
}

async fn health() -> impl IntoResponse {
    Json(json!({"status": "ok"}))
}

async fn livez() -> impl IntoResponse {
    Json(json!({"status": "alive"}))
}

async fn readyz(State(state): State<AppState>) -> impl IntoResponse {
    // Ready when storage root is reachable and membership (if any) is readable.
    match state.db.read().membership() {
        Ok(_) => (StatusCode::OK, Json(json!({"status": "ready"}))).into_response(),
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status": "not_ready", "error": e.to_string()})),
        )
            .into_response(),
    }
}

#[allow(clippy::result_large_err)]
fn require_auth(headers: &HeaderMap, state: &AppState) -> Result<String, Response> {
    authorize(headers, state.api_key.as_deref(), &state.tenant_keys)
}

fn qname(ns: &str, name: &str) -> String {
    qualify_collection(ns, name)
}

fn err(status: StatusCode, msg: impl ToString) -> Response {
    (status, Json(json!({"error": msg.to_string()}))).into_response()
}

async fn list_collections(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, Response> {
    let ns = require_auth(&headers, &state)?;
    let db = state.db.read();
    let names = db
        .list_collections()
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let names: Vec<_> = names
        .into_iter()
        .filter_map(|n| strip_namespace(&ns, &n).map(|s| s.to_string()))
        .collect();
    Ok(Json(json!({"collections": names, "namespace": ns})))
}

#[derive(Debug, Deserialize)]
struct CreateCollectionBody {
    name: String,
    dimension: usize,
    #[serde(default = "default_metric")]
    metric: String,
    #[serde(default = "default_index")]
    index: String,
    #[serde(default)]
    ivf: Option<IvfCreateConfig>,
}

#[derive(Debug, Deserialize)]
struct IvfCreateConfig {
    #[serde(default)]
    nlist: Option<usize>,
    #[serde(default)]
    nprobe: Option<usize>,
    #[serde(default)]
    pq_m: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    memory_bound: Option<bool>,
}

fn default_metric() -> String {
    "cosine".into()
}
fn default_index() -> String {
    "hnsw".into()
}

async fn create_collection(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<CreateCollectionBody>,
) -> Result<impl IntoResponse, Response> {
    let ns = require_auth(&headers, &state)?;
    let metric =
        DistanceMetric::from_str(&body.metric).map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    let index_kind = match body.index.to_lowercase().as_str() {
        "flat" => IndexKind::Flat,
        "zcolumn" => IndexKind::ZColumn,
        "ivf" | "ivfpq" | "pq" => IndexKind::Ivf,
        _ => IndexKind::Hnsw,
    };
    let qualified = qname(&ns, &body.name);
    let mut config = CollectionConfig::new(qualified.clone(), body.dimension, metric);
    config.index_kind = index_kind;
    if index_kind == IndexKind::Flat {
        config = config.with_flat_index();
    } else if index_kind == IndexKind::ZColumn {
        config = config.with_zcolumn_index();
    } else if index_kind == IndexKind::Ivf {
        config = config.with_ivf_index();
        let mut ivf = IvfConfig::default();
        if let Some(c) = &body.ivf {
            if let Some(n) = c.nlist {
                ivf.nlist = n;
            }
            if let Some(n) = c.nprobe {
                ivf.nprobe = n;
            }
            if let Some(m) = c.pq_m {
                ivf.pq_m = Some(m);
                ivf.memory_bound = c.memory_bound.unwrap_or(true);
            }
            if let Some(s) = c.seed {
                ivf.seed = s;
            }
            if let Some(mb) = c.memory_bound {
                ivf.memory_bound = mb;
            }
        }
        config.ivf = ivf;
    }
    state
        .db
        .write()
        .create_collection(config)
        .map_err(|e| err(StatusCode::CONFLICT, e))?;
    Ok((
        StatusCode::CREATED,
        Json(
            json!({"name": body.name, "namespace": ns, "dimension": body.dimension, "index": body.index}),
        ),
    ))
}

async fn get_collection(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> Result<impl IntoResponse, Response> {
    let ns = require_auth(&headers, &state)?;
    let mut db = state.db.write();
    let col = db
        .get_collection(&qname(&ns, &name))
        .map_err(|e| err(StatusCode::NOT_FOUND, e))?;
    let col = col.read();
    Ok(Json(json!({
        "name": col.name(),
        "dimension": col.config().dimension,
        "metric": col.config().metric.to_string(),
        "index": format!("{:?}", col.config().index_kind),
        "vectors": col.len(),
    })))
}

async fn delete_collection(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> Result<impl IntoResponse, Response> {
    let ns = require_auth(&headers, &state)?;
    state
        .db
        .write()
        .delete_collection(&qname(&ns, &name))
        .map_err(|e| err(StatusCode::NOT_FOUND, e))?;
    Ok(Json(json!({"deleted": name})))
}

#[derive(Debug, Deserialize)]
struct UpsertBody {
    ids: Vec<String>,
    vectors: Vec<Vec<f32>>,
    #[serde(default)]
    metadatas: Option<Vec<Option<Value>>>,
    /// Optional per-point document texts for BM25 / hybrid (B6).
    #[serde(default)]
    texts: Option<Vec<Option<String>>>,
}

async fn upsert(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
    Json(body): Json<UpsertBody>,
) -> Result<impl IntoResponse, Response> {
    let ns = require_auth(&headers, &state)?;
    if body.ids.len() != body.vectors.len() {
        return Err(err(StatusCode::BAD_REQUEST, "ids/vectors length mismatch"));
    }
    let mut db = state.db.write();
    let col = db
        .get_collection(&qname(&ns, &name))
        .map_err(|e| err(StatusCode::NOT_FOUND, e))?;
    let mut guard = col.write();
    let metas = body.metadatas.unwrap_or_default();
    let texts = body.texts.unwrap_or_default();
    for (i, (id, vec)) in body.ids.iter().zip(body.vectors).enumerate() {
        let meta = metas.get(i).and_then(|m| m.clone());
        let text = texts.get(i).and_then(|t| t.as_deref());
        guard
            .upsert_with_text(id, vec, meta, text)
            .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    }
    state
        .metrics
        .upsert_total
        .fetch_add(body.ids.len() as u64, Ordering::Relaxed);
    Ok(Json(json!({"upserted": body.ids.len()})))
}

#[derive(Debug, Deserialize)]
struct SearchBody {
    vector: Vec<f32>,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default)]
    filter: Option<Value>,
    #[serde(default = "default_ef")]
    ef: usize,
    /// IVF nprobe override (also accepted via `ef` for IndexKind::Ivf).
    #[serde(default)]
    nprobe: Option<usize>,
}

fn default_top_k() -> usize {
    10
}
fn default_ef() -> usize {
    64
}

#[derive(Debug, Serialize)]
struct HitOut {
    id: Option<String>,
    distance: f32,
    score: f32,
    metadata: Option<Value>,
}

async fn search(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
    Json(body): Json<SearchBody>,
) -> Result<impl IntoResponse, Response> {
    let ns = require_auth(&headers, &state)?;
    let filter = body
        .filter
        .as_ref()
        .map(Filter::from_json)
        .transpose()
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    let mut db = state.db.write();
    let col = db
        .get_collection(&qname(&ns, &name))
        .map_err(|e| err(StatusCode::NOT_FOUND, e))?;
    let ef = body.nprobe.unwrap_or(body.ef);
    let results = col
        .read()
        .query(&body.vector, body.top_k, filter.as_ref(), ef)
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    let hits: Vec<HitOut> = results
        .into_iter()
        .map(|r| HitOut {
            id: r.id,
            distance: r.distance,
            score: r.score,
            metadata: r.metadata,
        })
        .collect();
    state.metrics.search_total.fetch_add(1, Ordering::Relaxed);
    Ok(Json(json!({"hits": hits})))
}

#[derive(Debug, Deserialize)]
struct HybridBody {
    vector: Vec<f32>,
    text: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default)]
    filter: Option<Value>,
    #[serde(default = "default_ef")]
    ef: usize,
    #[serde(default)]
    rrf_k: Option<f32>,
    #[serde(default)]
    fusion: Option<String>,
    #[serde(default)]
    dense_weight: Option<f32>,
    #[serde(default)]
    prefetch: Option<usize>,
}

async fn hybrid_search(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
    Json(body): Json<HybridBody>,
) -> Result<impl IntoResponse, Response> {
    let ns = require_auth(&headers, &state)?;
    let filter = body
        .filter
        .as_ref()
        .map(Filter::from_json)
        .transpose()
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    let mut opts = HybridOptions::new(body.top_k, body.ef);
    opts.rrf_k = body.rrf_k;
    opts.dense_weight = body.dense_weight;
    opts.prefetch = body.prefetch;
    opts.fusion = match body
        .fusion
        .as_deref()
        .unwrap_or("rrf")
        .to_lowercase()
        .as_str()
    {
        "linear" | "weighted" => FusionMethod::Linear,
        _ => FusionMethod::Rrf,
    };
    let mut db = state.db.write();
    let col = db
        .get_collection(&qname(&ns, &name))
        .map_err(|e| err(StatusCode::NOT_FOUND, e))?;
    let results = col
        .read()
        .query_hybrid_opts(&body.vector, &body.text, filter.as_ref(), &opts)
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    let hits: Vec<HitOut> = results
        .into_iter()
        .map(|r| HitOut {
            id: r.id,
            distance: r.distance,
            score: r.score,
            metadata: r.metadata,
        })
        .collect();
    Ok(Json(json!({
        "hits": hits,
        "fusion": format!("{:?}", opts.fusion).to_ascii_lowercase(),
    })))
}

#[derive(Debug, Deserialize)]
struct SparseBody {
    text: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default)]
    filter: Option<Value>,
}

async fn sparse_search(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
    Json(body): Json<SparseBody>,
) -> Result<impl IntoResponse, Response> {
    let ns = require_auth(&headers, &state)?;
    let filter = body
        .filter
        .as_ref()
        .map(Filter::from_json)
        .transpose()
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    let mut db = state.db.write();
    let col = db
        .get_collection(&qname(&ns, &name))
        .map_err(|e| err(StatusCode::NOT_FOUND, e))?;
    let results = col
        .read()
        .query_sparse(&body.text, body.top_k, filter.as_ref())
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    let hits: Vec<HitOut> = results
        .into_iter()
        .map(|r| HitOut {
            id: r.id,
            distance: r.distance,
            score: r.score,
            metadata: r.metadata,
        })
        .collect();
    Ok(Json(json!({"hits": hits})))
}

async fn explain(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
    Json(body): Json<SearchBody>,
) -> Result<impl IntoResponse, Response> {
    let ns = require_auth(&headers, &state)?;
    let filter = body
        .filter
        .as_ref()
        .map(Filter::from_json)
        .transpose()
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    let mut db = state.db.write();
    let col = db
        .get_collection(&qname(&ns, &name))
        .map_err(|e| err(StatusCode::NOT_FOUND, e))?;
    let (results, explain) = col
        .read()
        .query_explain(&body.vector, body.top_k, filter.as_ref(), body.ef)
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    let hits: Vec<HitOut> = results
        .into_iter()
        .map(|r| HitOut {
            id: r.id,
            distance: r.distance,
            score: r.score,
            metadata: r.metadata,
        })
        .collect();
    Ok(Json(json!({
        "hits": hits,
        "explain": explain,
    })))
}

async fn persist_collection(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> Result<impl IntoResponse, Response> {
    let ns = require_auth(&headers, &state)?;
    let mut db = state.db.write();
    let col = db
        .get_collection(&qname(&ns, &name))
        .map_err(|e| err(StatusCode::NOT_FOUND, e))?;
    col.write()
        .persist()
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(json!({"persisted": name})))
}

async fn compact_collection(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> Result<impl IntoResponse, Response> {
    let ns = require_auth(&headers, &state)?;
    let mut db = state.db.write();
    let col = db
        .get_collection(&qname(&ns, &name))
        .map_err(|e| err(StatusCode::NOT_FOUND, e))?;
    col.write()
        .compact_segments()
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let stats = col
        .read()
        .segment_stats()
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(json!({"compacted": name, "segments": stats})))
}

async fn persist_all(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, Response> {
    let _ns = require_auth(&headers, &state)?;
    state
        .db
        .write()
        .persist_all()
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(json!({"persisted": "all"})))
}

/// Compatibility endpoint for fractal shard fan-out clients.
async fn shard_query(
    State(state): State<AppState>,
    Json(req): Json<ShardQueryRequest>,
) -> Result<impl IntoResponse, Response> {
    let name = state.shard_collection.as_ref().ok_or_else(|| {
        err(
            StatusCode::BAD_REQUEST,
            "shard query requires server --shard-collection",
        )
    })?;
    let filter = req
        .filter
        .as_ref()
        .map(Filter::from_json)
        .transpose()
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    state
        .metrics
        .shard_fanout_total
        .fetch_add(1, Ordering::Relaxed);
    let mut db = state.db.write();
    let col = db
        .get_collection(name)
        .map_err(|e| err(StatusCode::NOT_FOUND, e))?;
    let results = col
        .read()
        .query(&req.vector, req.top_k, filter.as_ref(), req.ef)
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    let hits = results
        .into_iter()
        .map(|r| dv_shard_remote::ShardQueryHit {
            id: r.id,
            internal_id: r.internal_id.0,
            distance: r.distance,
            score: r.score,
            metadata: r.metadata,
        })
        .collect();
    Ok(Json(ShardQueryResponse { hits }))
}

async fn metrics_endpoint(State(state): State<AppState>) -> impl IntoResponse {
    // Refresh WAL lag sample on scrape.
    if let Ok(lag) = state.db.read().sample_wal_lag() {
        state.metrics.set_wal_lag(lag);
    }
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        state.metrics.render_prometheus(),
    )
}

async fn get_membership(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, Response> {
    let _ns = require_auth(&headers, &state)?;
    let m = state
        .db
        .read()
        .membership()
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(m))
}

#[derive(Debug, serde::Deserialize)]
struct MembershipBody {
    nodes: Vec<ClusterNode>,
}

async fn put_membership(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<MembershipBody>,
) -> Result<impl IntoResponse, Response> {
    let _ns = require_auth(&headers, &state)?;
    let mut membership = ClusterMembership {
        nodes: body.nodes,
        generation: 0,
    };
    {
        let cur = state.db.read().membership().unwrap_or_default();
        membership.generation = cur.generation.saturating_add(1);
    }
    state
        .db
        .write()
        .set_membership(membership.clone())
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    Ok(Json(membership))
}

#[derive(Debug, serde::Deserialize)]
struct HeartbeatBody {
    id: String,
    #[serde(default = "default_true")]
    healthy: bool,
}

fn default_true() -> bool {
    true
}

async fn cluster_heartbeat(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<HeartbeatBody>,
) -> Result<impl IntoResponse, Response> {
    let _ns = require_auth(&headers, &state)?;
    state
        .db
        .write()
        .heartbeat_node(&body.id, body.healthy)
        .map_err(|e| err(StatusCode::NOT_FOUND, e))?;
    Ok(Json(json!({"id": body.id, "healthy": body.healthy})))
}

#[derive(Debug, serde::Deserialize)]
struct SnapshotBody {
    name: String,
    #[serde(default)]
    collections: Option<Vec<String>>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct RestoreBody {
    #[serde(default)]
    replace_all: bool,
}

async fn create_snapshot(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<SnapshotBody>,
) -> Result<impl IntoResponse, Response> {
    let _ns = require_auth(&headers, &state)?;
    let path = state
        .db
        .write()
        .create_snapshot_opts(&body.name, body.collections.as_deref())
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    state.metrics.snapshot_total.fetch_add(1, Ordering::Relaxed);
    Ok(Json(json!({"snapshot": body.name, "path": path})))
}

async fn list_snapshots(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, Response> {
    let _ns = require_auth(&headers, &state)?;
    let names = state
        .db
        .read()
        .list_snapshots()
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(json!({"snapshots": names})))
}

async fn get_snapshot_meta(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> Result<impl IntoResponse, Response> {
    let _ns = require_auth(&headers, &state)?;
    let meta = state
        .db
        .read()
        .snapshot_meta(&name)
        .map_err(|e| err(StatusCode::NOT_FOUND, e))?;
    Ok(Json(meta))
}

async fn restore_snapshot(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
    Json(body): Json<RestoreBody>,
) -> Result<impl IntoResponse, Response> {
    let _ns = require_auth(&headers, &state)?;
    state
        .db
        .write()
        .restore_snapshot_opts(&name, body.replace_all)
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    Ok(Json(
        json!({"restored": name, "replace_all": body.replace_all}),
    ))
}

async fn delete_snapshot(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> Result<impl IntoResponse, Response> {
    let _ns = require_auth(&headers, &state)?;
    state
        .db
        .write()
        .delete_snapshot(&name)
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    Ok(Json(json!({"deleted": name})))
}

#[derive(Debug, serde::Deserialize)]
struct ReplicaBody {
    shard_id: usize,
    url: String,
}

async fn add_replica(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(logical): Path<String>,
    Json(body): Json<ReplicaBody>,
) -> Result<impl IntoResponse, Response> {
    let ns = require_auth(&headers, &state)?;
    let logical = qname(&ns, &logical);
    state
        .db
        .write()
        .add_shard_replica(&logical, body.shard_id, body.url.clone())
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    Ok(Json(
        json!({"logical": logical, "shard_id": body.shard_id, "replica": body.url}),
    ))
}

#[derive(Debug, serde::Deserialize)]
struct ReplicaPolicyBody {
    require_replica_ack: bool,
    #[serde(default)]
    replica_timeout_ms: Option<u64>,
}

async fn set_replica_policy(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(logical): Path<String>,
    Json(body): Json<ReplicaPolicyBody>,
) -> Result<impl IntoResponse, Response> {
    let ns = require_auth(&headers, &state)?;
    let logical = qname(&ns, &logical);
    state
        .db
        .write()
        .set_replica_policy(&logical, body.require_replica_ack, body.replica_timeout_ms)
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    Ok(Json(json!({
        "logical": logical,
        "require_replica_ack": body.require_replica_ack,
        "replica_timeout_ms": body.replica_timeout_ms,
    })))
}

async fn replicate_upsert(
    State(state): State<AppState>,
    Json(body): Json<ReplicateUpsertRequest>,
) -> Result<impl IntoResponse, Response> {
    if body.ids.len() != body.vectors.len() {
        return Err(err(StatusCode::BAD_REQUEST, "ids/vectors length mismatch"));
    }
    let applied = body.ids.len();
    let mut db = state.db.write();
    let col = db
        .get_collection(&body.collection)
        .map_err(|e| err(StatusCode::NOT_FOUND, e))?;
    let mut guard = col.write();
    let metas = body.metadatas.clone().unwrap_or_default();
    let texts = body.texts.clone().unwrap_or_default();
    for (i, (id, vec)) in body.ids.iter().zip(body.vectors.iter()).enumerate() {
        let meta = metas.get(i).and_then(|m| m.clone());
        let text = texts.get(i).and_then(|t| t.as_deref());
        if let Err(e) = guard.upsert_with_text(id, vec.clone(), meta, text) {
            state
                .metrics
                .replicate_fail_total
                .fetch_add(1, Ordering::Relaxed);
            return Err(err(StatusCode::BAD_REQUEST, e));
        }
    }
    state
        .metrics
        .replicate_total
        .fetch_add(1, Ordering::Relaxed);
    Ok(Json(ReplicateUpsertResponse { applied }))
}

async fn replicate_delete(
    State(state): State<AppState>,
    Json(body): Json<ReplicateDeleteRequest>,
) -> Result<impl IntoResponse, Response> {
    let mut db = state.db.write();
    let col = db
        .get_collection(&body.collection)
        .map_err(|e| err(StatusCode::NOT_FOUND, e))?;
    let mut guard = col.write();
    let mut deleted = 0usize;
    for id in &body.ids {
        match guard.delete(id) {
            Ok(()) => deleted += 1,
            Err(e) => {
                state
                    .metrics
                    .replicate_fail_total
                    .fetch_add(1, Ordering::Relaxed);
                return Err(err(StatusCode::BAD_REQUEST, e));
            }
        }
    }
    state
        .metrics
        .replicate_total
        .fetch_add(1, Ordering::Relaxed);
    Ok(Json(ReplicateDeleteResponse { deleted }))
}

async fn shard_health(State(state): State<AppState>) -> impl IntoResponse {
    match &state.shard_collection {
        Some(name) => {
            let mut db = state.db.write();
            match db.get_collection(name) {
                Ok(col) => {
                    let vectors = col.read().len();
                    Json(ShardHealthResponse {
                        status: "ready".into(),
                        collection: Some(name.to_string()),
                        vectors: Some(vectors),
                    })
                    .into_response()
                }
                Err(e) => (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(ShardHealthResponse {
                        status: format!("not_ready: {e}"),
                        collection: Some(name.to_string()),
                        vectors: None,
                    }),
                )
                    .into_response(),
            }
        }
        None => Json(ShardHealthResponse {
            status: "ready".into(),
            collection: None,
            vectors: None,
        })
        .into_response(),
    }
}

#[derive(Debug, Deserialize)]
struct DeletePointsBody {
    ids: Vec<String>,
}

async fn delete_points(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
    Json(body): Json<DeletePointsBody>,
) -> Result<impl IntoResponse, Response> {
    let ns = require_auth(&headers, &state)?;
    let qualified = qname(&ns, &name);
    let mut db = state.db.write();
    let deleted = if db.is_sharded(&qualified) {
        for id in &body.ids {
            db.delete_sharded(&qualified, id)
                .map_err(|e| err(StatusCode::NOT_FOUND, e))?;
        }
        body.ids.len()
    } else {
        let col = db
            .get_collection(&qualified)
            .map_err(|e| err(StatusCode::NOT_FOUND, e))?;
        let mut guard = col.write();
        for id in &body.ids {
            guard
                .delete(id)
                .map_err(|e| err(StatusCode::NOT_FOUND, e))?;
        }
        body.ids.len()
    };
    state
        .metrics
        .delete_total
        .fetch_add(deleted as u64, Ordering::Relaxed);
    Ok(Json(json!({"deleted": deleted})))
}

/// Ensure path namespace matches the authorized namespace (tenant keys are sticky).
#[allow(clippy::result_large_err)]
fn require_ns(headers: &HeaderMap, state: &AppState, path_ns: &str) -> Result<String, Response> {
    let auth_ns = require_auth(headers, state)?;
    let path_ns = dv_query::normalize_namespace(path_ns);
    if auth_ns != path_ns && auth_ns != dv_query::DEFAULT_NAMESPACE {
        // Tenant-scoped keys cannot escape their namespace.
        if !state.tenant_keys.is_empty()
            && state.tenant_keys.values().any(|v| v == &auth_ns)
            && state.api_key.as_deref().is_none()
        {
            return Err(err(
                StatusCode::FORBIDDEN,
                format!("namespace mismatch: auth={auth_ns} path={path_ns}"),
            ));
        }
        if state.tenant_keys.values().any(|v| v == &auth_ns) && auth_ns != path_ns {
            return Err(err(
                StatusCode::FORBIDDEN,
                format!("namespace mismatch: auth={auth_ns} path={path_ns}"),
            ));
        }
    }
    // Prefer explicit path namespace for global-key / open auth.
    Ok(path_ns)
}

async fn list_collections_ns(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(ns): Path<String>,
) -> Result<impl IntoResponse, Response> {
    let ns = require_ns(&headers, &state, &ns)?;
    let db = state.db.read();
    let names = db
        .list_collections()
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let names: Vec<_> = names
        .into_iter()
        .filter_map(|n| strip_namespace(&ns, &n).map(|s| s.to_string()))
        .collect();
    Ok(Json(json!({"collections": names, "namespace": ns})))
}

async fn create_collection_ns(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(ns): Path<String>,
    Json(mut body): Json<CreateCollectionBody>,
) -> Result<impl IntoResponse, Response> {
    let ns = require_ns(&headers, &state, &ns)?;
    // Force create into path namespace by rewriting header-style auth path.
    let mut headers = headers.clone();
    headers.insert(
        "x-namespace",
        HeaderValue::from_str(&ns).unwrap_or_else(|_| HeaderValue::from_static("default")),
    );
    // Reuse create by temporarily using path ns via qname in a local copy of create logic:
    let metric =
        DistanceMetric::from_str(&body.metric).map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    let index_kind = match body.index.to_lowercase().as_str() {
        "flat" => IndexKind::Flat,
        "zcolumn" => IndexKind::ZColumn,
        "ivf" | "ivfpq" | "pq" => IndexKind::Ivf,
        _ => IndexKind::Hnsw,
    };
    let qualified = qname(&ns, &body.name);
    let mut config = CollectionConfig::new(qualified.clone(), body.dimension, metric);
    config.index_kind = index_kind;
    if index_kind == IndexKind::Flat {
        config = config.with_flat_index();
    } else if index_kind == IndexKind::ZColumn {
        config = config.with_zcolumn_index();
    } else if index_kind == IndexKind::Ivf {
        config = config.with_ivf_index();
    }
    let _ = &mut body;
    state
        .db
        .write()
        .create_collection(config)
        .map_err(|e| err(StatusCode::CONFLICT, e))?;
    Ok((
        StatusCode::CREATED,
        Json(json!({"name": body.name, "namespace": ns, "dimension": body.dimension})),
    ))
}

async fn get_collection_ns(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((ns, name)): Path<(String, String)>,
) -> Result<impl IntoResponse, Response> {
    let ns = require_ns(&headers, &state, &ns)?;
    let mut db = state.db.write();
    let col = db
        .get_collection(&qname(&ns, &name))
        .map_err(|e| err(StatusCode::NOT_FOUND, e))?;
    let col = col.read();
    Ok(Json(json!({
        "name": name,
        "namespace": ns,
        "dimension": col.config().dimension,
        "metric": col.config().metric.to_string(),
        "index": format!("{:?}", col.config().index_kind),
        "vectors": col.len(),
    })))
}

async fn delete_collection_ns(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((ns, name)): Path<(String, String)>,
) -> Result<impl IntoResponse, Response> {
    let ns = require_ns(&headers, &state, &ns)?;
    state
        .db
        .write()
        .delete_collection(&qname(&ns, &name))
        .map_err(|e| err(StatusCode::NOT_FOUND, e))?;
    Ok(Json(json!({"deleted": name, "namespace": ns})))
}

async fn upsert_ns(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((ns, name)): Path<(String, String)>,
    Json(body): Json<UpsertBody>,
) -> Result<impl IntoResponse, Response> {
    let ns = require_ns(&headers, &state, &ns)?;
    if body.ids.len() != body.vectors.len() {
        return Err(err(StatusCode::BAD_REQUEST, "ids/vectors length mismatch"));
    }
    let mut db = state.db.write();
    let col = db
        .get_collection(&qname(&ns, &name))
        .map_err(|e| err(StatusCode::NOT_FOUND, e))?;
    let mut guard = col.write();
    let metas = body.metadatas.unwrap_or_default();
    let texts = body.texts.unwrap_or_default();
    for (i, (id, vec)) in body.ids.iter().zip(body.vectors).enumerate() {
        let meta = metas.get(i).and_then(|m| m.clone());
        let text = texts.get(i).and_then(|t| t.as_deref());
        guard
            .upsert_with_text(id, vec, meta, text)
            .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    }
    state
        .metrics
        .upsert_total
        .fetch_add(body.ids.len() as u64, Ordering::Relaxed);
    Ok(Json(json!({"upserted": body.ids.len(), "namespace": ns})))
}

async fn search_ns(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((ns, name)): Path<(String, String)>,
    Json(body): Json<SearchBody>,
) -> Result<impl IntoResponse, Response> {
    let ns = require_ns(&headers, &state, &ns)?;
    let filter = body
        .filter
        .as_ref()
        .map(Filter::from_json)
        .transpose()
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    let mut db = state.db.write();
    let col = db
        .get_collection(&qname(&ns, &name))
        .map_err(|e| err(StatusCode::NOT_FOUND, e))?;
    let ef = body.nprobe.unwrap_or(body.ef);
    let results = col
        .read()
        .query(&body.vector, body.top_k, filter.as_ref(), ef)
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    let hits: Vec<HitOut> = results
        .into_iter()
        .map(|r| HitOut {
            id: r.id,
            distance: r.distance,
            score: r.score,
            metadata: r.metadata,
        })
        .collect();
    state.metrics.search_total.fetch_add(1, Ordering::Relaxed);
    Ok(Json(json!({"hits": hits, "namespace": ns})))
}
