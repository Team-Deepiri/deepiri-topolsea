use crate::auth::check_api_key;
use crate::state::AppState;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use dv_metadata::Filter;
use dv_shard_remote::{ShardQueryRequest, ShardQueryResponse, QUERY_PATH};
use dv_types::{CollectionConfig, DistanceMetric, IndexKind};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::str::FromStr;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route(
            "/v1/collections",
            get(list_collections).post(create_collection),
        )
        .route(
            "/v1/collections/:name",
            get(get_collection).delete(delete_collection),
        )
        .route("/v1/collections/:name/upsert", put(upsert))
        .route("/v1/collections/:name/points", put(upsert))
        .route("/v1/collections/:name/search", post(search))
        .route("/v1/collections/:name/hybrid", post(hybrid_search))
        .route("/v1/collections/:name/explain", post(explain))
        .route("/v1/collections/:name/persist", post(persist_collection))
        .route("/v1/persist", post(persist_all))
        .route(QUERY_PATH, post(shard_query))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn health() -> impl IntoResponse {
    Json(json!({"status": "ok"}))
}

#[allow(clippy::result_large_err)]
fn require_auth(headers: &HeaderMap, state: &AppState) -> Result<(), Response> {
    check_api_key(headers, state.api_key.as_deref())
}

fn err(status: StatusCode, msg: impl ToString) -> Response {
    (status, Json(json!({"error": msg.to_string()}))).into_response()
}

async fn list_collections(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, Response> {
    require_auth(&headers, &state)?;
    let db = state.db.read();
    let names = db
        .list_collections()
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(json!({"collections": names})))
}

#[derive(Debug, Deserialize)]
struct CreateCollectionBody {
    name: String,
    dimension: usize,
    #[serde(default = "default_metric")]
    metric: String,
    #[serde(default = "default_index")]
    index: String,
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
    require_auth(&headers, &state)?;
    let metric =
        DistanceMetric::from_str(&body.metric).map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    let index_kind = match body.index.to_lowercase().as_str() {
        "flat" => IndexKind::Flat,
        "zcolumn" => IndexKind::ZColumn,
        "ivf" | "ivfpq" | "pq" => IndexKind::Ivf,
        _ => IndexKind::Hnsw,
    };
    let mut config = CollectionConfig::new(body.name.clone(), body.dimension, metric);
    config.index_kind = index_kind;
    if index_kind == IndexKind::Flat {
        config = config.with_flat_index();
    } else if index_kind == IndexKind::ZColumn {
        config = config.with_zcolumn_index();
    } else if index_kind == IndexKind::Ivf {
        config = config.with_ivf_index();
    }
    state
        .db
        .write()
        .create_collection(config)
        .map_err(|e| err(StatusCode::CONFLICT, e))?;
    Ok((
        StatusCode::CREATED,
        Json(json!({"name": body.name, "dimension": body.dimension, "index": body.index})),
    ))
}

async fn get_collection(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> Result<impl IntoResponse, Response> {
    require_auth(&headers, &state)?;
    let mut db = state.db.write();
    let col = db
        .get_collection(&name)
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
    require_auth(&headers, &state)?;
    state
        .db
        .write()
        .delete_collection(&name)
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
    require_auth(&headers, &state)?;
    if body.ids.len() != body.vectors.len() {
        return Err(err(StatusCode::BAD_REQUEST, "ids/vectors length mismatch"));
    }
    let mut db = state.db.write();
    let col = db
        .get_collection(&name)
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
    require_auth(&headers, &state)?;
    let filter = body
        .filter
        .as_ref()
        .map(Filter::from_json)
        .transpose()
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    let mut db = state.db.write();
    let col = db
        .get_collection(&name)
        .map_err(|e| err(StatusCode::NOT_FOUND, e))?;
    let results = col
        .read()
        .query(&body.vector, body.top_k, filter.as_ref(), body.ef)
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
}

async fn hybrid_search(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
    Json(body): Json<HybridBody>,
) -> Result<impl IntoResponse, Response> {
    require_auth(&headers, &state)?;
    let filter = body
        .filter
        .as_ref()
        .map(Filter::from_json)
        .transpose()
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    let mut db = state.db.write();
    let col = db
        .get_collection(&name)
        .map_err(|e| err(StatusCode::NOT_FOUND, e))?;
    let results = col
        .read()
        .query_hybrid(
            &body.vector,
            &body.text,
            body.top_k,
            filter.as_ref(),
            body.ef,
            body.rrf_k,
        )
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
    Ok(Json(json!({"hits": hits, "fusion": "rrf"})))
}

async fn explain(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
    Json(body): Json<SearchBody>,
) -> Result<impl IntoResponse, Response> {
    require_auth(&headers, &state)?;
    let filter = body
        .filter
        .as_ref()
        .map(Filter::from_json)
        .transpose()
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    let mut db = state.db.write();
    let col = db
        .get_collection(&name)
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
    require_auth(&headers, &state)?;
    let mut db = state.db.write();
    let col = db
        .get_collection(&name)
        .map_err(|e| err(StatusCode::NOT_FOUND, e))?;
    col.write()
        .persist()
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(json!({"persisted": name})))
}

async fn persist_all(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, Response> {
    require_auth(&headers, &state)?;
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
    let mut db = state.db.write();
    let col = db
        .get_collection(name)
        .map_err(|e| err(StatusCode::NOT_FOUND, e))?;
    let results = col
        .read()
        .query(&req.vector, req.top_k, None, req.ef)
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    let hits = results
        .into_iter()
        .map(|r| dv_shard_remote::ShardQueryHit {
            id: r.id,
            internal_id: r.internal_id.0,
            distance: r.distance,
            score: r.score,
        })
        .collect();
    Ok(Json(ShardQueryResponse { hits }))
}
