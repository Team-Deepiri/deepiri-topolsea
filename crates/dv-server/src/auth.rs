//! Auth helpers: global API key + per-tenant keys (C13).

use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use std::collections::HashMap;

pub fn extract_api_key(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .or_else(|| {
            headers
                .get(axum::http::header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.strip_prefix("Bearer ").map(|t| t.to_string()))
        })
}

#[allow(clippy::result_large_err)]
pub fn check_api_key(headers: &HeaderMap, expected: Option<&str>) -> Result<(), Response> {
    let Some(expected) = expected else {
        return Ok(());
    };
    match extract_api_key(headers) {
        Some(key) if key == expected => Ok(()),
        _ => Err((
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": "invalid or missing API key"})),
        )
            .into_response()),
    }
}

/// Validate credentials and resolve the request namespace.
#[allow(clippy::result_large_err)]
pub fn authorize(
    headers: &HeaderMap,
    global_key: Option<&str>,
    tenant_keys: &HashMap<String, String>,
) -> Result<String, Response> {
    let needs_auth = global_key.is_some() || !tenant_keys.is_empty();
    if !needs_auth {
        if let Some(ns) = headers.get("x-namespace").and_then(|v| v.to_str().ok()) {
            return Ok(dv_query::normalize_namespace(ns));
        }
        return Ok(dv_query::DEFAULT_NAMESPACE.to_string());
    }

    let Some(key) = extract_api_key(headers) else {
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": "invalid or missing API key"})),
        )
            .into_response());
    };

    if let Some(ns) = tenant_keys.get(&key) {
        return Ok(ns.clone());
    }
    if let Some(expected) = global_key {
        if key == expected {
            if let Some(ns) = headers.get("x-namespace").and_then(|v| v.to_str().ok()) {
                return Ok(dv_query::normalize_namespace(ns));
            }
            return Ok(dv_query::DEFAULT_NAMESPACE.to_string());
        }
    }

    Err((
        StatusCode::UNAUTHORIZED,
        Json(json!({"error": "invalid or missing API key"})),
    )
        .into_response())
}
