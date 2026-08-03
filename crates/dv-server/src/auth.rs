use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

#[allow(clippy::result_large_err)]
pub fn check_api_key(headers: &HeaderMap, expected: Option<&str>) -> Result<(), Response> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let provided = headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .or_else(|| {
            headers
                .get(axum::http::header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.strip_prefix("Bearer ").map(|t| t.to_string()))
        });
    match provided {
        Some(key) if key == expected => Ok(()),
        _ => Err((
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": "invalid or missing API key"})),
        )
            .into_response()),
    }
}
