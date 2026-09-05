//! Every endpoint that accepts a caller-supplied `top_k` / `ef` must reject an
//! absurd value instead of handing it to an allocator.
//!
//! The bound originally landed on `search` and `shard_query` only, and the four
//! other endpoints that reach the same query path -- `search_ns`, `hybrid`,
//! `sparse` and `explain` -- were left open, so the limit could be stepped
//! around by changing the URL. These tests pin all six.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use dv_query::Database;
use dv_server::{router, AppState};
use serde_json::json;
use tempfile::tempdir;
use tower::ServiceExt;

/// Comfortably past MAX_QUERY_K (65_536) while staying well clear of the
/// `usize::MAX` region, so a failure here is the guard and not an overflow.
const ABSURD_K: u64 = 36_028_797_018_963_968;

async fn app_with_demo_collection() -> axum::Router {
    let dir = tempdir().unwrap();
    // Leak the tempdir: the router outlives this helper and the data directory
    // must survive with it. Test process, so the leak is bounded and deliberate.
    let path = dir.keep();
    let db = Database::open(&path).unwrap().into_shared();
    let app = router(AppState::new(db, None));

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/collections")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"name":"demo","dimension":2,"metric":"l2","index":"flat"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/v1/collections/demo/upsert")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"ids":["a","b"],"vectors":[[1.0,0.0],[0.0,1.0]]}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(res.status().is_success(), "upsert failed: {}", res.status());

    app
}

async fn post(app: &axum::Router, uri: &str, body: serde_json::Value) -> StatusCode {
    app.clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
        .status()
}

#[tokio::test]
async fn absurd_top_k_is_rejected_on_every_query_endpoint() {
    let app = app_with_demo_collection().await;

    // (endpoint, body) for each route that accepts a caller-supplied top_k.
    let cases: Vec<(&str, serde_json::Value)> = vec![
        (
            "/v1/collections/demo/search",
            json!({"vector":[1.0,0.0],"top_k":ABSURD_K}),
        ),
        (
            "/v1/ns/_default/collections/demo/search",
            json!({"vector":[1.0,0.0],"top_k":ABSURD_K}),
        ),
        (
            "/v1/collections/demo/hybrid",
            json!({"vector":[1.0,0.0],"text":"x","top_k":ABSURD_K}),
        ),
        (
            "/v1/collections/demo/sparse",
            json!({"text":"x","top_k":ABSURD_K}),
        ),
        (
            "/v1/collections/demo/explain",
            json!({"vector":[1.0,0.0],"top_k":ABSURD_K}),
        ),
    ];

    for (uri, body) in cases {
        let status = post(&app, uri, body).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "{uri} accepted top_k={ABSURD_K}; the bound is bypassable via this route"
        );
    }
}

#[tokio::test]
async fn absurd_ef_is_rejected_including_via_the_nprobe_alias() {
    let app = app_with_demo_collection().await;

    // `ef` directly...
    let status = post(
        &app,
        "/v1/collections/demo/search",
        json!({"vector":[1.0,0.0],"top_k":5,"ef":ABSURD_K}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "ef was not bounded");

    // ...and through `nprobe`, which overrides ef before the guard runs.
    let status = post(
        &app,
        "/v1/collections/demo/search",
        json!({"vector":[1.0,0.0],"top_k":5,"nprobe":ABSURD_K}),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "nprobe bypassed the ef bound"
    );
}

#[tokio::test]
async fn the_documented_maximum_is_still_accepted() {
    let app = app_with_demo_collection().await;

    // 65_536 is the limit, not one past it: the guard must reject only above it,
    // otherwise a legal query starts failing.
    let status = post(
        &app,
        "/v1/collections/demo/search",
        json!({"vector":[1.0,0.0],"top_k":65_536}),
    )
    .await;
    assert!(
        status.is_success(),
        "top_k at the documented maximum was rejected with {status}"
    );
}
