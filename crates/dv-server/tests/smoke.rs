use axum::body::Body;
use axum::http::{Request, StatusCode};
use dv_query::Database;
use dv_server::{router, AppState};
use http_body_util::BodyExt;
use serde_json::json;
use tempfile::tempdir;
use tower::ServiceExt;

#[tokio::test]
async fn health_create_upsert_search() {
    let dir = tempdir().unwrap();
    let db = Database::open(dir.path()).unwrap().into_shared();
    let app = router(AppState::new(db, None));

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

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
                    json!({
                        "ids": ["a", "b"],
                        "vectors": [[1.0, 0.0], [0.0, 1.0]],
                        "metadatas": [{"tag":"x"}, {"tag":"y"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/collections/demo/search")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "vector": [1.0, 0.0],
                        "top_k": 1,
                        "filter": {"tag": "x"}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["hits"][0]["id"], "a");
}
