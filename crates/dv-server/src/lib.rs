//! Deepiri Topolsea HTTP service library (Phase A3).
//!
//! Every handler and guard in this crate reports failure as an axum `Response`,
//! which is ~128 bytes, so `Result<T, Response>` trips `clippy::result_large_err`
//! wherever `T` is small. Five call sites already carried a local `#[allow]`;
//! Rust 1.98 widened the lint to fire on 32, which is more than a per-function
//! attribute can reasonably track.
//!
//! Allow it once for the crate. The alternative -- boxing the error -- would add
//! an allocation to every error path and change the signature of most public
//! handlers, for a type that never crosses a crate boundary.
#![allow(clippy::result_large_err)]

pub mod auth;
pub mod background;
pub mod routes;
pub mod state;

pub use background::{BackgroundServer, ServerConfig};
pub use routes::router;
pub use state::AppState;
