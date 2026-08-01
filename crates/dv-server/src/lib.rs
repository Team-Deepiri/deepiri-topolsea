//! Deepiri Topolsea HTTP service library (Phase A3).

pub mod auth;
pub mod routes;
pub mod state;

pub use routes::router;
pub use state::AppState;
