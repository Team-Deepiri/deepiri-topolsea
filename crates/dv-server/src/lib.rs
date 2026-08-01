//! Deepiri Topolsea HTTP service library (Phase A3).

pub mod auth;
pub mod background;
pub mod routes;
pub mod state;

pub use background::{BackgroundServer, ServerConfig};
pub use routes::router;
pub use state::AppState;
