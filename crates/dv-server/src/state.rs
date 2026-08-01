use dv_query::SharedDatabase;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub db: SharedDatabase,
    pub api_key: Option<Arc<str>>,
}

impl AppState {
    pub fn new(db: SharedDatabase, api_key: Option<String>) -> Self {
        Self {
            db,
            api_key: api_key.map(|s| Arc::<str>::from(s)),
        }
    }
}
