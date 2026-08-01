use dv_query::SharedDatabase;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub db: SharedDatabase,
    pub api_key: Option<Arc<str>>,
    /// When set, `/topolsea/v1/shard/query` serves this physical collection
    /// (replaces the old raw-TCP `ShardQueryServer`).
    pub shard_collection: Option<Arc<str>>,
}

impl AppState {
    pub fn new(db: SharedDatabase, api_key: Option<String>) -> Self {
        Self {
            db,
            api_key: api_key.map(Arc::<str>::from),
            shard_collection: None,
        }
    }

    pub fn with_shard_collection(mut self, name: impl Into<String>) -> Self {
        self.shard_collection = Some(Arc::<str>::from(name.into()));
        self
    }
}
