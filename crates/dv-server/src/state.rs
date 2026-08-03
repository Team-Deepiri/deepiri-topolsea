use dv_observe::ServiceMetrics;
use dv_query::SharedDatabase;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub db: SharedDatabase,
    pub api_key: Option<Arc<str>>,
    /// When set, `/topolsea/v1/shard/query` serves this physical collection
    /// (replaces the old raw-TCP `ShardQueryServer`).
    pub shard_collection: Option<Arc<str>>,
    /// Optional map of API key → namespace (C13 tenant isolation).
    pub tenant_keys: Arc<HashMap<String, String>>,
    pub metrics: Arc<ServiceMetrics>,
}

impl AppState {
    pub fn new(db: SharedDatabase, api_key: Option<String>) -> Self {
        Self {
            db,
            api_key: api_key.map(Arc::<str>::from),
            shard_collection: None,
            tenant_keys: Arc::new(HashMap::new()),
            metrics: ServiceMetrics::shared(),
        }
    }

    pub fn with_shard_collection(mut self, name: impl Into<String>) -> Self {
        self.shard_collection = Some(Arc::<str>::from(name.into()));
        self
    }

    pub fn with_tenant_keys(mut self, keys: HashMap<String, String>) -> Self {
        self.tenant_keys = Arc::new(keys);
        self
    }

    pub fn with_metrics(mut self, metrics: Arc<ServiceMetrics>) -> Self {
        self.metrics = metrics;
        self
    }
}
