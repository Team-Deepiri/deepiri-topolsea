use crate::collection::PyCollection;
use dv_query::Database;
use dv_types::{DistanceMetric, IndexKind};
use pyo3::prelude::*;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

#[pyclass]
pub struct PyClient {
    db: Arc<Mutex<Database>>,
    #[allow(dead_code)]
    data_dir: PathBuf,
}

#[pymethods]
impl PyClient {
    #[new]
    fn new(data_dir: &str) -> PyResult<Self> {
        let db = Database::open(data_dir)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self {
            db: Arc::new(Mutex::new(db)),
            data_dir: PathBuf::from(data_dir),
        })
    }

    fn list_collections(&self) -> PyResult<Vec<String>> {
        let db = self.db.lock().unwrap();
        db.list_collections()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    #[pyo3(signature = (name, dimension, metric, index=None))]
    fn get_or_create_collection(
        &self,
        name: &str,
        dimension: usize,
        metric: &str,
        index: Option<&str>,
    ) -> PyResult<PyCollection> {
        let metric = DistanceMetric::from_str(metric)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        let index_kind = match index.unwrap_or("hnsw").to_lowercase().as_str() {
            "flat" => IndexKind::Flat,
            "zcolumn" => IndexKind::ZColumn,
            _ => IndexKind::Hnsw,
        };
        let mut db = self.db.lock().unwrap();
        db.get_or_create_collection_with_config(name, dimension, metric, index_kind)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyCollection {
            db: Arc::clone(&self.db),
            name: name.to_string(),
        })
    }
}
