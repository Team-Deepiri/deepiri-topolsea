use crate::collection::PyCollection;
use dv_query::Database;
use dv_types::DistanceMetric;
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

    fn get_or_create_collection(
        &self,
        name: &str,
        dimension: usize,
        metric: &str,
    ) -> PyResult<PyCollection> {
        let metric = DistanceMetric::from_str(metric)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        let mut db = self.db.lock().unwrap();
        db.get_or_create_collection(name, dimension, metric)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyCollection {
            db: Arc::clone(&self.db),
            name: name.to_string(),
        })
    }
}
