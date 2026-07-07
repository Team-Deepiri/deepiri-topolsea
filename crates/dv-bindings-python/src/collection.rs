use dv_metadata::Filter;
use dv_query::Database;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use serde_json::Value;
use std::sync::{Arc, Mutex};

#[pyclass]
pub struct PyCollection {
    pub(crate) db: Arc<Mutex<Database>>,
    pub(crate) name: String,
}

#[pymethods]
impl PyCollection {
    fn name(&self) -> &str {
        &self.name
    }

    fn count(&self) -> PyResult<usize> {
        let mut db = self.db.lock().unwrap();
        let col = db
            .get_collection(&self.name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(col.len())
    }

    #[pyo3(signature = (ids, vectors, metadatas=None))]
    fn upsert(
        &self,
        ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
        metadatas: Option<Vec<Option<PyObject>>>,
    ) -> PyResult<()> {
        if ids.len() != vectors.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "ids and vectors length mismatch",
            ));
        }
        let meta_vec = metadatas.unwrap_or_default();
        if !meta_vec.is_empty() && meta_vec.len() != ids.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "metadatas length mismatch",
            ));
        }

        let mut db = self.db.lock().unwrap();
        let col = db
            .get_collection(&self.name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Python::with_gil(|py| {
            for (i, (id, vec)) in ids.into_iter().zip(vectors).enumerate() {
                let meta = if meta_vec.is_empty() {
                    None
                } else {
                    meta_vec.get(i).and_then(|m| m.as_ref()).map(|obj| {
                        let bound = obj.bind(py);
                        python_to_json(py, &bound).unwrap_or(Value::Null)
                    })
                };
                col.upsert(&id, vec, meta).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
            }
            Ok(())
        })
    }

    fn delete(&self, ids: Vec<String>) -> PyResult<()> {
        let mut db = self.db.lock().unwrap();
        let col = db
            .get_collection(&self.name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        for id in ids {
            col.delete(&id)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        }
        Ok(())
    }

    #[pyo3(signature = (query_vector, top_k=10, filter=None, ef=64))]
    fn query(
        &self,
        py: Python<'_>,
        query_vector: Vec<f32>,
        top_k: usize,
        filter: Option<&Bound<'_, PyAny>>,
        ef: usize,
    ) -> PyResult<PyObject> {
        let filter_rust = filter.map(|f| python_filter_to_rust(f)).transpose()?;

        let mut db = self.db.lock().unwrap();
        let col = db
            .get_collection(&self.name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let results = col
            .query(&query_vector, top_k, filter_rust.as_ref(), ef)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let list = PyList::empty_bound(py);
        for r in results {
            let dict = PyDict::new_bound(py);
            dict.set_item("id", r.id)?;
            dict.set_item("distance", r.distance)?;
            dict.set_item("score", r.score)?;
            if let Some(meta) = r.metadata {
                dict.set_item("metadata", json_to_python(py, &meta)?)?;
            }
            list.append(dict)?;
        }
        Ok(list.into())
    }

    #[pyo3(signature = (query_vectors, top_k=10, filter=None, ef=64))]
    fn query_batch(
        &self,
        py: Python<'_>,
        query_vectors: Vec<Vec<f32>>,
        top_k: usize,
        filter: Option<&Bound<'_, PyAny>>,
        ef: usize,
    ) -> PyResult<PyObject> {
        let filter_rust = filter.map(|f| python_filter_to_rust(f)).transpose()?;
        let refs: Vec<&[f32]> = query_vectors.iter().map(|v| v.as_slice()).collect();

        let mut db = self.db.lock().unwrap();
        let col = db
            .get_collection(&self.name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let batches = col
            .query_batch(&refs, top_k, filter_rust.as_ref(), ef)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let outer = PyList::empty_bound(py);
        for batch in batches {
            let list = PyList::empty_bound(py);
            for r in batch {
                let dict = PyDict::new_bound(py);
                dict.set_item("id", r.id)?;
                dict.set_item("distance", r.distance)?;
                dict.set_item("score", r.score)?;
                if let Some(meta) = r.metadata {
                    dict.set_item("metadata", json_to_python(py, &meta)?)?;
                }
                list.append(dict)?;
            }
            outer.append(list)?;
        }
        Ok(outer.into())
    }

    fn persist(&self) -> PyResult<()> {
        let mut db = self.db.lock().unwrap();
        let col = db
            .get_collection(&self.name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        col.persist()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(())
    }

    #[pyo3(signature = (query_vector, top_k=10, filter=None, ef=64))]
    fn explain_query(
        &self,
        py: Python<'_>,
        query_vector: Vec<f32>,
        top_k: usize,
        filter: Option<&Bound<'_, PyAny>>,
        ef: usize,
    ) -> PyResult<PyObject> {
        let filter_rust = filter.map(|f| python_filter_to_rust(f)).transpose()?;
        let mut db = self.db.lock().unwrap();
        let col = db
            .get_collection(&self.name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let (results, explain) = col
            .query_explain(&query_vector, top_k, filter_rust.as_ref(), ef)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let dict = PyDict::new_bound(py);
        let list = PyList::empty_bound(py);
        for r in results {
            let item = PyDict::new_bound(py);
            item.set_item("id", r.id)?;
            item.set_item("distance", r.distance)?;
            item.set_item("score", r.score)?;
            if let Some(meta) = r.metadata {
                item.set_item("metadata", json_to_python(py, &meta)?)?;
            }
            list.append(item)?;
        }
        dict.set_item("results", list)?;
        dict.set_item(
            "explain",
            json_to_python(
                py,
                &serde_json::to_value(explain).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?,
            )?,
        )?;
        Ok(dict.into())
    }

    fn zcolumn_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let mut db = self.db.lock().unwrap();
        let col = db
            .get_collection(&self.name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        match col.zcolumn_stats() {
            Some(v) => json_to_python(py, &v),
            None => Ok(py.None()),
        }
    }
}

fn python_filter_to_rust(filter: &Bound<'_, PyAny>) -> PyResult<Filter> {
    Python::with_gil(|py| {
        let json = python_to_json(py, filter)?;
        Filter::from_json(&json)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    })
}

fn python_to_json(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(Value::Bool(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Value::Number(i.into()));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(serde_json::Number::from_f64(f)
            .map(Value::Number)
            .unwrap_or(Value::Null));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Value::String(s));
    }
    if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (k, v) in dict.iter() {
            let key: String = k.extract()?;
            map.insert(key, python_to_json(py, &v)?);
        }
        return Ok(Value::Object(map));
    }
    if let Ok(list) = obj.downcast::<PyList>() {
        let mut arr = Vec::new();
        for item in list.iter() {
            arr.push(python_to_json(py, &item)?);
        }
        return Ok(Value::Array(arr));
    }
    Ok(Value::String(obj.str()?.to_string()))
}

fn json_to_python(py: Python<'_>, value: &Value) -> PyResult<PyObject> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => Ok(b.to_object(py)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.to_object(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.to_object(py))
            } else {
                Ok(py.None())
            }
        }
        Value::String(s) => Ok(s.to_object(py)),
        Value::Array(arr) => {
            let list = PyList::empty_bound(py);
            for v in arr {
                list.append(json_to_python(py, v)?)?;
            }
            Ok(list.into())
        }
        Value::Object(map) => {
            let dict = PyDict::new_bound(py);
            for (k, v) in map {
                dict.set_item(k, json_to_python(py, v)?)?;
            }
            Ok(dict.into())
        }
    }
}
