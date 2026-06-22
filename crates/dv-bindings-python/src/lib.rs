mod client;
mod collection;

use pyo3::prelude::*;

#[pymodule]
fn topolsea_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<client::PyClient>()?;
    m.add_class::<collection::PyCollection>()?;
    Ok(())
}
