use dv_types::{Result, SearchHit, Vector, VectorId};

/// Common interface for vector index implementations.
pub trait VectorIndex: Send + Sync {
    fn dimension(&self) -> usize;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn insert(&mut self, id: VectorId, vector: Vector) -> Result<()>;

    fn remove(&mut self, id: VectorId) -> Result<()>;

    fn get_vector(&self, id: VectorId) -> Result<Vector>;

    fn search(&self, query: &[f32], top_k: usize, ef: usize) -> Result<Vec<SearchHit>>;

    fn contains(&self, id: VectorId) -> bool;
}
