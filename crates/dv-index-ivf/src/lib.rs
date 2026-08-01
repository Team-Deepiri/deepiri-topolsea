//! Inverted File (IVF) index with optional Product Quantization.

mod index;
mod pq;

pub use index::IvfIndex;
pub use pq::{asymmetric_distance, decode_pq, encode_pq, train_pq_codebooks, PqCodebooks};
