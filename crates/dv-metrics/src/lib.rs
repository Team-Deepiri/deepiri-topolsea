mod batch;
mod column_scan;
mod quantized;
mod scalar;

pub use batch::{batch_distances, distance};
pub use column_scan::scan_column_distances;
pub use quantized::{
    decode, decode_u16, decode_u8, encode, l2_squared_u16, l2_squared_u8, quantized_distance,
};
pub use scalar::{cosine_distance, dot_product, l2_distance, l2_squared};
