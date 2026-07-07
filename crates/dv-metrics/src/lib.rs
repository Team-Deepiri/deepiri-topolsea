mod batch;
mod quantized;
mod scalar;

pub use batch::{batch_distances, distance};
pub use quantized::{
    decode_u16, decode_u8, l2_squared_u16, l2_squared_u8, quantized_distance, QuantTierKind,
};
pub use scalar::{cosine_distance, dot_product, l2_distance, l2_squared};
