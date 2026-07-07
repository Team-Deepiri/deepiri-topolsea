use serde::{Deserialize, Serialize};

/// Multi-resolution quantization tier per fractal layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum QuantTier {
    #[default]
    U8,
    U16,
    F32,
}

impl QuantTier {
    pub fn for_layer(layer: u8, max_layers: u8) -> Self {
        if layer == 0 {
            QuantTier::U8
        } else if layer + 1 >= max_layers {
            QuantTier::F32
        } else {
            QuantTier::U16
        }
    }

    pub fn bytes_per_dim(&self) -> usize {
        match self {
            QuantTier::U8 => 1,
            QuantTier::U16 => 2,
            QuantTier::F32 => 4,
        }
    }
}
