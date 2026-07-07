pub const COLUMN_MAGIC: &[u8; 8] = b"TOPZCOLM";
pub const COLUMN_VERSION: u32 = 2;

use dv_types::QuantTier;
use serde::{Deserialize, Serialize};

/// On-disk tier tag — alias of shared [`QuantTier`].
pub type QuantTierTag = QuantTier;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ColumnFileHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub layer: u8,
    pub quant_tier: QuantTierTag,
    pub dimension: u32,
    pub cell_count: u64,
}

impl ColumnFileHeader {
    pub fn new(layer: u8, quant_tier: QuantTierTag, dimension: usize, cell_count: u64) -> Self {
        Self {
            magic: *COLUMN_MAGIC,
            version: COLUMN_VERSION,
            layer,
            quant_tier,
            dimension: dimension as u32,
            cell_count,
        }
    }

    pub fn validate(&self) -> dv_types::Result<()> {
        if self.magic != *COLUMN_MAGIC {
            return Err(dv_types::TopolseaError::Storage(
                "invalid column magic bytes".into(),
            ));
        }
        if self.version != COLUMN_VERSION {
            return Err(dv_types::TopolseaError::Storage(format!(
                "unsupported column version {}",
                self.version
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZColumnManifest {
    pub outer_grid: (u16, u16),
    pub max_layers: u8,
    pub pitch_ratio: f32,
    pub dimension: usize,
    pub layer_files: Vec<String>,
}
