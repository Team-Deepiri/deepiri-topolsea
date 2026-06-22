pub const MAGIC: &[u8; 8] = b"TOPOLSEA";
pub const VERSION: u32 = 1;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub dimension: u32,
    pub count: u64,
}

impl FileHeader {
    pub fn new(dimension: usize, count: u64) -> Self {
        Self {
            magic: *MAGIC,
            version: VERSION,
            dimension: dimension as u32,
            count,
        }
    }

    pub fn validate(&self) -> dv_types::Result<()> {
        if self.magic != *MAGIC {
            return Err(dv_types::TopolseaError::Storage(
                "invalid magic bytes".into(),
            ));
        }
        if self.version != VERSION {
            return Err(dv_types::TopolseaError::Storage(format!(
                "unsupported version {}",
                self.version
            )));
        }
        Ok(())
    }
}
