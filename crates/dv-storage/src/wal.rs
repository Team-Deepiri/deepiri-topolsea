//! Append-only write-ahead log for crash-safe collection mutations.

use crc32fast::Hasher;
use dv_types::{Result, TopolseaError, VectorId};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub const WAL_MAGIC: &[u8; 8] = b"TOPWAL01";
pub const WAL_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WalRecord {
    Upsert {
        external_id: String,
        internal_id: u64,
        vector: Vec<f32>,
        metadata: Value,
    },
    Delete {
        external_id: String,
        internal_id: u64,
    },
}

#[derive(Debug)]
pub struct Wal {
    path: PathBuf,
    next_seq: u64,
}

impl Wal {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let next_seq = if path.exists() {
            let records = Self::read_all_from(&path)?;
            records.last().map(|(seq, _)| seq + 1).unwrap_or(1)
        } else {
            1
        };
        Ok(Self { path, next_seq })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }

    pub fn append(&mut self, record: &WalRecord) -> Result<u64> {
        let seq = self.next_seq;
        let payload = serde_json::to_vec(record)?;
        let mut hasher = Hasher::new();
        hasher.update(&seq.to_le_bytes());
        hasher.update(&(payload.len() as u32).to_le_bytes());
        hasher.update(&payload);
        let crc = hasher.finalize();

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;

        if file.metadata()?.len() == 0 {
            file.write_all(WAL_MAGIC)?;
            file.write_all(&WAL_VERSION.to_le_bytes())?;
        }

        file.write_all(&seq.to_le_bytes())?;
        file.write_all(&(payload.len() as u32).to_le_bytes())?;
        file.write_all(&crc.to_le_bytes())?;
        file.write_all(&payload)?;
        file.sync_all()?;

        self.next_seq = seq + 1;
        Ok(seq)
    }

    pub fn read_all(&self) -> Result<Vec<(u64, WalRecord)>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        Self::read_all_from(&self.path)
    }

    pub fn read_after(&self, after_seq: u64) -> Result<Vec<(u64, WalRecord)>> {
        Ok(self
            .read_all()?
            .into_iter()
            .filter(|(seq, _)| *seq > after_seq)
            .collect())
    }

    fn read_all_from(path: &Path) -> Result<Vec<(u64, WalRecord)>> {
        let mut file = File::open(path)?;
        let mut magic = [0u8; 8];
        if file.read(&mut magic)? == 0 {
            return Ok(Vec::new());
        }
        if &magic != WAL_MAGIC {
            return Err(TopolseaError::Storage(
                "invalid WAL magic — file corrupt or not a WAL".into(),
            ));
        }
        let mut ver_buf = [0u8; 4];
        file.read_exact(&mut ver_buf)?;
        let version = u32::from_le_bytes(ver_buf);
        if version != WAL_VERSION {
            return Err(TopolseaError::Storage(format!(
                "unsupported WAL version {version}"
            )));
        }

        let mut out = Vec::new();
        loop {
            let mut seq_buf = [0u8; 8];
            match file.read(&mut seq_buf)? {
                0 => break,
                n if n < 8 => {
                    return Err(TopolseaError::Storage("truncated WAL record (seq)".into()));
                }
                _ => {}
            }
            let seq = u64::from_le_bytes(seq_buf);

            let mut len_buf = [0u8; 4];
            file.read_exact(&mut len_buf)?;
            let len = u32::from_le_bytes(len_buf) as usize;

            let mut crc_buf = [0u8; 4];
            file.read_exact(&mut crc_buf)?;
            let crc = u32::from_le_bytes(crc_buf);

            let mut payload = vec![0u8; len];
            file.read_exact(&mut payload)?;

            let mut hasher = Hasher::new();
            hasher.update(&seq.to_le_bytes());
            hasher.update(&(len as u32).to_le_bytes());
            hasher.update(&payload);
            if hasher.finalize() != crc {
                return Err(TopolseaError::Storage(format!(
                    "WAL CRC mismatch at seq {seq}"
                )));
            }

            let record: WalRecord = serde_json::from_slice(&payload)?;
            out.push((seq, record));
        }
        Ok(out)
    }

    /// Truncate the WAL after a successful snapshot (atomic replace with empty file).
    pub fn truncate(&mut self) -> Result<()> {
        let tmp = self.path.with_extension("log.tmp");
        {
            let mut file = File::create(&tmp)?;
            file.write_all(WAL_MAGIC)?;
            file.write_all(&WAL_VERSION.to_le_bytes())?;
            file.sync_all()?;
        }
        fs::rename(&tmp, &self.path)?;
        self.next_seq = 1;
        Ok(())
    }
}

/// Helper for callers reconstructing upserts from WAL.
pub fn wal_upsert_ids(record: &WalRecord) -> Option<(String, VectorId)> {
    match record {
        WalRecord::Upsert {
            external_id,
            internal_id,
            ..
        } => Some((external_id.clone(), VectorId(*internal_id))),
        WalRecord::Delete { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn append_and_replay() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.log");
        let mut wal = Wal::open(&path).unwrap();
        let seq = wal
            .append(&WalRecord::Upsert {
                external_id: "a".into(),
                internal_id: 0,
                vector: vec![1.0, 2.0],
                metadata: json!({"k": 1}),
            })
            .unwrap();
        assert_eq!(seq, 1);
        wal.append(&WalRecord::Delete {
            external_id: "a".into(),
            internal_id: 0,
        })
        .unwrap();

        let wal2 = Wal::open(&path).unwrap();
        let records = wal2.read_all().unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].0, 1);
        assert_eq!(wal2.next_seq(), 3);
    }

    #[test]
    fn truncate_resets() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.log");
        let mut wal = Wal::open(&path).unwrap();
        wal.append(&WalRecord::Delete {
            external_id: "x".into(),
            internal_id: 1,
        })
        .unwrap();
        wal.truncate().unwrap();
        assert!(wal.read_all().unwrap().is_empty());
        assert_eq!(wal.next_seq(), 1);
    }
}
