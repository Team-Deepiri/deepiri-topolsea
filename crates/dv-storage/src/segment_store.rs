//! Sealed vector segments with mmap reads and incremental flush.

use crate::format::FileHeader;
use crate::segment::VectorSegment;
use dv_types::{Result, TopolseaError, VectorId};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SegmentManifest {
    pub dimension: usize,
    pub next_segment_id: u64,
    pub segments: Vec<SealedSegmentMeta>,
    /// Soft-deleted ids (skipped on read until compaction).
    #[serde(default)]
    pub deleted_ids: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SealedSegmentMeta {
    pub id: u64,
    pub file: String,
    pub count: u64,
}

/// Manages `{collection}/segments/` sealed vector files.
#[derive(Debug)]
pub struct SegmentStore {
    dir: PathBuf,
    dimension: usize,
}

impl SegmentStore {
    pub fn open(dir: impl AsRef<Path>, dimension: usize) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        Ok(Self { dir, dimension })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn manifest_path(&self) -> PathBuf {
        self.dir.join("manifest.json")
    }

    pub fn load_manifest(&self) -> Result<SegmentManifest> {
        let path = self.manifest_path();
        if !path.exists() {
            return Ok(SegmentManifest {
                dimension: self.dimension,
                ..Default::default()
            });
        }
        let bytes = fs::read(&path)?;
        let mut m: SegmentManifest = serde_json::from_slice(&bytes)?;
        if m.dimension == 0 {
            m.dimension = self.dimension;
        }
        Ok(m)
    }

    pub fn save_manifest(&self, manifest: &SegmentManifest) -> Result<()> {
        let tmp = self.dir.join("manifest.json.tmp");
        let bytes = serde_json::to_vec_pretty(manifest)?;
        fs::write(&tmp, bytes)?;
        fs::rename(tmp, self.manifest_path())?;
        Ok(())
    }

    /// Seal a new segment containing only `records` (incremental flush).
    pub fn seal_segment(
        &self,
        records: &[(VectorId, &[f32])],
    ) -> Result<Option<SealedSegmentMeta>> {
        if records.is_empty() {
            return Ok(None);
        }
        let mut manifest = self.load_manifest()?;
        let id = manifest.next_segment_id;
        manifest.next_segment_id += 1;
        let file = format!("seg_{id:06}.bin");
        let path = self.dir.join(&file);
        let seg = VectorSegment::new(&path, self.dimension);
        seg.write_all(records)?;
        let meta = SealedSegmentMeta {
            id,
            file,
            count: records.len() as u64,
        };
        manifest.segments.push(meta.clone());
        self.save_manifest(&manifest)?;
        Ok(Some(meta))
    }

    pub fn mark_deleted(&self, ids: &[VectorId]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let mut manifest = self.load_manifest()?;
        let mut set: HashSet<u64> = manifest.deleted_ids.iter().copied().collect();
        for id in ids {
            set.insert(id.raw());
        }
        manifest.deleted_ids = set.into_iter().collect();
        manifest.deleted_ids.sort_unstable();
        self.save_manifest(&manifest)
    }

    /// Read all live vectors via mmap (parse-then-copy, same pattern as column segments).
    pub fn read_all_mmap(&self) -> Result<Vec<(VectorId, Vec<f32>)>> {
        let manifest = self.load_manifest()?;
        let deleted: HashSet<u64> = manifest.deleted_ids.iter().copied().collect();
        let mut out = Vec::new();
        // Later segments win on duplicate ids.
        let mut seen: HashSet<u64> = HashSet::new();
        for meta in manifest.segments.iter().rev() {
            let path = self.dir.join(&meta.file);
            if !path.exists() {
                continue;
            }
            let records = read_segment_mmap(&path, self.dimension)?;
            for (id, vec) in records {
                if deleted.contains(&id.raw()) || seen.contains(&id.raw()) {
                    continue;
                }
                seen.insert(id.raw());
                out.push((id, vec));
            }
        }
        out.reverse(); // restore ascending insertion-ish order (optional)
        Ok(out)
    }

    /// Ids currently present in sealed segments (minus deletes).
    pub fn sealed_ids(&self) -> Result<HashSet<u64>> {
        let records = self.read_all_mmap()?;
        Ok(records.into_iter().map(|(id, _)| id.raw()).collect())
    }

    /// Compact: rewrite live vectors into a single sealed segment.
    pub fn compact(&self) -> Result<()> {
        let live = self.read_all_mmap()?;
        let refs: Vec<_> = live.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        // Clear old files after writing a fresh segment set.
        let mut manifest = SegmentManifest {
            dimension: self.dimension,
            next_segment_id: 0,
            segments: Vec::new(),
            deleted_ids: Vec::new(),
        };
        self.save_manifest(&manifest)?;
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("seg_") && name.ends_with(".bin") {
                let _ = fs::remove_file(entry.path());
            }
        }
        if let Some(meta) = self.seal_segment(&refs)? {
            manifest = self.load_manifest()?;
            let _ = meta;
            let _ = manifest;
        }
        Ok(())
    }
}

fn read_segment_mmap(path: &Path, dimension: usize) -> Result<Vec<(VectorId, Vec<f32>)>> {
    let file = File::open(path)?;
    let mmap = unsafe {
        memmap2::MmapOptions::new()
            .map(&file)
            .map_err(|e| TopolseaError::Storage(format!("mmap segment: {e}")))?
    };
    parse_vectors_from_bytes(&mmap[..], dimension)
}

fn parse_vectors_from_bytes(data: &[u8], dimension: usize) -> Result<Vec<(VectorId, Vec<f32>)>> {
    if data.len() < 24 {
        return Err(TopolseaError::Storage("segment too short".into()));
    }
    let mut magic = [0u8; 8];
    magic.copy_from_slice(&data[0..8]);
    let version = u32::from_le_bytes(data[8..12].try_into().unwrap());
    let dim = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
    let count = u64::from_le_bytes(data[16..24].try_into().unwrap()) as usize;
    let header = FileHeader {
        magic,
        version,
        dimension: dim as u32,
        count: count as u64,
    };
    header.validate()?;
    if dim != dimension {
        return Err(TopolseaError::DimensionMismatch {
            expected: dimension,
            got: dim,
        });
    }
    let record_bytes = 8 + dim * 4;
    let need = 24 + count * record_bytes;
    if data.len() < need {
        return Err(TopolseaError::Storage("segment truncated".into()));
    }
    let mut out = Vec::with_capacity(count);
    let mut off = 24;
    for _ in 0..count {
        let id = VectorId(u64::from_le_bytes(data[off..off + 8].try_into().unwrap()));
        off += 8;
        let mut vec = vec![0.0f32; dim];
        for slot in &mut vec {
            *slot = f32::from_le_bytes(data[off..off + 4].try_into().unwrap());
            off += 4;
        }
        out.push((id, vec));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn incremental_seal_and_mmap_read() {
        let dir = tempdir().unwrap();
        let store = SegmentStore::open(dir.path(), 2).unwrap();
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        store.seal_segment(&[(VectorId(1), a.as_slice())]).unwrap();
        store.seal_segment(&[(VectorId(2), b.as_slice())]).unwrap();
        let all = store.read_all_mmap().unwrap();
        assert_eq!(all.len(), 2);
        store.mark_deleted(&[VectorId(1)]).unwrap();
        let live = store.read_all_mmap().unwrap();
        assert_eq!(live.len(), 1);
        assert_eq!(live[0].0, VectorId(2));
    }
}
