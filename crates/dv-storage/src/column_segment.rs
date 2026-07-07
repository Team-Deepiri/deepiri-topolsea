use crate::column_format::{ColumnFileHeader, QuantTierTag};
use dv_types::{Result, TopolseaError, VectorId};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Serialized column cell: path key + stack of (id, quantized bytes).
#[derive(Debug, Clone)]
pub struct ColumnCellRecord {
    pub path_key: String,
    pub ids: Vec<VectorId>,
    pub payloads: Vec<Vec<u8>>,
    pub centroid: Vec<f32>,
}

/// Fractal column layer binary segment.
#[derive(Debug)]
pub struct ColumnSegment {
    path: PathBuf,
    dimension: usize,
    layer: u8,
    tier: QuantTierTag,
}

impl ColumnSegment {
    pub fn new(path: impl AsRef<Path>, dimension: usize, layer: u8, tier: QuantTierTag) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            dimension,
            layer,
            tier,
        }
    }

    pub fn write_all(&self, records: &[ColumnCellRecord]) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = File::create(&self.path)?;
        let header =
            ColumnFileHeader::new(self.layer, self.tier, self.dimension, records.len() as u64);
        write_header(&mut file, &header)?;

        for rec in records {
            let key_bytes = rec.path_key.as_bytes();
            let key_len = key_bytes.len() as u32;
            file.write_all(&key_len.to_le_bytes())?;
            file.write_all(key_bytes)?;

            file.write_all(&(rec.ids.len() as u32).to_le_bytes())?;
            for (id, payload) in rec.ids.iter().zip(&rec.payloads) {
                file.write_all(&id.raw().to_le_bytes())?;
                let plen = payload.len() as u32;
                file.write_all(&plen.to_le_bytes())?;
                file.write_all(payload)?;
            }

            file.write_all(&(rec.centroid.len() as u32).to_le_bytes())?;
            for &v in &rec.centroid {
                file.write_all(&v.to_le_bytes())?;
            }
        }
        file.sync_all()?;
        Ok(())
    }

    pub fn read_all(&self) -> Result<Vec<ColumnCellRecord>> {
        let mut file = File::open(&self.path)?;
        let header = read_header(&mut file)?;
        header.validate()?;
        let dim = header.dimension as usize;

        let mut out = Vec::with_capacity(header.cell_count as usize);
        for _ in 0..header.cell_count {
            let mut key_len_buf = [0u8; 4];
            file.read_exact(&mut key_len_buf)?;
            let key_len = u32::from_le_bytes(key_len_buf) as usize;
            let mut key_buf = vec![0u8; key_len];
            file.read_exact(&mut key_buf)?;
            let path_key =
                String::from_utf8(key_buf).map_err(|e| TopolseaError::Storage(e.to_string()))?;

            let mut count_buf = [0u8; 4];
            file.read_exact(&mut count_buf)?;
            let count = u32::from_le_bytes(count_buf) as usize;

            let mut ids = Vec::with_capacity(count);
            let mut payloads = Vec::with_capacity(count);
            for _ in 0..count {
                let mut id_buf = [0u8; 8];
                file.read_exact(&mut id_buf)?;
                ids.push(VectorId(u64::from_le_bytes(id_buf)));

                let mut plen_buf = [0u8; 4];
                file.read_exact(&mut plen_buf)?;
                let plen = u32::from_le_bytes(plen_buf) as usize;
                let mut payload = vec![0u8; plen];
                file.read_exact(&mut payload)?;
                payloads.push(payload);
            }

            let mut cent_len_buf = [0u8; 4];
            file.read_exact(&mut cent_len_buf)?;
            let cent_len = u32::from_le_bytes(cent_len_buf) as usize;
            let mut centroid = vec![0.0f32; cent_len.min(dim)];
            for slot in centroid.iter_mut().take(cent_len) {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *slot = f32::from_le_bytes(buf);
            }

            out.push(ColumnCellRecord {
                path_key,
                ids,
                payloads,
                centroid,
            });
        }
        Ok(out)
    }
}

fn write_header(w: &mut impl Write, header: &ColumnFileHeader) -> Result<()> {
    w.write_all(&header.magic)?;
    w.write_all(&header.version.to_le_bytes())?;
    w.write_all(&[header.layer])?;
    w.write_all(&[tier_tag_byte(header.quant_tier)])?;
    w.write_all(&header.dimension.to_le_bytes())?;
    w.write_all(&header.cell_count.to_le_bytes())?;
    Ok(())
}

fn read_header(r: &mut impl Read) -> Result<ColumnFileHeader> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    let mut ver = [0u8; 4];
    r.read_exact(&mut ver)?;
    let mut layer_buf = [0u8; 1];
    r.read_exact(&mut layer_buf)?;
    let mut tier_buf = [0u8; 1];
    r.read_exact(&mut tier_buf)?;
    let mut dim = [0u8; 4];
    r.read_exact(&mut dim)?;
    let mut cnt = [0u8; 8];
    r.read_exact(&mut cnt)?;
    Ok(ColumnFileHeader {
        magic,
        version: u32::from_le_bytes(ver),
        layer: layer_buf[0],
        quant_tier: tier_tag_from_byte(tier_buf[0]),
        dimension: u32::from_le_bytes(dim),
        cell_count: u64::from_le_bytes(cnt),
    })
}

fn tier_tag_byte(tier: QuantTierTag) -> u8 {
    match tier {
        QuantTierTag::U8 => 0,
        QuantTierTag::U16 => 1,
        QuantTierTag::F32 => 2,
    }
}

fn tier_tag_from_byte(b: u8) -> QuantTierTag {
    match b {
        1 => QuantTierTag::U16,
        2 => QuantTierTag::F32,
        _ => QuantTierTag::U8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn column_segment_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("L0.grid.bin");
        let seg = ColumnSegment::new(&path, 3, 0, QuantTierTag::U8);
        let records = vec![ColumnCellRecord {
            path_key: "0:1:2".to_string(),
            ids: vec![VectorId(1), VectorId(2)],
            payloads: vec![vec![1, 2, 3], vec![4, 5, 6]],
            centroid: vec![0.5, 0.5, 0.5],
        }];
        seg.write_all(&records).unwrap();
        let loaded = seg.read_all().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].ids.len(), 2);
        assert_eq!(loaded[0].path_key, "0:1:2");
    }
}
