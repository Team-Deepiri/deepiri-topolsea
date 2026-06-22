use crate::format::FileHeader;
use dv_types::{Result, TopolseaError, VectorId};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Dense binary segment: header + [id: u64][vector: dim * f32]...
#[derive(Debug)]
pub struct VectorSegment {
    path: PathBuf,
    dimension: usize,
}

impl VectorSegment {
    pub fn new(path: impl AsRef<Path>, dimension: usize) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            dimension,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn write_all(&self, records: &[(VectorId, &[f32])]) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = File::create(&self.path)?;
        let header = FileHeader::new(self.dimension, records.len() as u64);
        write_header(&mut file, &header)?;
        for (id, vec) in records {
            if vec.len() != self.dimension {
                return Err(TopolseaError::DimensionMismatch {
                    expected: self.dimension,
                    got: vec.len(),
                });
            }
            file.write_all(&id.raw().to_le_bytes())?;
            for &v in *vec {
                file.write_all(&v.to_le_bytes())?;
            }
        }
        file.sync_all()?;
        Ok(())
    }

    pub fn read_all(&self) -> Result<Vec<(VectorId, Vec<f32>)>> {
        let mut file = File::open(&self.path)?;
        let header = read_header(&mut file)?;
        header.validate()?;
        let dim = header.dimension as usize;
        let mut out = Vec::with_capacity(header.count as usize);
        for _ in 0..header.count {
            let mut id_buf = [0u8; 8];
            file.read_exact(&mut id_buf)?;
            let id = VectorId(u64::from_le_bytes(id_buf));
            let mut vec = vec![0.0f32; dim];
            for slot in &mut vec {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *slot = f32::from_le_bytes(buf);
            }
            out.push((id, vec));
        }
        Ok(out)
    }
}

fn write_header(w: &mut impl Write, header: &FileHeader) -> Result<()> {
    w.write_all(&header.magic)?;
    w.write_all(&header.version.to_le_bytes())?;
    w.write_all(&header.dimension.to_le_bytes())?;
    w.write_all(&header.count.to_le_bytes())?;
    Ok(())
}

fn read_header(r: &mut impl Read) -> Result<FileHeader> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    let mut ver = [0u8; 4];
    r.read_exact(&mut ver)?;
    let mut dim = [0u8; 4];
    r.read_exact(&mut dim)?;
    let mut cnt = [0u8; 8];
    r.read_exact(&mut cnt)?;
    Ok(FileHeader {
        magic,
        version: u32::from_le_bytes(ver),
        dimension: u32::from_le_bytes(dim),
        count: u64::from_le_bytes(cnt),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn roundtrip_segment() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("seg.bin");
        let seg = VectorSegment::new(&path, 3);
        let data = [
            (VectorId(1), vec![1.0, 2.0, 3.0]),
            (VectorId(2), vec![4.0, 5.0, 6.0]),
        ];
        let refs: Vec<_> = data.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        seg.write_all(&refs).unwrap();
        let loaded = seg.read_all().unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].1, vec![1.0, 2.0, 3.0]);
    }
}
