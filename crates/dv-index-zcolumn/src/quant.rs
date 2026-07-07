use dv_metrics::distance;
use dv_types::DistanceMetric;
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

/// Encode a vector at the given quantization tier.
pub fn encode(vector: &[f32], tier: QuantTier) -> Vec<u8> {
    match tier {
        QuantTier::U8 => vector
            .iter()
            .map(|&v| ((v.clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8)
            .collect(),
        QuantTier::U16 => {
            let mut out = Vec::with_capacity(vector.len() * 2);
            for &v in vector {
                let q = ((v.clamp(-1.0, 1.0) + 1.0) * 32767.5).round() as u16;
                out.extend_from_slice(&q.to_le_bytes());
            }
            out
        }
        QuantTier::F32 => {
            let mut out = Vec::with_capacity(vector.len() * 4);
            for &v in vector {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out
        }
    }
}

/// Decode quantized bytes back to f32.
pub fn decode(data: &[u8], tier: QuantTier, dimension: usize) -> Vec<f32> {
    match tier {
        QuantTier::U8 => data
            .iter()
            .take(dimension)
            .map(|&b| (b as f32 / 127.5) - 1.0)
            .collect(),
        QuantTier::U16 => {
            let mut out = Vec::with_capacity(dimension);
            for i in 0..dimension {
                let off = i * 2;
                if off + 2 > data.len() {
                    break;
                }
                let q = u16::from_le_bytes([data[off], data[off + 1]]);
                out.push((q as f32 / 32767.5) - 1.0);
            }
            out
        }
        QuantTier::F32 => {
            let mut out = Vec::with_capacity(dimension);
            for i in 0..dimension {
                let off = i * 4;
                if off + 4 > data.len() {
                    break;
                }
                let q =
                    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
                out.push(q);
            }
            out
        }
    }
}

/// Distance between query and quantized stored vector.
pub fn quantized_distance(
    metric: DistanceMetric,
    query: &[f32],
    data: &[u8],
    tier: QuantTier,
    dimension: usize,
) -> f32 {
    let decoded = decode(data, tier, dimension);
    if decoded.len() != query.len() {
        return f32::MAX;
    }
    distance(metric, query, &decoded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn u8_roundtrip_approx() {
        let v = vec![0.5, -0.3, 0.8];
        let enc = encode(&v, QuantTier::U8);
        let dec = decode(&enc, QuantTier::U8, 3);
        for (a, b) in v.iter().zip(dec.iter()) {
            assert!((a - b).abs() < 0.02);
        }
    }

    #[test]
    fn f32_roundtrip_exact() {
        let v = vec![0.5, -0.3, 0.8];
        let enc = encode(&v, QuantTier::F32);
        let dec = decode(&enc, QuantTier::F32, 3);
        for (a, b) in v.iter().zip(dec.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-6);
        }
    }
}
