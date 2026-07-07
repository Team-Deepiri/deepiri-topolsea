use super::{cosine_distance, dot_product, l2_squared};
use dv_types::{DistanceMetric, QuantTier};

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
        QuantTier::U8 => decode_u8(data, dimension),
        QuantTier::U16 => decode_u16(data, dimension),
        QuantTier::F32 => {
            let mut out = Vec::with_capacity(dimension);
            for i in 0..dimension {
                let off = i * 4;
                if off + 4 > data.len() {
                    break;
                }
                out.push(f32::from_le_bytes([
                    data[off],
                    data[off + 1],
                    data[off + 2],
                    data[off + 3],
                ]));
            }
            out
        }
    }
}

/// Squared L2 distance against U8-quantized stored vector.
#[inline]
pub fn l2_squared_u8(query: &[f32], data: &[u8]) -> f32 {
    let n = query.len().min(data.len());
    let mut sum = 0.0f32;
    for (i, qv) in query.iter().enumerate().take(n) {
        let stored = (data[i] as f32 / 127.5) - 1.0;
        let d = *qv - stored;
        sum += d * d;
    }
    sum
}

/// Squared L2 distance against U16-quantized stored vector.
#[inline]
pub fn l2_squared_u16(query: &[f32], data: &[u8]) -> f32 {
    let mut sum = 0.0f32;
    for (i, qv) in query.iter().enumerate() {
        let off = i * 2;
        if off + 2 > data.len() {
            break;
        }
        let q = u16::from_le_bytes([data[off], data[off + 1]]);
        let stored = (q as f32 / 32767.5) - 1.0;
        let d = *qv - stored;
        sum += d * d;
    }
    sum
}

/// Decode U8 bytes to f32 for metric computation.
pub fn decode_u8(data: &[u8], dimension: usize) -> Vec<f32> {
    data.iter()
        .take(dimension)
        .map(|&b| (b as f32 / 127.5) - 1.0)
        .collect()
}

/// Decode U16 bytes to f32 for metric computation.
pub fn decode_u16(data: &[u8], dimension: usize) -> Vec<f32> {
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

/// Distance between query and quantized payload using the given metric.
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
    match metric {
        DistanceMetric::L2 => l2_squared(query, &decoded).sqrt(),
        DistanceMetric::Cosine => cosine_distance(query, &decoded),
        DistanceMetric::DotProduct => dot_product(query, &decoded),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn u8_l2_reasonable() {
        let query = vec![0.5, -0.3];
        let data = vec![191, 89]; // approx encoded
        let d = l2_squared_u8(&query, &data);
        assert!(d < 0.5);
    }

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
