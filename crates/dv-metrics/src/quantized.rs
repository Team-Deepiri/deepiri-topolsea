use super::{cosine_distance, dot_product, l2_squared};
use dv_types::DistanceMetric;

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
    tier: QuantTierKind,
    dimension: usize,
) -> f32 {
    let decoded = match tier {
        QuantTierKind::U8 => decode_u8(data, dimension),
        QuantTierKind::U16 => decode_u16(data, dimension),
        QuantTierKind::F32 => {
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
    };
    if decoded.len() != query.len() {
        return f32::MAX;
    }
    match metric {
        DistanceMetric::L2 => l2_squared(query, &decoded).sqrt(),
        DistanceMetric::Cosine => cosine_distance(query, &decoded),
        DistanceMetric::DotProduct => dot_product(query, &decoded),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantTierKind {
    U8,
    U16,
    F32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u8_l2_reasonable() {
        let query = vec![0.5, -0.3];
        let data = vec![191, 89]; // approx encoded
        let d = l2_squared_u8(&query, &data);
        assert!(d < 0.5);
    }
}
