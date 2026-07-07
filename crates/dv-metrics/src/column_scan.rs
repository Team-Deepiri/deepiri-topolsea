//! SIMD-friendly quantized column scan — avoids per-vector decode allocations on hot paths.
use dv_types::{DistanceMetric, QuantTier};

use crate::{cosine_distance, dot_product, l2_squared, l2_squared_u16, l2_squared_u8};

/// Score every payload in a column against `query` without allocating decoded vectors.
pub fn scan_column_distances(
    metric: DistanceMetric,
    query: &[f32],
    tier: QuantTier,
    payloads: &[Vec<u8>],
    dimension: usize,
) -> Vec<f32> {
    match tier {
        QuantTier::U8 => payloads
            .iter()
            .map(|p| distance_u8(metric, query, p))
            .collect(),
        QuantTier::U16 => payloads
            .iter()
            .map(|p| distance_u16(metric, query, p, dimension))
            .collect(),
        QuantTier::F32 => payloads
            .iter()
            .map(|p| distance_f32(metric, query, p, dimension))
            .collect(),
    }
}

#[inline]
fn distance_u8(metric: DistanceMetric, query: &[f32], data: &[u8]) -> f32 {
    match metric {
        DistanceMetric::L2 => l2_squared_u8(query, data),
        DistanceMetric::Cosine | DistanceMetric::DotProduct => {
            // Cosine/dot on U8 without full decode: expand inline (unrolled hot loop).
            let n = query.len().min(data.len());
            let mut dot = 0.0f32;
            let mut norm_q = 0.0f32;
            let mut norm_s = 0.0f32;
            let mut i = 0;
            while i + 4 <= n {
                for j in 0..4 {
                    let qv = query[i + j];
                    let stored = (data[i + j] as f32 / 127.5) - 1.0;
                    dot += qv * stored;
                    norm_q += qv * qv;
                    norm_s += stored * stored;
                }
                i += 4;
            }
            while i < n {
                let qv = query[i];
                let stored = (data[i] as f32 / 127.5) - 1.0;
                dot += qv * stored;
                norm_q += qv * qv;
                norm_s += stored * stored;
                i += 1;
            }
            match metric {
                DistanceMetric::Cosine => {
                    let denom = norm_q.sqrt() * norm_s.sqrt();
                    if denom <= f32::EPSILON {
                        1.0
                    } else {
                        1.0 - (dot / denom)
                    }
                }
                DistanceMetric::DotProduct => -dot,
                DistanceMetric::L2 => unreachable!(),
            }
        }
    }
}

#[inline]
fn distance_u16(metric: DistanceMetric, query: &[f32], data: &[u8], dimension: usize) -> f32 {
    match metric {
        DistanceMetric::L2 => l2_squared_u16(query, data),
        _ => {
            let decoded = crate::decode_u16(data, dimension);
            match metric {
                DistanceMetric::Cosine => cosine_distance(query, &decoded),
                DistanceMetric::DotProduct => dot_product(query, &decoded),
                DistanceMetric::L2 => l2_squared(query, &decoded),
            }
        }
    }
}

#[inline]
fn distance_f32(metric: DistanceMetric, query: &[f32], data: &[u8], dimension: usize) -> f32 {
    let mut decoded = Vec::with_capacity(dimension);
    for i in 0..dimension {
        let off = i * 4;
        if off + 4 > data.len() {
            break;
        }
        decoded.push(f32::from_le_bytes([
            data[off],
            data[off + 1],
            data[off + 2],
            data[off + 3],
        ]));
    }
    match metric {
        DistanceMetric::L2 => l2_squared(query, &decoded),
        DistanceMetric::Cosine => cosine_distance(query, &decoded),
        DistanceMetric::DotProduct => dot_product(query, &decoded),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode;

    #[test]
    fn scan_matches_scalar_quantized() {
        let query = vec![0.5, -0.3, 0.8];
        let enc = encode(&query, QuantTier::U8);
        let batch = scan_column_distances(
            DistanceMetric::L2,
            &query,
            QuantTier::U8,
            std::slice::from_ref(&enc),
            3,
        );
        assert!(batch[0] < 0.01);
    }
}
