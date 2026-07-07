//! Runtime-dispatched AVX2 (x86_64) / NEON (aarch64) L2 on U8-quantized vectors.

const U8_SCALE: f32 = 1.0 / 127.5;

/// Squared L2 with SIMD when available, scalar tail otherwise.
#[inline]
pub fn l2_squared_u8(query: &[f32], data: &[u8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { l2_squared_u8_avx2(query, data) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { l2_squared_u8_neon(query, data) };
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        l2_squared_u8_scalar(query, data)
    }
    #[cfg(target_arch = "x86_64")]
    {
        l2_squared_u8_scalar(query, data)
    }
}

#[inline]
pub fn l2_squared_u8_scalar(query: &[f32], data: &[u8]) -> f32 {
    let n = query.len().min(data.len());
    let mut sum = 0.0f32;
    for i in 0..n {
        let stored = (data[i] as f32 * U8_SCALE) - 1.0;
        let d = query[i] - stored;
        sum += d * d;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_squared_u8_avx2(query: &[f32], data: &[u8]) -> f32 {
    use std::arch::x86_64::*;

    let n = query.len().min(data.len());
    let scale = _mm256_set1_ps(U8_SCALE);
    let neg_one = _mm256_set1_ps(-1.0);
    let mut sum = _mm256_setzero_ps();
    let mut i = 0usize;

    while i + 8 <= n {
        let q = _mm256_loadu_ps(query.as_ptr().add(i));
        let bytes = _mm_loadl_epi64(data.as_ptr().add(i) as *const __m128i);
        let widened = _mm256_cvtepu8_epi32(bytes);
        let stored = _mm256_fmadd_ps(_mm256_cvtepi32_ps(widened), scale, neg_one);
        let d = _mm256_sub_ps(q, stored);
        sum = _mm256_fmadd_ps(d, d, sum);
        i += 8;
    }

    let mut total = horizontal_sum_f32x8(sum);
    while i < n {
        let stored = (data[i] as f32 * U8_SCALE) - 1.0;
        let d = query[i] - stored;
        total += d * d;
        i += 1;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn horizontal_sum_f32x8(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum4 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum4);
    let sums = _mm_add_ps(sum4, shuf);
    let shuf = _mm_movehl_ps(shuf, sums);
    let sums = _mm_add_ss(sums, shuf);
    _mm_cvtss_f32(sums)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn l2_squared_u8_neon(query: &[f32], data: &[u8]) -> f32 {
    use std::arch::aarch64::*;

    let n = query.len().min(data.len());
    let scale = vdupq_n_f32(U8_SCALE);
    let neg_one = vdupq_n_f32(-1.0);
    let mut sum = vdupq_n_f32(0.0);
    let mut i = 0usize;

    while i + 4 <= n {
        let q = vld1q_f32(query.as_ptr().add(i));
        let bytes = vld1_u8(data.as_ptr().add(i));
        let widened_u16 = vmovl_u8(bytes);
        let widened_u32 = vmovl_u16(vget_low_u16(widened_u16));
        let stored = vmlaq_f32(neg_one, vcvtq_f32_u32(widened_u32), scale);
        let d = vsubq_f32(q, stored);
        sum = vmlaq_f32(sum, d, d);
        i += 4;
    }

    let mut total = vaddvq_f32(sum);
    while i < n {
        let stored = (data[i] as f32 * U8_SCALE) - 1.0;
        let d = query[i] - stored;
        total += d * d;
        i += 1;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode;
    use dv_types::QuantTier;

    #[test]
    fn simd_matches_scalar() {
        let query = vec![0.5, -0.3, 0.8, 0.1, -0.9, 0.0, 0.4, -0.2, 0.7];
        let enc = encode(&query, QuantTier::U8);
        let scalar = l2_squared_u8_scalar(&query, &enc);
        let simd = l2_squared_u8(&query, &enc);
        assert!((scalar - simd).abs() < 1e-5, "scalar={scalar} simd={simd}");
    }
}
