/// Squared L2 (Euclidean) distance — avoids sqrt when only ordering matters.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// L2 (Euclidean) distance.
#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    l2_squared(a, b).sqrt()
}

/// Cosine distance = 1 - cosine_similarity. Vectors need not be pre-normalized.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom <= f32::EPSILON {
        return 1.0;
    }
    1.0 - (dot / denom)
}

/// Negative inner product (smaller = more similar for max-inner-product search).
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn l2_identical_is_zero() {
        let v = [1.0, 2.0, 3.0];
        assert_relative_eq!(l2_distance(&v, &v), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn cosine_identical_is_zero() {
        let v = [1.0, 0.0, 0.0];
        assert_relative_eq!(cosine_distance(&v, &v), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn dot_product_opposite() {
        let a = [1.0, 0.0];
        let b = [-1.0, 0.0];
        assert!(dot_product(&a, &b) > dot_product(&a, &a));
    }
}
