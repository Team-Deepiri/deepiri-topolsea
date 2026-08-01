//! Shared helpers for dv-bench binaries.

/// L2-normalize a vector in place (no-op for near-zero vectors).
pub fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > f32::EPSILON {
        for x in &mut v {
            *x /= n;
        }
    }
    v
}
