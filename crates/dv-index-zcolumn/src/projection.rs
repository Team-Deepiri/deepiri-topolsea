use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Fixed random projection for routing high-D vectors into [0,1]^2 fractal space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingProjection {
    pub seed: u64,
    pub dimension: usize,
    row_x: Vec<f32>,
    row_y: Vec<f32>,
}

impl RoutingProjection {
    pub fn new(dimension: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = 1.0 / (dimension as f32).sqrt();
        let row_x: Vec<f32> = (0..dimension)
            .map(|_| if rng.gen_bool(0.5) { scale } else { -scale })
            .collect();
        let row_y: Vec<f32> = (0..dimension)
            .map(|_| if rng.gen_bool(0.5) { scale } else { -scale })
            .collect();
        Self {
            seed,
            dimension,
            row_x,
            row_y,
        }
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn project(&self, vector: &[f32]) -> (f32, f32) {
        let n = self.dimension.min(vector.len());
        let mut x = 0.0f32;
        let mut y = 0.0f32;
        for ((rx, ry), v) in self
            .row_x
            .iter()
            .zip(self.row_y.iter())
            .zip(vector.iter())
            .take(n)
        {
            x += rx * v;
            y += ry * v;
        }
        let px = (x.tanh() + 1.0) / 2.0;
        let py = (y.tanh() + 1.0) / 2.0;
        (px.clamp(0.0, 0.9999), py.clamp(0.0, 0.9999))
    }
}

impl Default for RoutingProjection {
    fn default() -> Self {
        Self::new(1, 42)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_is_deterministic() {
        let p = RoutingProjection::new(128, 42);
        let v = vec![0.1; 128];
        let a = p.project(&v);
        let b = p.project(&v);
        assert_eq!(a, b);
    }
}
