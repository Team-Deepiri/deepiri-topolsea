use crate::{Result, TopolseaError};

#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    pub data: Vec<f32>,
}

impl Vector {
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    pub fn from_slice(data: &[f32]) -> Self {
        Self {
            data: data.to_vec(),
        }
    }

    pub fn dimension(&self) -> usize {
        self.data.len()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn validate_dimension(&self, expected: usize) -> Result<()> {
        if self.dimension() != expected {
            return Err(TopolseaError::DimensionMismatch {
                expected,
                got: self.dimension(),
            });
        }
        Ok(())
    }

    pub fn l2_normalize(&mut self) {
        let norm: f32 = self.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for x in &mut self.data {
                *x /= norm;
            }
        }
    }
}
