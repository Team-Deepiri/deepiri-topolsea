use crate::{ExternalId, VectorId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchHit {
    pub id: VectorId,
    pub external_id: Option<ExternalId>,
    pub distance: f32,
    pub score: f32,
}

impl SearchHit {
    pub fn new(id: VectorId, distance: f32) -> Self {
        Self {
            id,
            external_id: None,
            distance,
            score: distance_to_score(distance),
        }
    }

    pub fn with_external_id(mut self, external_id: ExternalId) -> Self {
        self.external_id = Some(external_id);
        self
    }
}

/// Higher score = better match (inverse distance for ranking display).
fn distance_to_score(distance: f32) -> f32 {
    1.0 / (1.0 + distance)
}
