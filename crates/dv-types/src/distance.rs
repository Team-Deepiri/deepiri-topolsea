use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DistanceMetric {
    #[default]
    L2,
    Cosine,
    DotProduct,
}

impl DistanceMetric {
    pub fn all() -> &'static [DistanceMetric] {
        &[
            DistanceMetric::L2,
            DistanceMetric::Cosine,
            DistanceMetric::DotProduct,
        ]
    }

    pub fn as_str(self) -> &'static str {
        match self {
            DistanceMetric::L2 => "l2",
            DistanceMetric::Cosine => "cosine",
            DistanceMetric::DotProduct => "dot_product",
        }
    }
}

impl fmt::Display for DistanceMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for DistanceMetric {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "l2" | "euclidean" => Ok(DistanceMetric::L2),
            "cosine" => Ok(DistanceMetric::Cosine),
            "dot" | "dot_product" | "ip" | "inner_product" => Ok(DistanceMetric::DotProduct),
            other => Err(format!("unknown distance metric: {other}")),
        }
    }
}
