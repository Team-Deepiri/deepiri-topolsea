//! G1∧G2∧G3 gate evaluation helpers for Track M (M5).

use serde::Serialize;

/// Protocol gates for Z-Column vs HNSW.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct GateThresholds {
    pub recall_ratio_min: f32,
    pub latency_ratio_max: f32,
    pub touch_frac_max: f32,
}

impl Default for GateThresholds {
    fn default() -> Self {
        Self {
            recall_ratio_min: 0.98,
            latency_ratio_max: 1.5,
            touch_frac_max: 0.5,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GateInput {
    pub recall_z: f32,
    pub recall_hnsw: f32,
    pub p50_z_ms: f32,
    pub p50_hnsw_ms: f32,
    pub candidates_touched: f32,
    pub corpus_n: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct GateReport {
    pub g1_recall: bool,
    pub g2_latency: bool,
    pub g3_touch: bool,
    pub recall_ratio: f32,
    pub latency_ratio: f32,
    pub touch_frac: f32,
    pub all_pass: bool,
}

impl GateReport {
    pub fn evaluate(input: &GateInput, thr: &GateThresholds) -> Self {
        let recall_ratio = input.recall_z / input.recall_hnsw.max(1e-6);
        let latency_ratio = input.p50_z_ms / input.p50_hnsw_ms.max(1e-6);
        let touch_frac = input.candidates_touched / input.corpus_n.max(1.0);
        let g1_recall = recall_ratio >= thr.recall_ratio_min;
        let g2_latency = latency_ratio <= thr.latency_ratio_max;
        let g3_touch = touch_frac < thr.touch_frac_max;
        Self {
            g1_recall,
            g2_latency,
            g3_touch,
            recall_ratio,
            latency_ratio,
            touch_frac,
            all_pass: g1_recall && g2_latency && g3_touch,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_pass_when_ratios_met() {
        let r = GateReport::evaluate(
            &GateInput {
                recall_z: 0.99,
                recall_hnsw: 1.0,
                p50_z_ms: 1.0,
                p50_hnsw_ms: 1.0,
                candidates_touched: 100.0,
                corpus_n: 1000.0,
            },
            &GateThresholds::default(),
        );
        assert!(r.all_pass);
    }

    #[test]
    fn fail_touch_when_exhaustive() {
        let r = GateReport::evaluate(
            &GateInput {
                recall_z: 1.0,
                recall_hnsw: 1.0,
                p50_z_ms: 1.0,
                p50_hnsw_ms: 1.0,
                candidates_touched: 900.0,
                corpus_n: 1000.0,
            },
            &GateThresholds::default(),
        );
        assert!(!r.g3_touch);
        assert!(!r.all_pass);
    }
}
