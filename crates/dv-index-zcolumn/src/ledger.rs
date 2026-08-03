use serde::{Deserialize, Serialize};

const EMA_ALPHA: f32 = 0.1;

/// Z-axis access weight for a column — tracks how hot/cold it is.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AccessLedger {
    pub hit_count: u64,
    pub last_access: u64,
    pub ema_weight: f32,
}

impl AccessLedger {
    pub fn record_hit(&mut self, now_ms: u64) {
        self.hit_count += 1;
        self.last_access = now_ms;
        self.ema_weight = self.ema_weight * (1.0 - EMA_ALPHA) + EMA_ALPHA;
    }

    pub fn decay(&mut self, now_ms: u64, half_life_ms: u64) {
        if self.last_access == 0 || half_life_ms == 0 {
            return;
        }
        let elapsed = now_ms.saturating_sub(self.last_access);
        let factor = 0.5f32.powf(elapsed as f32 / half_life_ms as f32);
        self.ema_weight *= factor;
    }

    pub fn is_hot(&self, threshold: f32) -> bool {
        self.ema_weight >= threshold
    }

    pub fn is_cold(&self, threshold: f32) -> bool {
        self.ema_weight < threshold && self.hit_count > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decay_reduces_stale_weight() {
        let mut ledger = AccessLedger::default();
        ledger.record_hit(1_000);
        let before = ledger.ema_weight;
        ledger.decay(1_000 + 3_600_000, 3_600_000);
        assert!(ledger.ema_weight < before);
    }

    #[test]
    fn hit_increases_weight() {
        let mut ledger = AccessLedger::default();
        ledger.record_hit(1000);
        assert!(ledger.ema_weight > 0.0);
        assert_eq!(ledger.hit_count, 1);
    }
}
