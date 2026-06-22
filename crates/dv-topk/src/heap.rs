use dv_types::VectorId;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    pub id: VectorId,
    pub distance: f32,
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap by distance: largest distance sits at the top for eviction.
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Fixed-size max-heap retaining the k closest candidates (smallest distance).
pub struct TopKHeap {
    heap: BinaryHeap<Candidate>,
    k: usize,
}

impl TopKHeap {
    pub fn new(k: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(k + 1),
            k,
        }
    }

    pub fn push(&mut self, candidate: Candidate) {
        if self.k == 0 {
            return;
        }
        if self.heap.len() < self.k {
            self.heap.push(candidate);
            return;
        }
        if let Some(worst) = self.heap.peek() {
            if candidate.distance < worst.distance {
                self.heap.pop();
                self.heap.push(candidate);
            }
        }
    }

    pub fn into_sorted_vec(mut self) -> Vec<Candidate> {
        let mut v: Vec<Candidate> = self.heap.drain().collect();
        v.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        v
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    pub fn farthest_distance(&self) -> Option<f32> {
        self.heap.peek().map(|c| c.distance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_k_smallest() {
        let mut heap = TopKHeap::new(2);
        for (id, dist) in [(0, 3.0), (1, 1.0), (2, 2.0), (3, 0.5)] {
            heap.push(Candidate {
                id: VectorId(id),
                distance: dist,
            });
        }
        let results = heap.into_sorted_vec();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id.raw(), 3);
        assert_eq!(results[1].id.raw(), 1);
    }
}
