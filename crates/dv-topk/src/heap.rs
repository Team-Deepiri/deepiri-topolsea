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

/// Upper bound on the capacity reserved up front by [`TopKHeap::new`].
///
/// `push` never lets the heap exceed `k`, so reserving `k + 1` eagerly is only
/// an optimisation -- but it made `k` a direct allocation lever. `k` reaches
/// this constructor straight from a request body (`top_k`/`ef` on the search
/// endpoints), so a single caller could name a `k` whose reservation cannot be
/// satisfied. A failed allocation is not a catchable panic: Rust calls
/// `handle_alloc_error`, which aborts the process.
///
/// Small `k` -- every realistic query -- still gets its exact reservation. A
/// large one now grows on demand, so the cost tracks the elements that actually
/// arrive rather than the number requested.
const MAX_RESERVE: usize = 1024;

impl TopKHeap {
    pub fn new(k: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(k.saturating_add(1).min(MAX_RESERVE)),
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

    pub fn capacity(&self) -> usize {
        self.k
    }

    pub fn at_capacity(&self) -> bool {
        self.len() >= self.k
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    pub fn best_distance(&self) -> Option<f32> {
        self.heap
            .iter()
            .map(|c| c.distance)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    pub fn farthest_distance(&self) -> Option<f32> {
        self.heap.peek().map(|c| c.distance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absurd_k_does_not_abort_the_process() {
        // `usize::MAX` here would have asked the allocator for 16 exabytes.
        // Reaching the assertions at all is the point: a failed reservation
        // aborts, so a regression kills the test binary rather than failing it.
        let heap = TopKHeap::new(usize::MAX);
        assert_eq!(heap.capacity(), usize::MAX, "logical capacity is still k");
        assert!(heap.is_empty());

        let mut heap = TopKHeap::new(1_000_000_000_000_000);
        heap.push(Candidate {
            id: VectorId(7),
            distance: 1.5,
        });
        let out = heap.into_sorted_vec();
        assert_eq!(
            out.len(),
            1,
            "a huge k still accepts and returns candidates"
        );
        assert_eq!(out[0].id.raw(), 7);
    }

    #[test]
    fn reserve_is_exact_for_realistic_k() {
        // The optimisation is preserved where it matters.
        let heap = TopKHeap::new(10);
        assert!(heap.heap.capacity() >= 11);
    }

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
