# Experiment plan — Z-Column vs HNSW

## Goals
Falsify or support protocol go/no-go from `docs/Z_COLUMN_PROTOCOL.md` using measurable quantities from applied-math discovery.

## Gates
| Gate | Pass if |
|---|---|
| G1 Recall | `recall_Z / recall_HNSW ≥ 0.98` at k=10 |
| G2 Latency | `p50_Z / p50_HNSW ≤ 1.5` |
| G3 Touch | `V_touch / N < 0.5` (sublinear spirit; protocol implied via columns) |
| G4 Revert | Prefer: fraction of queries with `revert_count>0` < 0.30 **when fallbacks disabled**; today’s avg count is reported separately |

## Matrix
| n | dim | metric | queries | seed | ef | k | rings | fcols |
|---|-----|--------|---------|------|----|---|-------|-------|
| 10_000 | 128 | cosine | 40–50 | 42 | 128 | 10 | sweep | sweep |

## Commands
```bash
cargo run -p dv-bench --release --bin topolsea-math-probe -- 10000
cargo run -p dv-bench --release --bin topolsea-prove -- --scale 10000 --queries 50
```

## Held-out later
- Clustered Gaussian blobs (not only unit sphere)
- 100k / 1M scales
- Equal-memory HNSW M/efConstruction sweep vs Z-Column quant tiers
- ANN-Benchmarks datasets (Phase B item 9)
