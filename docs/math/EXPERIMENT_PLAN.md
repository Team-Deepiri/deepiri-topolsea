# Experiment plan — Z-Column vs HNSW

## Goals
Map dimensionless groups `(τ, ρ_r, λ, β, φ)` and find (or prove absence of) regimes that pass G1∧G2∧G3 under **honest** search (no accidental exhaustive fallback).

## Gates
| Gate | Pass if |
|---|---|
| G1 Recall | `recall_Z / recall_HNSW ≥ 0.98` at k=10 |
| G2 Latency | `p50_Z / p50_HNSW ≤ 1.5` |
| G3 Touch | `V_touch / N < 0.5` |

## Matrix
| Factor | Values |
|---|---|
| N | 2 000, 10 000, 50 000 |
| dim | 128 |
| distro | unit sphere; 32 Gaussian clusters σ=0.08 |
| ef | 8…128 |
| rings / beam / fcols | pure beam (0/0/0) → light → default-like |
| seed | 42 (+ seed sensitivity {1,42,999}) |

## Commands
```bash
cargo run -p dv-bench --release --bin topolsea-math-probe -- --json
# optional faster: --quick
cargo run -p dv-bench --release --bin topolsea-prove -- --scale 10000 --queries 50
```

## Held-out / next after M-graph
- ANN-Benchmarks datasets (SIFT1M / glove) once centroid-graph search exists  
- Equal-memory HNSW M/efConstruction sweep  
- Text embedding dumps from Deepiri RAG stacks  
