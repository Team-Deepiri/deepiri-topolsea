# Experiment results — 2026-07-31

Machine: local release build. Seed 42. Cosine, dim=128, k=10, ef=128.

## Fix applied before fair measure
`ZColumnConfig.max_fallback_columns` existed but was **not wired** into `ranked_column_fallback`. Wired in `dv-index-zcolumn` so ranked fallback respects the cap. Neighborhood rings still expand independently.

## `topolsea-prove` @ n=10_000 (default Z-Column config)

| Index | recall@10 | p50 ms | QPS | notes |
|---|---:|---:|---:|---|
| Flat | 1.000 | ~2.4 | ~405 | GT |
| HNSW | 0.954 | ~1.66–1.74 | ~570–600 | baseline |
| Z-Column | 1.000 | ~8.4–8.9 | ~110–118 | |

| Ratio | Value | Gate |
|---|---:|---|
| recall Z/HNSW | **1.048** | G1 **GO** |
| p50 Z/HNSW | **~5.1–5.5** | G2 **NO-GO** |
| touch V/N | **1.000** | G3 **NO-GO** |
| footprint Z/HNSW | ~1.96 | heavier serialized index |

## `topolsea-math-probe` fallback sweep @ n=10_000, 40 queries

HNSW recall@10 ≈ 0.9525, p50 ≈ 1.796 ms

| rings | fcols | recall | vs HNSW | p50 ms | ×HNSW | avg cands | avg cols | revert_avg |
|------:|------:|-------:|--------:|-------:|-------:|----------:|---------:|-----------:|
| 0 | 0 | 0.815 | 0.856 | 8.07 | 4.49 | 7849 | 37 | 0.00 |
| 1 | 16 | 0.955 | 1.003 | 8.95 | 4.99 | 9714 | 46 | 0.00 |
| 2 | 32 | 1.000 | 1.050 | 9.86 | 5.49 | 10000 | 73 | 0.00 |
| 2 | 96 | 1.000 | 1.050 | 9.49 | 5.29 | 10000 | 73 | 0.00 |
| 4 | 96 | 1.000 | 1.050 | 8.85 | 4.93 | 10000 | 158 | 0.00 |
| 8 | 96 | 1.000 | 1.050 | 10.01 | 5.57 | 10000 | 329 | 0.00 |
| 8 | 10000 | 1.000 | 1.050 | 9.31 | 5.18 | 10000 | 329 | 0.00 |

## Interpretation (applied-math)

1. **High recall under default settings is purchased by near-exhaustive candidate touch**, not by sparse fractal walk.
2. **Pure beam (rings=0,fcols=0) misses G1** (0.856× HNSW) and still touches ~78% of corpus because `ef=128` opens fat columns with exact FP32 scans.
3. **Callback-reverse is idle** (`revert_avg=0`) — the novel control path is not load-bearing under these budgets.
4. **Latency gate fails in every regime tested** (~4.5–5.5× HNSW).

## Overall go/no-go (novel ANN claims at 10k/128d)

| Claim | Result |
|---|---|
| Recall within 2% of HNSW | GO only when touch≈1; **NO-GO under bounded touch** |
| p50 ≤ 1.5× HNSW | **NO-GO** |
| Sublinear scan / density win | **NO-GO** |
| Explainability + DR rebuild | Still valid product features (not refuted) |
| Fractal column = shard key | Valid engineering primitive (M4) |

**Decision:** do **not** market Z-Column as HNSW replacement on speed/density until a bounded-`τ` regime passes G1∧G2. Proceed with Phase A database hardening in parallel; keep a **math fix track** (conditional fallback, vector budget, coarse scan, compaction move-not-copy) as prerequisite to public ANN-Benchmarks claims.
