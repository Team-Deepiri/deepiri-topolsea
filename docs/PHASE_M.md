# Track M — Z-Column ANN honesty (full)

Stacked on Phase C. Goal: make Z-Column a credible ANN under gates **G1∧G2∧G3**, or keep HNSW as the product default.

## Acceptance map

| ID | Delivered |
|---|---|
| **M0** | Honest knobs: do not inflate `ef` with `coarse_pool`; fallback only when caps &gt; 0; `used_fallback_scan` only when fired |
| **M3** | Quantized coarse scan + **intra-column prune** (`coarse_keep_per_column`); FP32 only at hybrid rerank; explain `coarse_scored` / `coarse_kept` |
| **M4** | Hot promote **moves** (not copies); tall-column split; **centroid rebuild** after move/split/remove |
| **M-graph** | Cached kNN graph over centroids; hop-limited BFS (`graph_beam_hops`); rebuild on mutate/rebalance |
| **M2** | Hard `V_touch` budget (`touch_budget` or `touch_budget_frac × N`); mid-column stop; `hit_touch_budget` |
| **M1** | Conditional fallback: heap &lt; k **or** farthest/best score gap |
| **M5** | `topolsea-math-probe` / `topolsea-math-localize`; `dv_bench::GateReport` for G1∧G2∧G3 |

## Go / no-go (do not claim “beats HNSW” unless all pass)

| Gate | Pass if |
|---|---|
| **G1** | `recall_Z / recall_HNSW ≥ 0.98` @ k=10 |
| **G2** | `p50_Z / p50_HNSW ≤ 1.5` |
| **G3** | `candidates_touched / N < 0.5` |

Whole-column expand of oracle’s ~8 tall columns still fails G3 (τ≈0.68). **M3 prune** is what makes visiting those columns compatible with a touch budget.

## Config knobs (`ZColumnConfig`)

```json
{
  "use_centroid_graph": true,
  "graph_degree": 8,
  "graph_beam_hops": 3,
  "conditional_fallback": true,
  "fallback_score_gap": 2.0,
  "touch_budget_frac": 0.5,
  "touch_budget": null,
  "coarse_keep_per_column": 32,
  "max_column_height_ratio": 4.0,
  "fallback_beam_radius": 2,
  "max_fallback_rings": 8,
  "max_fallback_columns": 96
}
```

Pure-beam / gate experiments: set `fallback_beam_radius=0`, `max_fallback_rings=0`, `max_fallback_columns=0`.

## Re-measure (M5)

```bash
cargo run -p dv-bench --release --bin topolsea-math-probe -- --json
cargo run -p dv-bench --release --bin topolsea-math-localize -- --n=10000
```

Use `GateReport::evaluate` to print G1/G2/G3. Until all pass: **product ANN default remains HNSW**.

## Related

- [`docs/math/EXPERIMENT_RESULTS.md`](math/EXPERIMENT_RESULTS.md)
- [`docs/PHASE_A_PREP.md`](PHASE_A_PREP.md)
- [`docs/NEXT_STEPS.md`](NEXT_STEPS.md)
