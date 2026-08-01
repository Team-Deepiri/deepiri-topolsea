# Track M — Z-Column ANN honesty

Stacked on Phase C. Goal: make Z-Column a credible ANN under gates **G1∧G2∧G3**, or keep HNSW as the product default.

## Acceptance map

| ID | Delivered |
|---|---|
| **M0** | Honest knobs: do not inflate `ef` with `coarse_pool`; fallback only when caps &gt; 0; `used_fallback_scan` only when fired |
| **M3** | Beam / graph / fallback scan uses **quantized coarse** distances; FP32 only in hybrid rerank |
| **M4** | Hot promote **moves** (not copies); tall-column split when height &gt; `max_column_height_ratio × mean` |
| **M-graph** | kNN graph over nonempty column centroids; beam seeds + neighbor walk (`use_centroid_graph`) |
| **M2** | Hard `V_touch` budget (`touch_budget` or `touch_budget_frac × N`); explain reports `hit_touch_budget` |
| **M1** | Conditional fallback: ring/ranked only when heap &lt; k |
| **M5** | `topolsea-math-probe` + `topolsea-math-localize` restored; re-measure before marketing claims |

## Go / no-go (do not claim “beats HNSW” unless all pass)

| Gate | Pass if |
|---|---|
| **G1** | `recall_Z / recall_HNSW ≥ 0.98` @ k=10 |
| **G2** | `p50_Z / p50_HNSW ≤ 1.5` |
| **G3** | `candidates_touched / N < 0.5` |

## Config knobs (`ZColumnConfig`)

```json
{
  "use_centroid_graph": true,
  "graph_degree": 8,
  "conditional_fallback": true,
  "touch_budget_frac": 0.5,
  "touch_budget": null,
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

Until G1∧G2∧G3: **product ANN default remains HNSW**; Z-Column stays explain + fractal shard keys.

## Related

- [`docs/math/EXPERIMENT_RESULTS.md`](math/EXPERIMENT_RESULTS.md)
- [`docs/PHASE_A_PREP.md`](PHASE_A_PREP.md) (Track M table)
- [`docs/NEXT_STEPS.md`](NEXT_STEPS.md)
