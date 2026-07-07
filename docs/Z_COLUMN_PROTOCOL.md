# Z-Column Protocol — Why This Is a Different Species of Vector Engine

Topolsea Z-Column is not "HNSW with extra steps." It is a **fractal, vertically-addressed sparse tensor** for billion-scale ANN — built to beat graph indexes on density, explainability, and recovery.

## The Pitch (30 seconds)

Traditional vector DBs store points in a **flat grid**: find row (cluster), push depth (coil/slot). That's two operations, horizontal waste, and greedy graphs that can't backtrack.

Z-Column stores vectors in **nested vertical columns** that shrink toward the center like a fractal. One address axis. Gravity does the rest. When the query is wrong, **callback reverse** propagates a miss signal up the stack and tries sibling columns — fixing HNSW's "stuck at local optimum" problem.

## Core Primitives

| Primitive | What it does |
|-----------|--------------|
| `FractalGrid` | Spatial layers nesting toward center (`pitch_ratio`) |
| `RoutingProjection` | Seeded random projection → 2D fractal coordinates (not naive dim 0/1) |
| `ColumnStack` | Vertical LIFO stack per cell with quantized payloads |
| `LayerPredictor` | Predictive revert — start outer (generic) or inner (specific) |
| `RevertBeamSearch` | Callback-reverse beam search with `ef` width |
| `Hybrid rerank` | Coarse fractal pool → exact FP32 rerank on full vectors |
| `CompactionEngine` | Center collapse, hot promote, cold demote |
| `IndexPlanner` | Recommends Z-Column at scale (≥1k vectors, ≥64 dims) |

## Query Explain (the audit trail)

Every Z-Column query can return:

```json
{
  "entry_layer": 0,
  "deepest_layer": 2,
  "revert_count": 1,
  "columns_scanned": 14,
  "column_paths": ["0:3:2", "1:1:0", "2:0:1"],
  "strategy": "predictive_revert_hybrid_rerank",
  "planner_reason": "high-D large collection — fractal Z-Column with hybrid rerank"
}
```

This is your **observability moat** — no black-box ANN.

## Disaster Recovery

On open, if `index.bin` is gone but `vectors.bin` survives:

1. Rebuild full index from `vectors.bin`
2. Optionally hydrate column stacks from `columns/L*.grid.bin`

You don't lose the vending machine because someone deleted the graph file.

## API Surface

### Rust / CLI

```bash
topolsea create docs --dimension 384 --metric cosine --index zcolumn
topolsea search docs --vector 0.1,0.2,... --top-k 10 --explain
topolsea plan --size 1000000 --dimension 768 --top-k 20
```

### Python

```python
col = client.get_or_create_collection("docs", dimension=384, index="zcolumn")
out = col.explain_query(query_vector=vec, top_k=10)
stats = col.zcolumn_stats()
```

## Benchmarks

```bash
cargo bench -p dv-bench --bench index_bench
cargo bench -p dv-bench --bench recall_bench
```

Integration tests enforce **recall@10 within 15% of flat ground truth** on 200-vector synthetic workloads.

## Roadmap to Production Scale

| Milestone | Target |
|-----------|--------|
| M1 | Z-Column index + explain API ✅ |
| M2 | Hybrid rerank + planner ✅ |
| M3 | Column segment persistence + DR ✅ |
| M4 | Distributed fractal shards (column = partition key) |
| M5 | GPU batch projection + quantized scan |
| M6 | Learned layer predictor (replace heuristics) |

## API notes

- `Collection::query` and `query_explain` take `&mut self` so Z-Column can update per-column access ledgers (used by compaction). Other index kinds ignore this side effect.
- Access ledger timestamps use wall-clock milliseconds for decay; compaction promotes hot columns and demotes cold ones based on EMA weights.

## Go / No-Go (unchanged)

- recall@10 within 2% of HNSW at equal memory on 10k/128d
- p50 latency ≤ 1.5× HNSW
- callback reverse on < 30% of queries
- compaction preserves recall after 10k cycles

---

*The vending machine metaphor is the intuition. The payload is a new index species in your vector engine — with explainability, recovery, and a path to billion-vector fractal sharding.*
