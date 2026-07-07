# Z-Column Protocol

The **Z-Column Protocol** is Topolsea's fractal, vertically-stacked vector index. It replaces the traditional 2D matrix model (row = cluster, column = depth) with a **3D sparse tensor**:

| Axis | Matrix model | Z-Column tensor |
|------|-------------|-----------------|
| X | drink type / cluster | spatial column coordinate |
| Y | depth (retired) | **height** — stack count per column |
| Z | — | **access weight** — time/frequency signal |

## Addressing

A vector's address is a variable-length recursive path through fractal layers:

```
[Layer0_X, Layer0_Y, Layer1_Offset, ..., StackIndex]
```

Outer layers cover the full projected space. Inner layers nest toward the center with halving cell pitch (`pitch_ratio`, default 0.5).

## Architecture

```
Collection API
    └── ZColumnIndex (dv-index-zcolumn)
            ├── FractalGrid        — spatial nesting
            ├── ColumnStack        — vertical vector stacks
            ├── LayerPredictor     — predictive revert entry
            ├── RevertBeamSearch   — callback-reverse backtracking
            ├── CompactionEngine   — center collapse + hot/cold migration
            └── AccessLedger       — Z-axis weight per column
```

## Search: Predictive Revert + Callback Reverse

1. **LayerPredictor** estimates query specificity from centroid distances. Generic queries start at layer 0; specific queries tunnel inward.
2. **RevertBeamSearch** descends the fractal grid with beam width `ef`. On dead-end, it **callbacks upward** to sibling columns at the parent layer — fixing HNSW's greedy no-backtrack limitation.
3. Full-precision vectors in `vectors` HashMap ground-truth distances; quantized column payloads accelerate scanning.

## Multi-Resolution Storage

| Layer | Quantization | Role |
|-------|-------------|------|
| 0 (outer) | U8 | hot/common vectors, fast scan |
| 1 (middle) | U16 | medium specificity |
| 2+ (inner) | F32 | rare/outlier exact match |

On-disk layout per collection:

```
{collection}/
├── manifest.json
├── index.bin              # serialized ZColumnIndex graph
├── vectors.bin            # full-precision backup
└── columns/
    ├── manifest.json
    ├── L0.grid.bin
    ├── L1.grid.bin
    └── L2.grid.bin
```

## Self-Compaction

`CompactionEngine` runs on `persist()`:

- **Center collapse**: empty innermost columns removed; fractal layer count may shrink
- **Hot promote**: high `ema_weight` vectors move to outer (coarser) layers
- **Cold demote**: low-access vectors sink to inner (finer) layers

## Usage

### Rust / CLI

```bash
cargo run -p dv-cli -- --data-dir ./data create mycol --dimension 128 --metric cosine --index zcolumn
```

### Python

```python
from deepiri_topolsea import Client

client = Client("./data")
col = client.get_or_create_collection("docs", dimension=384, metric="cosine", index="zcolumn")
col.upsert(ids=["a"], vectors=[[0.1] * 384])
results = col.query(query_vector=[0.15] * 384, top_k=5)
col.persist()
```

## Relation to HNSW

Z-Column is a parallel `IndexKind` alongside `Flat` and `Hnsw`. It borrows HNSW's hierarchical layer intuition but replaces random probabilistic promotion with **spatial fractal nesting**, adds **callback-reverse backtracking**, and tiers storage precision by layer depth.

## Go / No-Go Gates

- recall@10 within 2% of HNSW at equal memory on 10k/128d synthetic
- p50 latency ≤ 1.5× HNSW
- callback reverse on < 30% of queries
- compaction preserves recall after 10k insert/delete cycles

See `dv-bench` for comparative benchmarks.
