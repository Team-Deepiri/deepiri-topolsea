# Deepiri Topolsea

A production-grade vector database engine written in **Rust**, with a **Python** client library.

Topolsea provides real approximate-nearest-neighbor (ANN) search via **HNSW**, **Z-Column** (fractal nested column stacks with callback-reverse search), and exact **flat** search — with on-disk persistence, metadata filtering, fractal sharding, and SIMD-accelerated distance metrics.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Python API (deepiri_topolsea)                          │
│  Collection · Client · batch query · explain API        │
└────────────────────────┬────────────────────────────────┘
                         │ PyO3 (dv-bindings-python)
┌────────────────────────▼────────────────────────────────┐
│  Query Engine (dv-query)                                │
│  Database · Collection · IndexPlanner · FractalShardRouter│
└──────┬──────────────┬──────────────┬────────────────────┘
       │              │              │
┌──────▼──────┐ ┌─────▼─────┐ ┌─────▼──────┐
│ dv-index-   │ │ dv-index- │ │ dv-metadata│
│ hnsw        │ │ zcolumn   │ │ + dv-storage│
│             │ │ (fractal) │ │ (segments +│
│             │ │           │ │  shard manifest)│
└──────┬──────┘ └─────┬─────┘ └────────────┘
       │              │
┌──────▼──────────────▼──────────────────────────────────┐
│  dv-index-api · dv-metrics · dv-topk · dv-types        │
└─────────────────────────────────────────────────────────┘
```

## Stack

| Layer | Technology | Role |
|-------|-----------|------|
| Core engine | Rust 2021 | Index structures, distance math, persistence |
| ANN indexes | HNSW + **Z-Column** | Graph ANN + fractal column ANN with explainability |
| Exact index | Flat (brute-force) | Ground-truth / small collections |
| Sharding | Fractal column partition keys | Horizontal scale (M4) |
| Distances | L2, Cosine, Dot Product | Metric-space search |
| Storage | Binary segments + JSON metadata | Durability + disaster recovery |
| Python SDK | Poetry + Pydantic + NumPy | Application integration |
| Bindings | PyO3 | Zero-copy NumPy ↔ Rust vector transfer |

## Quick Start

### Rust / CLI

```bash
cargo build --release
topolsea create docs --dimension 384 --metric cosine --index zcolumn
topolsea shard-create corpus --shards 8 --dimension 384 --index zcolumn
topolsea search docs --vector 0.1,0.2,... --top-k 10 --explain
topolsea plan --size 1000000 --dimension 768
```

### Python

```bash
poetry install
poetry run pytest
```

```python
from deepiri_topolsea import Client

client = Client("./data")
col = client.get_or_create_collection("docs", dimension=384, metric="cosine", index="zcolumn")

col.upsert(
    ids=["a", "b"],
    vectors=[[0.1] * 384, [0.2] * 384],
    metadatas=[{"topic": "rust"}, {"topic": "python"}],
)

results = col.query(query_vector=[0.15] * 384, top_k=5)
batches = col.query_batch([[0.15] * 384, [0.2] * 384], top_k=5)
explain = col.explain_query(query_vector=[0.15] * 384, top_k=5)
```

## Z-Column (fractal index)

See [docs/Z_COLUMN_PROTOCOL.md](docs/Z_COLUMN_PROTOCOL.md) for the full protocol. Highlights:

- **Fractal nested columns** — vectors stack vertically per spatial cell, nesting toward center
- **Callback-reverse beam search** — backtracks on miss instead of getting stuck like greedy graphs
- **Hybrid rerank** — coarse fractal pool → exact FP32 rerank
- **Query explain API** — entry layer, revert count, column paths (observability moat)
- **Fractal sharding** — column key = partition key for billion-vector scale
- **Disaster recovery** — rebuild from `vectors.bin` when `index.bin` is lost

## Crates

| Crate | Description |
|-------|-------------|
| `dv-types` | Core types, errors, configuration, `QuantTier` |
| `dv-metrics` | Distance functions + quantization |
| `dv-topk` | Top-K heap selection |
| `dv-index-api` | `VectorIndex` trait |
| `dv-index-flat` | Exact flat index |
| `dv-index-hnsw` | HNSW graph index |
| `dv-index-zcolumn` | Fractal Z-Column index + routing |
| `dv-storage` | Binary persistence + shard manifests |
| `dv-metadata` | Per-vector metadata + filters |
| `dv-query` | Database, collection, planner, shard router |
| `dv-bench` | Criterion benchmarks (latency + recall) |
| `dv-bindings-python` | PyO3 extension module |
| `dv-cli` | Command-line interface |

## License

Apache-2.0 — see [LICENSE](LICENSE).
