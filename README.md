# Deepiri Topolsea

A production-grade vector database engine written in **Rust**, with a **Python** client library.

Topolsea provides real approximate-nearest-neighbor (ANN) search via **HNSW** (Hierarchical Navigable Small World graphs), exact search via a **flat index**, on-disk **persistence**, **metadata filtering**, and SIMD-accelerated distance metrics.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Python API (deepiri_topolsea)                          │
│  Collection · Client · QueryBuilder · Filters           │
└────────────────────────┬────────────────────────────────┘
                         │ PyO3 (dv-bindings-python)
┌────────────────────────▼────────────────────────────────┐
│  Query Engine (dv-query)                                │
│  Database · Collection · hybrid search orchestration    │
└──────┬──────────────┬──────────────┬────────────────────┘
       │              │              │
┌──────▼──────┐ ┌─────▼─────┐ ┌─────▼──────┐
│ dv-index-   │ │ dv-index- │ │ dv-metadata│
│ hnsw        │ │ flat      │ │ + dv-storage│
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
| ANN index | HNSW | Sub-linear approximate search at scale |
| Exact index | Flat (brute-force) | Ground-truth / small collections |
| Distances | L2, Cosine, Dot Product | Metric-space search |
| Storage | Binary segments + JSON metadata sidecar | Durability |
| Python SDK | Poetry + Pydantic + NumPy | Application integration |
| Bindings | PyO3 | Zero-copy NumPy ↔ Rust vector transfer |

## Quick Start

### Rust

```bash
cargo build --release
cargo test
cargo run --release -p dv-cli -- --help
```

### Python

```bash
poetry install
poetry run pytest
```

```python
from deepiri_topolsea import Client

client = Client("./data")
col = client.get_or_create_collection("docs", dimension=384, metric="cosine")

col.upsert(
    ids=["a", "b"],
    vectors=[[0.1] * 384, [0.2] * 384],
    metadatas=[{"topic": "rust"}, {"topic": "python"}],
)

results = col.query(query_vector=[0.15] * 384, top_k=5, filter={"topic": "rust"})
```

## Crates

| Crate | Description |
|-------|-------------|
| `dv-types` | Core types, errors, configuration |
| `dv-metrics` | Distance functions |
| `dv-topk` | Top-K heap selection |
| `dv-index-api` | `VectorIndex` trait |
| `dv-index-flat` | Exact flat index |
| `dv-index-hnsw` | HNSW graph index |
| `dv-index-zcolumn` | Fractal Z-Column index |
| `dv-storage` | Binary persistence format |
| `dv-metadata` | Per-vector metadata + filters |
| `dv-query` | Database and collection API |
| `dv-bench` | Criterion benchmarks |
| `dv-bindings-python` | PyO3 extension module |
| `dv-cli` | Command-line interface |

## License

Apache-2.0 — see [LICENSE](LICENSE).
