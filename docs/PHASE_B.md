# Phase B — RAG / product expectations

Implements Phase B from the production roadmap (hybrid search, segmented storage, IVF/PQ, ANN-Benchmarks).

Stacked on Phase A. Track M (Z-Column ANN gates) remains parallel — **do not** claim Z-Column beats HNSW as production ANN without G1∧G2∧G3.

## Acceptance map

| ID | Delivered |
|---|---|
| B6 | BM25 (`dv-sparse`) + dense fusion: **RRF** (default) and **linear** (`alpha`); `query_hybrid` / `query_hybrid_opts` / `query_sparse`; HTTP `/hybrid` + `/sparse`; upsert `texts`; WAL-durable text |
| B7 | Sealed mmap segments; incremental seal; soft-delete tombstones; **`POST .../compact`**; auto-compact on persist when deletes≥1k or segs≥32; rebuild ANN from segments if `index.bin` empty |
| B8 | `IndexKind::Ivf` + optional PQ; **`memory_bound`** drops raw vectors after seal (codes stay in RAM); HTTP create `ivf` config; search `nprobe` |
| B9 | `topolsea-ann-bench` (+ `--compare` Flat/HNSW/IVF); CI synthetic compare; published policy below |

## Hybrid search

```bash
curl -s -X PUT localhost:6333/v1/collections/demo/upsert -H 'content-type: application/json' \
  -d '{"ids":["a"],"vectors":[[1,0,0,0]],"texts":["quantum fractal topology"]}'

# RRF (default)
curl -s -X POST localhost:6333/v1/collections/demo/hybrid -H 'content-type: application/json' \
  -d '{"vector":[1,0,0,0],"text":"quantum topology","top_k":5}'

# Linear fusion (dense_weight = alpha)
curl -s -X POST localhost:6333/v1/collections/demo/hybrid -H 'content-type: application/json' \
  -d '{"vector":[1,0,0,0],"text":"quantum","fusion":"linear","dense_weight":0.7}'

# Sparse-only
curl -s -X POST localhost:6333/v1/collections/demo/sparse -H 'content-type: application/json' \
  -d '{"text":"quantum topology","top_k":5}'
```

## Segmented storage

On `persist()`:

1. Seal **new** full-precision vectors into `segments/seg_NNNNNN.bin` (incremental).
2. Soft-delete ids removed since last seal.
3. For IVF+PQ `memory_bound`: drop raw vectors from RAM after seal; rewrite `index.bin`.
4. Auto-compact when tombstones/segment count thresholds hit; or `POST /v1/collections/:name/compact`.

## IVF / PQ

```bash
curl -s -X POST localhost:6333/v1/collections -H 'content-type: application/json' \
  -d '{"name":"big","dimension":128,"metric":"l2","index":"ivf","ivf":{"nlist":256,"nprobe":16,"pq_m":16,"memory_bound":true}}'
```

Search: pass `"nprobe": 32` (or `ef`) on `/search`.

## ANN-Benchmarks (B9)

```bash
cargo run -p dv-bench --release --bin topolsea-ann-bench -- --index hnsw --top-k 10
cargo run -p dv-bench --release --bin topolsea-ann-bench -- --compare --top-k 10
```

### Published numbers policy

| Claim | Allowed when |
|---|---|
| HNSW recall / QPS | Always — product default ANN |
| IVF(+PQ) memory/QPS | Same dataset; prefer `--compare` |
| Z-Column “beats HNSW” | **Only** if Track M gates G1∧G2∧G3 pass |

Refresh JSON under `docs/bench/` when publishing release numbers.
