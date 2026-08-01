# Phase B — RAG / product expectations

Implements Phase B from the production roadmap (hybrid search, segmented storage, IVF/PQ, ANN-Benchmarks harness).

Stacked on Phase A. Track M (Z-Column ANN gates) remains parallel — **do not** claim Z-Column beats HNSW as production ANN without G1∧G2∧G3.

## Acceptance map

| ID | Delivered |
|---|---|
| B6 | BM25 sparse index (`dv-sparse`) + dense search fused with **RRF**; `Collection::query_hybrid` / `upsert_with_text`; HTTP `POST /v1/collections/:name/hybrid`; optional `texts` on upsert |
| B7 | Sealed vector segments under `{collection}/segments/` with **mmap** reads; `persist()` seals only the delta (no rewrite of prior segs); soft-delete tombstones |
| B8 | `IndexKind::Ivf` + `dv-index-ivf` (coarse IVF lists, optional PQ via `ivf.pq_m`); create with `"index":"ivf"` |
| B9 | `topolsea-ann-bench` harness (`.fvecs`/`.ivecs` or synthetic); published protocol in this doc — HNSW is the honest product default |

## Hybrid search

```bash
# upsert with document text
curl -s -X PUT localhost:6333/v1/collections/demo/upsert -H 'content-type: application/json' \
  -d '{"ids":["a"],"vectors":[[1,0,0,0]],"texts":["quantum fractal topology"]}'

# dense + BM25 via RRF
curl -s -X POST localhost:6333/v1/collections/demo/hybrid -H 'content-type: application/json' \
  -d '{"vector":[1,0,0,0],"text":"quantum topology","top_k":5}'
```

Python:

```python
from deepiri_topolsea import HttpClient
c = HttpClient("http://127.0.0.1:6333")
col = c.get_or_create_collection("demo", dimension=4)
col.upsert(["a"], [[1, 0, 0, 0]], texts=["quantum fractal topology"])
print(col.query_hybrid([1, 0, 0, 0], "quantum topology"))
```

## Segmented storage

On `persist()`:

1. Write/update `index.bin`, metadata, sparse blob as before.
2. Seal new vectors into `segments/seg_NNNNNN.bin` (incremental).
3. Mark ids removed since last seal in the segment manifest.
4. Drop legacy monolithic `vectors.bin` once segments are authoritative.

## IVF / PQ

```bash
curl -s -X POST localhost:6333/v1/collections -H 'content-type: application/json' \
  -d '{"name":"big","dimension":128,"metric":"l2","index":"ivf"}'
```

Tune via `CollectionConfig.ivf` (`nlist`, `nprobe`, `pq_m`).

## ANN-Benchmarks (B9)

```bash
# Synthetic (CI / no downloads)
cargo run -p dv-bench --release --bin topolsea-ann-bench -- --index hnsw --top-k 10

# With SIFT1M-style files in DATA_DIR:
#   sift_base.fvecs  sift_query.fvecs  sift_groundtruth.ivecs
cargo run -p dv-bench --release --bin topolsea-ann-bench -- \
  --data-dir "$DATA_DIR" --index hnsw --metric l2 --top-k 10
```

### Published numbers policy

| Claim | Allowed when |
|---|---|
| HNSW recall / QPS | Always — product default ANN |
| IVF(+PQ) memory/QPS | After measuring on the same dataset + equal-ish memory budget |
| Z-Column “beats HNSW” | **Only** if Track M gates G1∧G2∧G3 pass; otherwise publish as diagnostic / explain path |

Example synthetic run (machine-local; re-run to refresh):

```text
# cargo run -p dv-bench --release --bin topolsea-ann-bench -- --index hnsw
# → JSON: dataset, recall_at_k, qps, build_secs, notes
```

Commit refreshed JSON snippets under `docs/bench/` when publishing release numbers.
