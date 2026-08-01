# Phase A — Database product

Implements Phase A from the production roadmap (WAL, concurrency, REST service, filtered ANN, filter DSL).

Track M (Z-Column ANN gates) remains parallel and is **not** required for this cut. Default product ANN is HNSW.

## Acceptance map

| ID | Delivered |
|---|---|
| A1 | Append-only `wal.log` with CRC; upsert/delete durable before ack; `persist()` snapshots + truncates WAL; crash recovery via replay; **background auto-flush** (`--flush-secs`, default 30s) |
| A2 | `Collection::query` is `&self`; Z-Column access ledger queued off the exclusive path; **Z-Column rebalance runs on `persist()` / auto-flush** (not per query — optional query-count rebalance remains a Track M knob); `CollectionHandle = Arc<RwLock<Collection>>` for multi-reader + writer |
| A3 | `dv-server` / `topolsea-server` + `topolsea serve`: `/health`, collections CRUD, upsert, search, explain, persist; API key; **TLS** via `--tls-cert`/`--tls-key` (rustls); shard fan-out on axum (`/topolsea/v1/shard/query`); Python `HttpClient` |
| A4 | Inverted index (roaring) + payload-constrained search; tiny eligible sets use exact scan; selectivity tests at 1% / 10% / 50% |
| A5 | `$ne` / `$gt` / `$gte` / `$lt` / `$lte` / `$in` wired — see [`FILTER_DIALECT.md`](FILTER_DIALECT.md) |

## Run the server

```bash
cargo run -p dv-server --release -- --data-dir ./data --bind 127.0.0.1:6333
# optional: --api-key secret   or TOPOLSEA_API_KEY=secret
# optional TLS: --tls-cert cert.pem --tls-key key.pem
# optional shard mode: --shard-collection physical_name
# auto-flush: --flush-secs 30  (0 to disable)
```

```bash
curl -s localhost:6333/health
curl -s -X POST localhost:6333/v1/collections -H 'content-type: application/json' \
  -d '{"name":"demo","dimension":4,"metric":"cosine","index":"hnsw"}'
```

Python HTTP client:

```python
from deepiri_topolsea import HttpClient
c = HttpClient("http://127.0.0.1:6333", api_key=None)
col = c.get_or_create_collection("demo", dimension=4)
col.upsert(["a"], [[1, 0, 0, 0]], [{"tag": "x"}])
print(col.query([1, 0, 0, 0], filter={"tag": "x"}))
```
