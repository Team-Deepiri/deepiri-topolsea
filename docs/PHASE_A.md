# Phase A — Database product

Implements Phase A from the production roadmap (WAL, concurrency, REST service, filtered ANN, filter DSL).

Track M (Z-Column ANN gates) remains parallel and is **not** required for this cut. Default product ANN is HNSW.

## Acceptance map

| ID | Delivered |
|---|---|
| A1 | Append-only `wal.log` with CRC; upsert/delete durable before ack; `persist()` snapshots + truncates WAL; crash recovery via replay |
| A2 | `Collection::query` is `&self`; Z-Column access ledger queued off the exclusive path; `CollectionHandle = Arc<RwLock<Collection>>` for multi-reader + writer |
| A3 | `dv-server` / `topolsea-server`: `/health`, collections CRUD, upsert, search, explain; optional API key |
| A4 | Inverted index (roaring) + payload-constrained search; tiny eligible sets use exact scan |
| A5 | `$ne` / `$gt` / `$gte` / `$lt` / `$lte` / `$in` wired — see [`FILTER_DIALECT.md`](FILTER_DIALECT.md) |

## Run the server

```bash
cargo run -p dv-server --release -- --data-dir ./data --bind 127.0.0.1:6333
# optional: --api-key secret   or TOPOLSEA_API_KEY=secret
```

```bash
curl -s localhost:6333/health
curl -s -X POST localhost:6333/v1/collections -H 'content-type: application/json' \
  -d '{"name":"demo","dimension":4,"metric":"cosine","index":"hnsw"}'
```
