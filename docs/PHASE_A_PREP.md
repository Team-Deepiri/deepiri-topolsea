# Phase A prep — become a real database

Prepared after applied-math measurement. Math track and DB track run **in parallel**; public ANN-Benchmarks (item 4 / Phase B.9) wait on bounded-touch GO.

See the full vision and sequencing in [`docs/NEXT_STEPS.md`](NEXT_STEPS.md).

## Track M — Math / index (unblock honest benches)

| ID | Work | Crate | Acceptance |
|---|---|---|---|
| M1 | Conditional fallback (only if heap < k or score gap) | `dv-index-zcolumn` | `used_fallback_scan` true only when fired; revert rate measurable |
| M2 | Hard `V_touch` budget in explain + search stop | `dv-index-zcolumn` | `candidate_pool ≤ budget` |
| M3 | Beam uses quantized coarse scan; FP32 only in rerank | `dv-index-zcolumn`, `dv-metrics` | p50 drops; recall held within 2% of current at same budget |
| M4 | Compaction promote **moves** id (remove from source) | `compact.rs` | Σ heights = N invariant under rebalance |
| M5 | Re-run `topolsea-math-probe` + publish numbers | `dv-bench` | G1∧G2∧G3 on 10k; then 100k |

## Track A — Phase A database must-haves

### A1. WAL + durable upsert / auto-flush
- **New:** `dv-storage` WAL segment (`wal.log` append records: Upsert/Delete/Meta) + CRC
- **Flow:** mutate memory → append WAL → ack; background snapshot to `vectors.bin`/`index.bin`/`metadata.json` (atomic rename)
- **Recovery:** replay WAL after last snapshot seq
- **API:** `Collection::upsert` durable by default; `persist()` becomes snapshot trigger
- **Tests:** crash mid-upsert (kill after WAL write); recover equals pre-crash state

### A2. Thread-safe collection
- Split `ZColumnIndex` search to `&self` only (predictor already `RwLock`; access ledger → `Mutex`/`DashMap` or async queue)
- `Database`/`Collection` behind `Arc<RwLock<_>>` or sharded locks
- Stop requiring `&mut self` on `query` for ledger side effects
- **Tests:** N reader threads + 1 writer; loom or stress test

### A3. Service API (REST + optional gRPC)
- **New crate:** `dv-server` (axum): `/health`, `/v1/collections`, upsert, search, explain
- Auth: API key header; TLS via rustls in deploy config
- Replace toy `ShardQueryServer` raw TCP with shared axum app
- Python: thin HTTP client path alongside PyO3 embedded client
- **Tests:** HTTP integration smoke (create, upsert, search, health)

### A4. Payload-aware filtered ANN
- **New:** inverted index in `dv-metadata` (`field → value → Roaring bitmap of VectorId`)
- Search: compute eligible set first; constrain Z-Column/HNSW candidate generation (not `top_k*10` post-filter)
- HNSW: skip non-eligible neighbors; Z-Column: skip ids not in bitmap during `scan_column`
- **Tests:** selectivity 1%, 10%, 50% — recall vs filtered flat GT

### A5. Finish filter DSL
- Wire `FilterOp::{Ne,Gt,Gte,Lt,Lte,In}` into `Filter` AST + `from_json`
- Document JSON dialect (`$ne`, `$gt`, `$in`, …)
- **Tests:** unit matrix per op; integration with A4

## Track B/C (stubs only — after A)

See [`docs/NEXT_STEPS.md`](NEXT_STEPS.md): hybrid BM25, mmap segments, PQ/IVF, ANN-Benchmarks datasets, replication, shard hardening, Prometheus, Helm, snapshots.

## Suggested sequencing (2–3 sprints)

```
Week 1: M1–M3 + A5 (filters) + A2 sketch
Week 2: A1 WAL + A3 axum skeleton
Week 3: A4 filtered ANN + M4/M5 re-measure
Week 4: public ANN-Benchmarks only if G1∧G2∧G3 pass
```

## Non-goals this phase
- Langlands / p-adic / ghost
- GPU M5 / learned predictor M6 until observe signal fixed (M1)
