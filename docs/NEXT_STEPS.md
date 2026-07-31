# What's Next — Deepiri Topolsea

Priority order for turning the engine into a real vector database. Langlands / p-adic / ghost momentum stay research-only until Phase A–B land.

## Immediate build order

1. WAL + auto-flush
2. Concurrent server (REST/gRPC)
3. Real filtered search + complete filter ops
4. Public ANN-Benchmarks numbers for HNSW vs Z-Column

---

## Phase A — Become a real database (must-have)

1. **WAL + durable upsert** — append-only log, periodic snapshot, crash recovery; make persist automatic or background.
2. **Thread-safe collection** — shared read path; lock/segment writers; stop requiring `&mut` for search (move access ledger off critical path or make it lock-free).
3. **Proper service API** — axum/tonic (or FastAPI over Rust) with collections CRUD, upsert, search, health; API keys + TLS.
4. **Payload-aware filtered ANN** — inverted indexes on metadata fields; integrate into HNSW/Z-Column search (not overfetch-and-drop).
5. **Finish filter DSL** — wire `ne` / `gt` / `in` / ranges that `FilterOp` already names.

---

## Phase B — Match RAG / product expectations

6. **Hybrid search** — sparse/BM25 (or learned sparse) + dense fusion / RRF.
7. **Segmented storage + mmap** — sealed segments, incremental flush, avoid full rewrite; optional DiskANN-class cold path.
8. **PQ / IVF** (or adopt Faiss/usearch for secondary indexes) for memory-bound billion-scale.
9. **ANN-Benchmarks + published recall/QPS** — prove Z-Column go/no-go (≤2% of HNSW recall@10 at equal memory on 10k+/128d).

---

## Phase C — Operate like Milvus/Qdrant

10. **Replication + membership** — at least primary-backup per shard; then Raft/consensus if multi-writer.
11. **Harden fractal shards** — pass filters/metadata on remote path; keep warm shard processes; retries/timeouts/circuit breakers.
12. **Metrics + tracing** — Prometheus counters (QPS, p99, recall samples, WAL lag), OTel.
13. **Auth / multi-tenant namespaces** + Docker/Helm.
14. **Snapshots / backup API**.

---

## Defer (research, not the production bar)

- Langlands / p-adic / ghost momentum from `DESIGN_PLAN.md`
- M5 GPU / M6 learned predictor — valuable after Phase A–B
