# What's Next — Deepiri Topolsea

Vision: turn Topolsea from a strong **embedded ANN engine** (Flat / HNSW / Z-Column + explain + fractal shards) into a **production-standard vector database** — durable, concurrent, filterable, operable, and honest about recall/latency tradeoffs — in the same class as self-hosted Qdrant / Milvus / Weaviate for the workloads that matter to Deepiri (RAG, metadata-filtered retrieval, multi-node shards).

Langlands / p-adic / ghost momentum in `DESIGN_PLAN.md` stay **research-only** until Phase A–B land. Novel index claims must pass measured gates before marketing.

---

## Where we are (baseline)

| Already real | Not yet production |
|---|---|
| HNSW + Flat + Z-Column indexes | WAL / crash-safe auto-durability |
| SIMD distances, U8/U16 column quant | Concurrent multi-client server |
| On-disk segments + index DR rebuild | Payload-aware filtered ANN (today: post-filter ×10) |
| Metadata eq / `$and` / `$or` | Full filter DSL (`ne`/`gt`/`in`/ranges named but unused) |
| Fractal sharding + toy HTTP fan-out | Replication, HA, warm shard processes |
| Python client + CLI + CI | Docker/Helm, Prometheus, auth/tenancy |
| Explain API (Z-Column) | Hybrid sparse+dense search |
| `topolsea-prove` / `topolsea-math-probe` | Public ANN-Benchmarks under **bounded** touch |

**Measured (2026-07-31, n=10k, 128d, cosine):** default Z-Column hits recall ≥ HNSW but touches ~100% of the corpus and is ~5× slower p50. High recall is bought by exhaustive-ish fallback, not sparse fractal walk. Details: [`docs/math/EXPERIMENT_RESULTS.md`](math/EXPERIMENT_RESULTS.md), math writeup: [`docs/math/Z_COLUMN_APPLIED_MATH.md`](math/Z_COLUMN_APPLIED_MATH.md).

**Implication:** run a **Math track (M)** in parallel with database Phase A. Do **not** publish “beats HNSW” ANN-Benchmarks until gates G1∧G2∧G3 pass under bounded candidate touch.

---

## North star — “as good as a production vector DB”

Done means an outsider can:

1. **Run it as a service** — REST (and ideally gRPC), health, API keys, TLS  
2. **Trust writes** — upsert/delete survive crash (WAL + snapshot); recovery is automatic  
3. **Serve many clients** — concurrent readers + writers; no `&mut` on the hot search path  
4. **Filter correctly** — selective metadata filters with recall close to filtered exact search (not overfetch-and-drop)  
5. **Retrieve for RAG** — dense ANN + hybrid (BM25/sparse) + rerank hooks  
6. **Scale out** — fractal shards that pass filters/metadata, with retries and warm processes; later replicas  
7. **Operate it** — metrics, traces, backups, Docker/Helm, multi-tenant namespaces  
8. **Prove it** — published recall/QPS/memory vs HNSW (and ANN-Benchmarks) with explicit touch budgets  

Milvus/Qdrant-level polish (coordinator HA, Woodpecker/Kafka WAL, DiskANN, GPU CAGRA) is Phase C+ aspiration — not required to call the first production cut “real.”

---

## Two tracks (do both)

```
Track M (index honesty)          Track A→C (database product)
───────────────────────          ───────────────────────────
M3 coarse quant / intra-col      A1 WAL + auto-flush
M4 height-balance + move-not-copy A2 thread-safe collection
M-graph centroid column graph    A3 axum/tonic service
M2 hard V_touch budget           A4 payload-aware filtered ANN
M1 conditional fallback          A5 finish filter DSL
M5 re-measure gates
        \                               /
         \                             /
          └── public ANN-Benchmarks ──┘
                    (only if G1∧G2∧G3)
```

### Go/no-go gates (Z-Column vs HNSW)

| Gate | Pass if |
|---|---|
| **G1** Recall | `recall_Z / recall_HNSW ≥ 0.98` @ k=10 |
| **G2** Latency | `p50_Z / p50_HNSW ≤ 1.5` |
| **G3** Touch | `candidates_touched / N < 0.5` (sublinear spirit) |

Protocol extras (revert rate, compaction recall) in [`docs/Z_COLUMN_PROTOCOL.md`](Z_COLUMN_PROTOCOL.md).

---

## Immediate build order (corrected)

Not “ship benches first.” Order:

1. **Track M — make Z-Column ANN real** — after Phase-2 localize/oracle: whole-column expand (even perfect column pick) cannot hit G1∧G3 because GT lives in ~6–8 **tall** columns (τ≈0.68). Next code is **M3 intra-column prune + M4 height-balance**, then **centroid-graph** to close the centroid→oracle gap, then hard `V_touch` + conditional fallback (see [`math/EXPERIMENT_RESULTS.md`](math/EXPERIMENT_RESULTS.md)). Until then default product ANN = **HNSW**; Z-Column = explain + shard keys.  
2. **WAL + auto-flush** — durable by default  
3. **Thread-safe collection + concurrent server** (REST; gRPC when REST is stable)  
4. **Real filtered search + complete filter ops**  
5. **Public ANN-Benchmarks** — only after G1∧G2∧G3 (or publish HNSW numbers and Z-Column as explain/shard story)

Crate-level prep: [`docs/PHASE_A_PREP.md`](PHASE_A_PREP.md).

---

## Phase A — Become a real database (must-have)

| # | Item | Outcome |
|---|---|---|
| A1 | **WAL + durable upsert** | Append-only log + CRC; periodic/background snapshot; crash recovery; `persist()` = snapshot, not the only durability path |
| A2 | **Thread-safe collection** | Shared read path; writers locked/segmented; search does not need `&mut` (ledger off hot path or lock-free) |
| A3 | **Service API** | `dv-server` (axum; tonic optional): collections CRUD, upsert, search, explain, health; API keys + TLS |
| A4 | **Payload-aware filtered ANN** | Inverted indexes / bitmaps on metadata; constrain HNSW & Z-Column candidates — **not** `top_k×10` post-filter |
| A5 | **Finish filter DSL** | Wire `ne` / `gt` / `gte` / `lt` / `lte` / `in` that `FilterOp` already names; document JSON dialect |

**Phase A exit:** single-node service you would run in staging for an internal RAG app with filtered search and crash-safe writes.

---

## Phase B — Match RAG / product expectations

| # | Item | Outcome |
|---|---|---|
| B6 | **Hybrid search** | Sparse/BM25 (or learned sparse) + dense fusion / RRF |
| B7 | **Segmented storage + mmap** | Sealed segments, incremental flush, no full-corpus rewrite on every snapshot; optional DiskANN-class cold path |
| B8 | **PQ / IVF** (or Faiss/usearch secondary) | Memory-bound larger-than-RAM / billion-scale path |
| B9 | **ANN-Benchmarks + published numbers** | Honest HNSW vs Z-Column recall/QPS/memory; equal-memory curves; only claim “production ANN” if gates pass |

**Phase B exit:** RAG-ready feature set + public, reproducible performance story.

---

## Phase C — Operate like Milvus/Qdrant

| # | Item | Outcome |
|---|---|---|
| C10 | **Replication + membership** | Primary-backup per shard first; Raft/consensus if multi-writer needed |
| C11 | **Harden fractal shards** | Filters + metadata on remote path; warm shard processes; retries / timeouts / circuit breakers |
| C12 | **Metrics + tracing** | Prometheus (QPS, p99, recall samples, WAL lag), OpenTelemetry |
| C13 | **Auth / multi-tenant namespaces** + Docker/Helm | Deployable multi-tenant cut |
| C14 | **Snapshots / backup API** | Operator-triggered backup/restore beyond raw file copy |

**Phase C exit:** multi-node operable deployment an SRE would keep alive.

---

## Suggested sequencing

```
Sprint 1  M1–M3 + A5 filter DSL + A2 concurrency sketch
Sprint 2  A1 WAL + A3 axum skeleton (health/collections/search)
Sprint 3  A4 filtered ANN + M4 compaction fix + M5 re-measure
Sprint 4  If G1∧G2∧G3: B9 ANN-Benchmarks; else keep HNSW default
Sprint 5+ B6–B8 RAG features, then C10–C14 ops
```

---

## What “good” looks like vs peers (cheat sheet)

| Capability | Peers do | Topolsea target phase |
|---|---|---|
| Durability (WAL) | Qdrant / Milvus | **A1** |
| Network API + auth | All | **A3**, **C13** |
| Filtered ANN | Payload indexes / prefilter | **A4–A5** |
| Hybrid dense+sparse | Milvus / Weaviate / Qdrant | **B6** |
| Horizontal scale | Shards + replicas | M4 shards today → **C10–C11** |
| Observability | Prometheus / OTel | **C12** |
| Packaging | Docker / Helm | **C13** |
| Differentiator | — | Z-Column **explain**, fractal **partition keys**, DR rebuild — keep; speed claims only after gates |

---

## Defer (research, not the production bar)

- Langlands / p-adic / ghost momentum (`DESIGN_PLAN.md`)  
- M5 GPU batch projection / quantized scan  
- M6 learned layer predictor — only after observe signal is honest (conditional fallback; stop always-setting `used_fallback_scan`)

---

## Related docs

| Doc | Role |
|---|---|
| [`docs/math/Z_COLUMN_APPLIED_MATH.md`](math/Z_COLUMN_APPLIED_MATH.md) | Discovery-mode math for shipping Z-Column |
| [`docs/math/EXPERIMENT_PLAN.md`](math/EXPERIMENT_PLAN.md) | Measure matrix |
| [`docs/math/EXPERIMENT_RESULTS.md`](math/EXPERIMENT_RESULTS.md) | Latest go/no-go numbers |
| [`docs/PHASE_A_PREP.md`](PHASE_A_PREP.md) | Crate-level acceptance for A + M |
| [`docs/Z_COLUMN_PROTOCOL.md`](Z_COLUMN_PROTOCOL.md) | Index protocol + original gates |
| [`IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md) | Older checklist (many items superseded by this file) |
