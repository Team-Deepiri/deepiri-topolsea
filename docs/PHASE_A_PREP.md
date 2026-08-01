# Phase A prep — become a real database (archived checklist)

> **Status (2026-08-01):** Phase A–C and Track M are **implemented** in stacked PRs `#13`–`#16`.  
> This file is an **archived crate-level checklist** kept for history.  
> **Current priorities:** [`docs/NEXT_STEPS.md`](NEXT_STEPS.md) → merge stack → **M5** re-measure → publish benches → [`docs/PHASE_D.md`](PHASE_D.md).

Prepared after applied-math measurement. Math track and DB track ran **in parallel**; public ANN-Benchmarks (B9) still wait on bounded-touch GO (**G1∧G2∧G3**).

---

## Archived — Track M checklist (all rows shipped in #16 except open M5 re-measure)

| ID | Work | Crate | Acceptance | Code status |
|---|---|---|---|---|
| M0 | Honest search knobs: no forced min-1 fallback; do not inflate beam `ef` with `coarse_pool` | `dv-index-zcolumn` | Pure beam τ≪1; ef/fallback knobs change measured τ | **Done** (#16) |
| M3 | Quantized coarse filter + **intra-column prune**; FP32 only in rerank | `dv-index-zcolumn`, `dv-metrics` | Tall-column visit with keep-m at τ&lt;0.5 path | **Done** (#16) |
| M4 | Compaction promote **moves** id; split/rebalance hot columns | `compact.rs` | Σ heights = N; max/mean shrinks under load | **Done** (#16) |
| M-graph | Neighbor graph over nonempty **column centroids** | `dv-index-zcolumn` | Cached graph + hop-limited walk | **Done** (#16) |
| M1 | Conditional fallback (heap &lt; k or score gap) | `dv-index-zcolumn` | `used_fallback_scan` only when fired | **Done** (#16) |
| M2 | Hard `V_touch` budget in explain + search stop | `dv-index-zcolumn` | `candidate_pool ≤ budget` | **Done** (#16) |
| M5 | Re-run `topolsea-math-localize` + `topolsea-math-probe` | `dv-bench` | **G1∧G2∧G3** on 10k; then 100k | **Harness ready** — re-measure still open |

Phase-2 result: **oracle whole-column expand cannot hit G1∧G3** (needs ~8 columns at τ≈0.68). M3+M4 remain critical; M-graph alone is not enough. See [`math/EXPERIMENT_RESULTS.md`](math/EXPERIMENT_RESULTS.md).

---

## Archived — Track A checklist (done in #13)

### A1. WAL + durable upsert / auto-flush — **done (#13)**
- `dv-storage` WAL (`wal.log` Upsert/Delete/Meta) + CRC  
- mutate → append WAL → ack; background snapshot; replay after snapshot seq  

### A2. Thread-safe collection — **done (#13)**
- Search on `&self`; `CollectionHandle = Arc<RwLock<Collection>>`  

### A3. Service API — **done (#13)**
- `dv-server` (axum): health, collections, upsert, search, explain; API key + TLS  

### A4. Payload-aware filtered ANN — **done (#13)**
- Roaring inverted index; constrain HNSW & Z-Column candidates  

### A5. Finish filter DSL — **done (#13)**
- `$ne` / `$gt` / `$gte` / `$lt` / `$lte` / `$in`; `docs/FILTER_DIALECT.md`  

---

## Archived — Phases B/C (done in #14 / #15)

Hybrid BM25, mmap segments, IVF/PQ, ann-bench harness; replication, shard harden, Prometheus, Helm, snapshots. See [`NEXT_STEPS.md`](NEXT_STEPS.md) and [`PHASE_D.md`](PHASE_D.md) for what remains toward peer-grade production.

## Non-goals (unchanged)

- Langlands / p-adic / ghost  
- GPU / learned predictor until M5 gates are understood  
- Marketing Z-Column as production ANN without G1∧G2∧G3  
