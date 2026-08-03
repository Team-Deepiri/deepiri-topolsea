# What's Next — Deepiri Topolsea

Vision: turn Topolsea from a strong **embedded ANN engine** into a **production-standard vector database** — durable, concurrent, filterable, operable, and honest about recall/latency tradeoffs — in the same class as self-hosted Qdrant / Milvus / Weaviate for Deepiri workloads (RAG, metadata-filtered retrieval, multi-node shards).

Langlands / p-adic / ghost momentum in `DESIGN_PLAN.md` stay **research-only**. Novel index claims must pass measured gates before marketing.

---

## Status (2026-08-01)

Phased product work and Track M landed as **stacked PRs** (merge order **#13 → #14 → #15 → #16**, after or alongside this docs line):

| Track | PR | Status |
|---|---|---|
| Phase A — database product | [#13](https://github.com/Team-Deepiri/deepiri-topolsea/pull/13) | Implemented (`docs/PHASE_A.md` on branch) |
| Phase B — RAG / product | [#14](https://github.com/Team-Deepiri/deepiri-topolsea/pull/14) | Implemented (`docs/PHASE_B.md`) |
| Phase C — ops | [#15](https://github.com/Team-Deepiri/deepiri-topolsea/pull/15) | Implemented (`docs/PHASE_C.md`) |
| Track M — Z-Column honesty | [#16](https://github.com/Team-Deepiri/deepiri-topolsea/pull/16) | Implemented (`docs/PHASE_M.md`); **gates not yet proven** |

**Still true:** product ANN default = **HNSW**. Z-Column = explain + fractal shard keys until **G1∧G2∧G3** pass under bounded touch. Do not market “beats HNSW.”

### Where we are now

| Already real (A–C + M code) | Not yet production-complete |
|---|---|
| WAL + auto-flush, crash recovery | External / replicated WAL (Kafka, Woodpecker-class) |
| Thread-safe collections + axum REST, API keys, TLS | gRPC / stable multi-language SDKs at peer parity |
| Filter DSL + payload-constrained ANN | Filtered recall SLOs under real tenant schemas |
| Hybrid BM25 + dense (RRF / linear) | Learned sparse / cross-encoder rerank product path |
| Sealed mmap segments, IVF/PQ memory-bound path | DiskANN-class cold tier; billion-scale soak |
| ANN-bench harness + synthetic compare | Published public ANN-Benchmarks with gate-backed Z-Column story |
| Replica sync + membership + circuit failover | Multi-writer consensus / coordinator HA |
| Prometheus + request ids / traceparent | Full OTLP export + SLO dashboards |
| Namespaces, Docker/Helm, ServiceMonitor | Multi-region, backup encryption, GDPR delete propagation |
| Snapshots create/restore (scoped) | Continuous backup, PITR, restore drills |
| Track M: prune, graph, budgets, `GateReport` | **M5 re-measure: G1∧G2∧G3 on 10k then 100k** |

Math cliff (2026-07-31): whole-column expand of oracle’s ~6–8 tall columns hits τ≈0.68 — G1∧G3 impossible without **intra-column prune + height balance** (now in Track M code). Details: [`docs/math/EXPERIMENT_RESULTS.md`](math/EXPERIMENT_RESULTS.md).

---

## North star — “as good as a production vector DB”

An outsider should be able to:

1. **Run it as a service** — REST (and ideally gRPC), health/ready, API keys, TLS  
2. **Trust writes** — upsert/delete survive crash; recovery automatic  
3. **Serve many clients** — concurrent readers + writers on the hot path  
4. **Filter correctly** — selective metadata with recall close to filtered exact  
5. **Retrieve for RAG** — dense ANN + hybrid + rerank hooks  
6. **Scale out** — shards with filters/metadata, retries, replicas  
7. **Operate it** — metrics, traces, backups, Docker/Helm, multi-tenant namespaces  
8. **Prove it** — published recall/QPS/memory vs HNSW with explicit touch budgets  

**First production cut (A–C):** staging-ready for an internal RAG app — largely **done in code**, pending merge + soak.  
**Production-standard bar (Phase D+):** what peers expect for self-hosted Qdrant/Milvus-class ops and proof — [`docs/PHASE_D.md`](PHASE_D.md).

---

## Go / no-go gates (G1∧G2∧G3)

Canonical definitions (also used by `dv_bench::GateReport` / `GateInput.candidates_touched`):

| Gate | Pass if |
|---|---|
| **G1** Recall | `recall_Z / recall_HNSW ≥ 0.98` @ k=10 |
| **G2** Latency | `p50_Z / p50_HNSW ≤ 1.5` |
| **G3** Touch | `candidates_touched / N < 0.5` |

Protocol extras: [`docs/Z_COLUMN_PROTOCOL.md`](Z_COLUMN_PROTOCOL.md).

---

## Immediate next steps (ordered)

Not “more features before proof.” Order:

### 1. Land the stack

1. Merge Phase A → B → C → Track M (`#13` → `#14` → `#15` → `#16`) into `main` (or a release branch).  
2. Cut a staging image from that tip; run Docker Compose / Helm smoke (health, upsert, search, hybrid, snapshot, `/metrics`).

### 2. Prove honesty (Track M milestone **M5** — blocking for Z-Column marketing)

```bash
cargo run -p dv-bench --release --bin topolsea-math-probe -- --json
cargo run -p dv-bench --release --bin topolsea-math-localize -- --n=10000
# then N=100000 if 10k looks close
```

Use `dv_bench::GateReport` against the [gates above](#go--no-go-gates-g1g2g3). **If any gate fails:** keep HNSW as default; publish HNSW (and IVF) ANN-Benchmarks; Z-Column stays explain/shard story. **If all pass:** only then claim Z-Column as a production ANN option and publish equal-memory curves.

### 3. Publish the proof story (B9 completion)

- Run `topolsea-ann-bench` (and `--compare`) on a fixed synthetic + optional SIFT/GIST corpus.  
- Check in `docs/bench/` numbers with git SHA, hardware, and gate outcome.  
- Prefer **honest HNSW numbers now**; Z-Column numbers only with τ reported.

### 4. Phase D — production-standard hardening

Full acceptance map: [`docs/PHASE_D.md`](PHASE_D.md). This is the gap between “staging RAG DB” and “SRE would bet a tenant on it.”

---

## Phases A–C and Track M (delivered — reference)

Keep these as the historical acceptance maps; implementation docs live on the stacked branches until merged.

| Phase | Exit (original) | Code PRs |
|---|---|---|
| **A** | Staging single-node RAG DB | [#13](https://github.com/Team-Deepiri/deepiri-topolsea/pull/13) — WAL, concurrency, REST, filtered ANN, filter DSL |
| **B** | RAG features + bench harness | [#14](https://github.com/Team-Deepiri/deepiri-topolsea/pull/14) — hybrid, segments, IVF/PQ, ann-bench |
| **C** | Multi-node operable cut | [#15](https://github.com/Team-Deepiri/deepiri-topolsea/pull/15) — replicas, shard harden, metrics, tenants, snapshots |
| **M** | Honest Z-Column path + gate harness | [#16](https://github.com/Team-Deepiri/deepiri-topolsea/pull/16) — prune, graph, budgets, conditional fallback |

Prep checklist (historical): [`docs/PHASE_A_PREP.md`](PHASE_A_PREP.md).

---

## Suggested sequencing (from here)

```
Now       Merge #13→#16; staging smoke
Week 1    M5 re-measure 10k/100k; GateReport; decide Z-Column marketing
Week 1–2  Publish HNSW (± IVF) ANN-Bench numbers (B9); Z only if gates pass
Week 2–4  Phase D1 soak/chaos + D4 dashboards
Week 4–6  D2 gRPC/SDK + D3 replication SLAs
Week 6+   D5–D8 as tenant demand requires
```

---

## What “good” looks like vs peers (updated)

| Capability | Peers do | Topolsea now | Next |
|---|---|---|---|
| Durability (WAL) | Qdrant / Milvus | **A1 done** | External WAL / quorum (D3) |
| Network API + auth | All | REST + keys/TLS (**A3/C13**) | gRPC + SDK parity (**D2**) |
| Filtered ANN | Payload prefilter | **A4–A5 done** | Filtered SLO benches |
| Hybrid dense+sparse | Milvus / Weaviate / Qdrant | **B6 done** | Rerank productization |
| Horizontal scale | Shards + replicas | **C10–C11 done** | Quorum + coordinator HA (**D3/D8**) |
| Observability | Prometheus / OTel | Prom + request ids (**C12**) | OTLP + alerts (**D4**) |
| Packaging | Docker / Helm | **C13 done** | Hardened chart / multi-env |
| Proof | Public benches | Harness (**B9**); gates open (**M5**) | Publish numbers; Z only if G123 |
| Differentiator | — | Explain + fractal keys | Keep; speed claims only after gates |

---

## Defer (research, not the production bar)

- Langlands / p-adic / ghost momentum (`DESIGN_PLAN.md`)  
- GPU batch projection / CAGRA  
- M6 learned layer predictor — only after M5 observe signal is clean and gates are understood  
- Claiming Z-Column “production ANN” without G1∧G2∧G3  

---

## Related docs

| Doc | Role |
|---|---|
| [`docs/math/Z_COLUMN_APPLIED_MATH.md`](math/Z_COLUMN_APPLIED_MATH.md) | Discovery-mode math for shipping Z-Column |
| [`docs/math/EXPERIMENT_PLAN.md`](math/EXPERIMENT_PLAN.md) | Measure matrix |
| [`docs/math/EXPERIMENT_RESULTS.md`](math/EXPERIMENT_RESULTS.md) | Latest go/no-go numbers (pre–M5 re-measure) |
| [`docs/PHASE_A_PREP.md`](PHASE_A_PREP.md) | Original A + M crate checklist (archived / historical) |
| [`docs/PHASE_D.md`](PHASE_D.md) | Production-standard hardening map |
| [`docs/Z_COLUMN_PROTOCOL.md`](Z_COLUMN_PROTOCOL.md) | Index protocol + gates |
| Phase A/B/C/M docs | On stacked PRs `#13`–`#16` until merged to `main` |
| [`IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md) | Older checklist (many items superseded) |
