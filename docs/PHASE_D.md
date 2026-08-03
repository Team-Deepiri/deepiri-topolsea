# Phase D — Production-standard hardening

Exit criteria: an SRE would keep a **paying multi-tenant** deployment alive without tribal knowledge.

This phase follows merge of Phases A–C and Track M (PRs `#13`–`#16`) and **honest M5 gate re-measure**. See [`NEXT_STEPS.md`](NEXT_STEPS.md) for sequencing.

| ID | Item | Outcome |
|---|---|---|
| **D1** | **Soak + chaos** | 24h write/search soak; kill-9 mid-WAL; replica partition; restore drill from snapshot |
| **D2** | **gRPC + SDK parity** | tonic search/upsert; Python/HTTP clients cover hybrid, filters, snapshots, namespaces |
| **D3** | **Stronger replication** | Quorum / configurable sync ack SLAs; delete/upsert lag metrics; membership heartbeats drive routing |
| **D4** | **Observability product** | OTLP export; RED/USE dashboards; alert examples (WAL lag, replica fail, p99) |
| **D5** | **Security / tenancy** | Per-tenant quotas; audit log of admin ops; secret rotation docs; optional mTLS between shards |
| **D6** | **Backup / compliance** | Encrypted snapshot export; retention policy; delete propagates to snapshots (document GDPR path) |
| **D7** | **Scale path** | DiskANN or mmap-IVF cold path; documented memory envelope for 10M–100M vectors |
| **D8** | **Coordinator HA** (optional) | Stateless query coordinators behind LB; or Raft for membership if multi-writer required |

Milvus/Qdrant-level extras still deferred until D1–D4 are green: Woodpecker/Kafka WAL, GPU CAGRA, full multi-region.

**Gates reminder (Z-Column marketing):** only after **G1∧G2∧G3** — see the single gate table in [`NEXT_STEPS.md`](NEXT_STEPS.md#go--no-go-gates-g1g2g3).
