# Deepiri Topolsea — Implementation Plan

## Priority Queue

| Priority | Task | Phase | Status |
|---|---|---|---|
| P0 | Repository skeleton and CI | P0 | In Progress |
| P0 | Core type definitions | P0 | Planned |
| P0 | Native exact flat search | P0 | Planned |
| P0 | REST API skeleton | P0 | Planned |
| P1 | pgvector adapter | P1 | Planned |
| P1 | Qdrant adapter | P1 | Planned |
| P1 | Backend capability registry | P1 | Planned |
| P2 | Native embedded engine (in-memory) | P2 | Planned |
| P2 | Ghost parameter momentum (prototype) | P2 | Planned |
| P2 | File-backed persistence | P2 | Planned |
| P3 | Benchmark harness | P3 | Planned |
| P3 | Standard dataset support | P3 | Planned |
| P4 | Embedding version migration | P4 | Planned |
| P4 | Temporal transformation axis | P4 | Planned |
| P5 | ZepGPU batch integration | P5 | Planned |
| P6 | Python SDK | P6 | Planned |
| P6 | API auth and multi-tenancy | P6 | Planned |
| P7 | \(p\)-adic memory addressing prototype | P7 | Planned |
| P8 | Langlands search prototype | P8 | Research |
| P9 | Production hardening | P9 | Planned |

---

## Phase 0: Foundation (Week 1-2)

### 0.1 Core Types

- [ ] `Collection` — namespace, backend, metric, dimension, embedding version
- [ ] `Namespace` — tenant boundary, ownership
- [ ] `VectorRecord` — id, vectors dict, payload, embedding metadata
- [ ] `SearchRequest` — query vector, top-k, filters, backend hints
- [ ] `SearchResult` — scored matches with payloads
- [ ] `EmbeddingVersion` — model id, version, dimension, metric
- [ ] `BackendCapability` — support matrix for exact/ANN/filter/transactions
- [ ] `VectorMetric` — enum: Cosine, DotProduct, L2

### 0.2 Native Exact Search

- [ ] Exact flat cosine search
- [ ] Exact flat dot product search
- [ ] Exact flat L2 search
- [ ] Top-k bounded heap selection
- [ ] Zero-vector validation for cosine
- [ ] Non-finite vector rejection
- [ ] Dimension mismatch rejection

### 0.3 API Skeleton

- [ ] FastAPI application
- [ ] POST /collections
- [ ] GET /collections
- [ ] POST /collections/{id}/records:upsert
- [ ] POST /collections/{id}:search
- [ ] Health endpoint

### 0.4 Testing

- [ ] Unit tests for distance metrics
- [ ] Unit tests for top-k heap
- [ ] Unit tests for validation
- [ ] Integration test: create, insert, search, delete

---

## Phase 1: Backend Adapters (Week 3-6)

### 1.1 pgvector Adapter

- [ ] PostgreSQL connection pool
- [ ] Schema: tables with vector(768) columns, JSONB metadata
- [ ] SQLAlchemy models
- [ ] Exact search via `<->`, `<=>`, `<#>`
- [ ] Metadata filter pushdown (WHERE clauses)
- [ ] HNSW index creation (if pgvector version supports)
- [ ] IVFFlat index creation
- [ ] Conformance test suite

### 1.2 Qdrant Adapter

- [ ] Qdrant client connection
- [ ] Collection management
- [ ] Point upsert with payload
- [ ] Search with payload filters
- [ ] Score mapping (Qdrant scores -> Deepiri result format)
- [ ] Conformance test suite

### 1.3 Backend Registry

- [ ] `native` backend registration
- [ ] `pgvector` backend registration
- [ ] `qdrant` backend registration
- [ ] Capability query endpoint
- [ ] Health check per backend
- [ ] Validation for unsupported operations

---

## Phase 2: Native Embedded Engine (Week 7-12)

### 2.1 In-Memory Store

- [ ] Contiguous vector storage (NumPy array)
- [ ] ID-to-index mapping
- [ ] Batch insert/delete
- [ ] Exact filtered search path
- [ ] Memory-mapped file support

### 2.2 Ghost Parameter Prototype

- [ ] Forward/backward paired memory cell struct
- [ ] Symplectic momentum update rule
- [ ] Friction decay (\(-\gamma p_i\))
- [ ] Gradient observation (\(\nabla \mathcal{E}\))
- [ ] Self-stabilizing quantization (dynamic bit allocation)
- [ ] Interleaved cache-line layout

### 2.3 Persistence

- [ ] Collection manifest (JSON)
- [ ] Vector segment serialization
- [ ] Payload serialization (JSONB)
- [ ] Snapshot export/import
- [ ] Crash-safe write strategy

### 2.4 HNSW Research Track (Feature Flag)

- [ ] Define HNSW interface
- [ ] Evaluate hnswlib binding vs native implementation
- [ ] Benchmark recall/latency/memory
- [ ] Decision: embed or defer

---

## Phase 3: Benchmark Harness (Week 13-16)

### 3.1 Datasets

- [ ] Synthetic clustered (sklearn make_blobs)
- [ ] Synthetic random (uniform unit sphere)
- [ ] SIFT1M downloader
- [ ] GIST1M downloader
- [ ] Small CI dataset (10k vectors, 64d)

### 3.2 Metrics

- [ ] Recall@k (k=1,5,10,100)
- [ ] Latency p50/p95/p99
- [ ] Throughput (QPS)
- [ ] Index build time
- [ ] Memory usage (peak RSS)
- [ ] Disk usage
- [ ] Filter selectivity curve

### 3.3 Ghost-Specific Metrics

- [ ] Bit-width distribution (ghost allocation map)
- [ ] Ghost momentum magnitude over time
- [ ] Stability convergence rate
- [ ] Compression ratio vs recall curve

### 3.4 Reports

- [ ] JSON output
- [ ] Markdown report
- [ ] CSV export
- [ ] Comparison table generator

---

## Phase 4: Embedding Version Migration (Week 17-20)

### 4.1 Temporal Axis

- [ ] EmbeddingVersion CRUD
- [ ] Orthogonal Procrustes alignment between versions
- [ ] Version-to-version transformation matrix storage
- [ ] Read alias management
- [ ] Write alias management
- [ ] Rollback support

### 4.2 Migration Workflow

- [ ] Create new embedding version with new model
- [ ] Backfill vectors in background
- [ ] Compare old vs new retrieval (benchmark integration)
- [ ] Promote read alias
- [ ] Support rollback with old alias

---

## Phase 5: ZepGPU Integration (Week 21-24)

### 5.1 Job Types

- [ ] `vector.embed_batch` — bulk embedding via GPU
- [ ] `vector.search_batch_exact` — exact batch similarity
- [ ] `vector.rerank_batch` — GPU reranking
- [ ] `vector.train_index` — index training
- [ ] `vector.build_index` — index building
- [ ] `vector.benchmark_ground_truth` — ground-truth generation

### 5.2 Job Orchestration

- [ ] Job submission
- [ ] Status tracking
- [ ] S3 artifact storage
- [ ] Retry logic
- [ ] Cancel support
- [ ] Webhook for completion

---

## Phase 6: Service API + SDKs (Week 25-28)

### 6.1 API Server

- [ ] All collection routes
- [ ] All record routes
- [ ] All search routes (search, hybrid-search, rerank, explain)
- [ ] Embedding version routes
- [ ] Backend routes
- [ ] Benchmark routes
- [ ] Job routes
- [ ] Auth (JWT + API keys)
- [ ] Rate limiting

### 6.2 Python SDK

- [ ] `TopolseaClient` class
- [ ] Collection operations
- [ ] Record operations
- [ ] Search operations
- [ ] Benchmark operations
- [ ] Job operations
- [ ] Typed models

### 6.3 CLI

- [ ] `deepiri-topolsea create-collection`
- [ ] `deepiri-topolsea upsert`
- [ ] `deepiri-topolsea search`
- [ ] `deepiri-topolsea benchmark`
- [ ] `deepiri-topolsea migrate`

---

## Phase 7: \(p\)-adic Memory Prototype (Week 29-32)

### 7.1 Theory

- [ ] Implement \(p\)-adic integer arithmetic in Rust
- [ ] Prove ultrametric inequality for memory addressing
- [ ] Derive cache-miss-free allocation strategy
- [ ] Pair forward/ghost params in single \(p\)-adic coordinate

### 7.2 Prototype

- [ ] Simulate \(p\)-adic memory in software
- [ ] Benchmark cache miss rates vs standard layout
- [ ] Measure ghost materialization overhead

---

## Phase 8: Langlands Search (Research Track)

### 8.1 Theory

- [ ] Formalize dataset as Moduli Stack of \(G\)-bundles
- [ ] Define Hecke operator \(H_{\alpha}\) for vector search
- [ ] Prove \(\mathcal{O}(1)\) bound under reasonable assumptions
- [ ] Design approximation for non-algebraic embedding spaces

### 8.2 Prototype

- [ ] Toy implementation on synthetic 2d data
- [ ] Benchmark vs brute force
- [ ] Measure approximation error
- [ ] Go/no-go decision

---

## File Structure

```
deepiri-topolsea/
├── Cargo.toml                     # Rust workspace
├── pyproject.toml                 # Python package
├── DESIGN_PLAN.md                 # Mathematical architecture
├── IMPLEMENTATION_PLAN.md         # This file
├── README.md                      # Quickstart
├── LICENSE                        # Apache 2.0
├── .gitignore
├── .github/
│   ├── pull_request_template.md
│   ├── codeql/
│   │   ├── README.md
│   │   └── codeql-config.yml
│   └── workflows/
│       ├── codeql.yml
│       └── ci.yml
├── deepiri_topolsea/              # Python package
│   ├── __init__.py
│   ├── api/
│   │   ├── server.py
│   │   └── routes/
│   ├── backends/
│   │   ├── base.py
│   │   ├── capabilities.py
│   │   ├── native.py
│   │   ├── pgvector.py
│   │   └── qdrant.py
│   ├── catalog/
│   │   ├── models.py
│   │   └── repositories.py
│   ├── native/
│   │   ├── exact.py
│   │   ├── ghost.py
│   │   ├── storage.py
│   │   └── manifest.py
│   ├── planner/
│   │   ├── planner.py
│   │   └── filters.py
│   ├── jobs/
│   │   ├── zepgpu_client.py
│   │   └── embedding.py
│   ├── benchmarks/
│   │   ├── datasets.py
│   │   ├── harness.py
│   │   ├── metrics.py
│   │   └── reports.py
│   └── sdk/
│       ├── client.py
│       └── models.py
├── crates/                        # Rust crates
│   ├── dv-types/
│   ├── dv-metrics/
│   ├── dv-topk/
│   ├── dv-index-flat/
│   ├── dv-index-hnsw/
│   ├── dv-storage/
│   ├── dv-metadata/
│   ├── dv-query/
│   ├── dv-bench/
│   └── dv-bindings-python/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── conformance/
│   └── benchmarks/
├── examples/
│   ├── quickstart.py
│   ├── ghost_demo.py
│   └── migration.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs/
│   ├── ARCHITECTURE.md
│   ├── GHOST_PARAMS.md
│   ├── P_ADIC_MEMORY.md
│   └── LANGLANDS_SEARCH.md
├── scripts/
│   ├── run_benchmarks.py
│   └── seed_demo.py
└── k8s/
    └── deployment.yaml
```
