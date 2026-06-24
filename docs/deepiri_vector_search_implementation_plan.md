# Deepiri Vector Search - Implementation Plan



# Project Vision

Deepiri Vector Search should become an open-source vector runtime for portable retrieval backends, reproducible benchmarks, embedding-version migrations, and GPU-scheduled vector workloads.

The project should **not** start as a full distributed vector database. Instead, it should begin as a practical, backend-neutral runtime that lets developers use mature vector systems while Deepiri gradually builds small, measurable native capabilities.

The first version should provide one API over three execution modes:

1. **External durable backends**
   - pgvector
   - Qdrant

2. **Native embedded backend**
   - exact flat search first
   - HNSW only after a benchmark harness exists
   - local/dev use cases first

3. **ZepGPU jobs**
   - bulk embedding
   - exact batch search
   - index training/building
   - reranking
   - benchmark ground-truth generation

---

# Recommended Product Direction

## Decision: Prototype First

Deepiri should build:

- adapter-first vector runtime
- pgvector adapter
- Qdrant adapter
- native exact embedded engine
- benchmark harness
- embedding-version migration tools
- ZepGPU batch integration

Deepiri should **not** build yet:

- full distributed consensus
- multi-region replication
- billion-vector disk-native engine
- full text search engine
- hosted control plane
- every ANN algorithm
- direct Pinecone/Milvus/Qdrant clone

## Why

### Verified Fact

Mature vector databases are more than ANN indexes. Production systems include durability, metadata filtering, snapshots, replication, sharding, consistency, operations, and deployment tooling.

### Reasoned Inference

Deepiri can build useful vector execution and migration tooling faster than it can reproduce the operational maturity of systems like Qdrant, Milvus, Pinecone, Weaviate, Elasticsearch, or PostgreSQL.

### Deepiri Proposal

Deepiri’s differentiation should be:

- backend portability
- benchmark-driven backend/index selection
- embedding-version migration
- local-to-service developer flow
- ZepGPU-based batch embedding, indexing, and reranking
- transparent, reproducible retrieval evaluation

---

# Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Runtime Core | Rust | Native engine, metrics, indexing, storage-safe core |
| API Server | Python FastAPI or Rust Axum | REST/gRPC API layer |
| SDK - Python | Python 3.11+ | User-facing ingestion/search workflows |
| SDK - TypeScript | TypeScript | Web/app integrations |
| Native Engine | Rust | Exact flat search, later HNSW |
| External Backend 1 | PostgreSQL + pgvector | Durable SQL-backed vector storage |
| External Backend 2 | Qdrant | Dedicated vector DB backend |
| Batch Compute | ZepGPU | Bulk embedding, exact batch search, reranking |
| Queue | Redis + Celery or ZepGPU queue layer | Async batch jobs |
| Metadata Store | PostgreSQL | Catalog, collections, embedding versions, jobs |
| Object Storage | S3/MinIO | Snapshots, benchmark artifacts, index artifacts |
| API Auth | JWT / API keys | Tenant and namespace access |
| Observability | Prometheus + Grafana | Metrics, latency, recall, throughput |
| Containers | Docker | Local/runtime packaging |
| Deployment | Docker Compose, Kubernetes later | Local and production deployment |
| Benchmarking | Python + Rust harness | Recall, latency, throughput, memory |

---

# Architecture Overview

```mermaid
flowchart TB
    subgraph Clients
        PY[Python SDK]
        TS[TypeScript SDK]
        CLI[CLI]
    end

    PY --> API
    TS --> API
    CLI --> API

    API[Deepiri Vector API] --> Auth
    Auth --> Catalog
    Auth --> Planner

    Catalog --> MetadataDB[(PostgreSQL Catalog)]

    Planner --> Capability[Backend Capability Registry]

    Capability --> PG[pgvector Adapter]
    Capability --> QD[Qdrant Adapter]
    Capability --> Native[Native Embedded Engine]

    Native --> Flat[Exact Flat Search]
    Native --> HNSW[Future HNSW]
    Native --> MetaIndex[Metadata Index]
    Native --> NativeStore[Native Segment Store]

    Planner --> Jobs[Job Orchestrator]
    Jobs --> ZepGPU[ZepGPU Batch Workers]

    ZepGPU --> Embed[Bulk Embedding]
    ZepGPU --> BatchSearch[Exact Batch Search]
    ZepGPU --> Train[Index Training]
    ZepGPU --> Rerank[Reranking]
    ZepGPU --> Bench[Benchmark Ground Truth]

    API --> Metrics[Metrics / Traces / Audit]
```

---

# Core Components

## 1. Catalog

Tracks collections, namespaces, backend configuration, embedding versions, aliases, jobs, and index metadata.

Responsibilities:

- collection creation
- namespace ownership
- embedding version metadata
- backend capability records
- read/write alias management
- migration status
- benchmark run metadata

---

## 2. Backend Capability Registry

Defines what each backend supports.

Example capabilities:

```json
{
  "backend": "pgvector",
  "supports_exact": true,
  "supports_hnsw": true,
  "supports_ivfflat": true,
  "supports_filter_pushdown": true,
  "supports_sparse": false,
  "supports_transactions": true,
  "supports_snapshots": true,
  "supports_multi_vector": false
}
```

---

## 3. Planner

Chooses execution strategy for each query.

Planner inputs:

- collection size
- metric
- top-k
- namespace
- metadata filter
- backend capabilities
- embedding version
- requested consistency
- latency target
- recall target

Planner outputs:

- backend route
- exact vs approximate mode
- filter strategy
- rerank strategy
- fallback behavior

---

## 4. pgvector Adapter

Uses PostgreSQL as the durable database backend.

Responsibilities:

- create vector tables/indexes
- insert/upsert/delete records
- execute exact search
- execute HNSW/IVFFlat search when configured
- push down SQL metadata filters
- preserve relational durability semantics

---

## 5. Qdrant Adapter

Uses Qdrant as a dedicated vector backend.

Responsibilities:

- create collections
- manage points and payloads
- perform vector search
- push down payload filters
- support snapshots where available
- expose Qdrant-specific capabilities through the backend registry

---

## 6. Native Embedded Engine

Small local engine for development, benchmarks, and constrained use cases.

MVP scope:

- exact flat search
- cosine / dot product / L2
- simple metadata filters
- JSONL or segment-file storage
- no distributed replication
- no multi-region support
- no disk-native billion-scale ANN

Later scope:

- HNSW
- immutable segments
- WAL
- snapshots
- compaction
- scalar quantization

---

## 7. Benchmark Harness

Measures recall, latency, throughput, memory, indexing time, and filter behavior.

Required benchmark modes:

- exact ground truth
- pgvector exact
- pgvector HNSW
- Qdrant search
- native exact
- future native HNSW
- ZepGPU batch exact search
- reranking pipeline

---

## 8. ZepGPU Integration

Uses ZepGPU for batch-oriented work where GPU scheduling may help.

Workloads:

- bulk embedding
- exact batch similarity
- index training
- reranking
- benchmark ground-truth generation
- migration backfill

Important rule:

Deepiri should only claim GPU advantage after measuring total end-to-end benefit, including transfer and queue overhead.

---

# Implementation Phases

---

# Phase 1: MVP Runtime Foundation 🔄 PLANNED

## Goal

Create the initial Deepiri Vector runtime with collection metadata, basic ingestion, exact search, and a backend-neutral API.

## 1.1 Project Setup

- [ ] Create repository `deepiri-vector` or `deepiri-vector-search`
- [ ] Add README with project scope and non-goals
- [ ] Add license and contribution guide
- [ ] Add Python package structure
- [ ] Add optional Rust workspace structure
- [ ] Add Docker Compose skeleton
- [ ] Add CI with lint, type check, tests, and build
- [ ] Add example `.env`

## 1.2 Core Types

- [ ] Define `Collection`
- [ ] Define `Namespace`
- [ ] Define `VectorRecord`
- [ ] Define `EmbeddingVersion`
- [ ] Define `VectorMetric`
- [ ] Define `SearchRequest`
- [ ] Define `SearchResult`
- [ ] Define `BackendCapability`
- [ ] Define `IndexConfig`

## 1.3 Basic API

- [ ] Create collection
- [ ] List collections
- [ ] Get collection
- [ ] Delete collection
- [ ] Upsert records
- [ ] Delete records
- [ ] Search records
- [ ] Get record by ID

## 1.4 Native Exact Search Baseline

- [ ] Implement exact flat cosine search
- [ ] Implement exact flat dot product search
- [ ] Implement exact flat L2 search
- [ ] Add top-k heap selection
- [ ] Add zero-vector validation for cosine
- [ ] Add non-finite vector validation
- [ ] Add dimension validation

## Acceptance Criteria

- [ ] Can create a collection
- [ ] Can insert vectors
- [ ] Can search with exact top-k
- [ ] Supports cosine, dot product, and L2
- [ ] Rejects invalid vector dimensions
- [ ] Rejects invalid metric/vector combinations
- [ ] Has unit tests for exact ranking
- [ ] Has a working local quickstart

---

# Phase 2: Adapter Backend Support 🔄 PLANNED

## Goal

Add durable external backends so Deepiri can provide useful production behavior without building a full database from scratch.

## 2.1 pgvector Adapter

- [ ] Connect to PostgreSQL
- [ ] Create collection table
- [ ] Store vectors using pgvector
- [ ] Store payload metadata as JSONB
- [ ] Support exact search
- [ ] Support metadata filter pushdown
- [ ] Support HNSW index creation if available
- [ ] Support IVFFlat index creation if available
- [ ] Add adapter conformance tests

## 2.2 Qdrant Adapter

- [ ] Connect to Qdrant
- [ ] Create Qdrant collection
- [ ] Upsert points
- [ ] Delete points
- [ ] Execute vector search
- [ ] Push down payload filters
- [ ] Map Qdrant scores to Deepiri result format
- [ ] Add adapter conformance tests

## 2.3 Backend Registry

- [ ] Register `native`
- [ ] Register `pgvector`
- [ ] Register `qdrant`
- [ ] Expose backend capabilities
- [ ] Add validation for unsupported operations
- [ ] Add backend health checks

## Acceptance Criteria

- [ ] Same search API works across native, pgvector, and Qdrant
- [ ] Adapter conformance suite passes for all supported backends
- [ ] Backend capability endpoint works
- [ ] Unsupported operations return clear errors
- [ ] pgvector backend can run with Docker Compose
- [ ] Qdrant backend can run with Docker Compose

---

# Phase 3: Native Embedded Engine 🔄 PLANNED

## Goal

Build a small native embedded engine that is useful for local development, exact search, benchmarks, and controlled workloads.

## 3.1 Native Storage

- [ ] Add in-memory vector store
- [ ] Add file-backed persistence
- [ ] Add collection manifest
- [ ] Add vector record serialization
- [ ] Add payload serialization
- [ ] Add snapshot export/import
- [ ] Add crash-safe write strategy for MVP

## 3.2 Exact Search Optimization

- [ ] Store vectors contiguously
- [ ] Add batch query support
- [ ] Add SIMD-friendly distance functions where possible
- [ ] Add bounded heap top-k
- [ ] Add exact filtered search path
- [ ] Add benchmark comparison against NumPy/FAISS baseline

## 3.3 HNSW Research Track

HNSW should not be included until the benchmark harness exists.

- [ ] Define HNSW interface
- [ ] Compare build vs embedding hnswlib/USearch
- [ ] Add prototype only behind feature flag
- [ ] Benchmark recall/latency/memory
- [ ] Decide whether native HNSW is worth owning

## Acceptance Criteria

- [ ] Native exact engine passes correctness tests
- [ ] Native exact engine has reproducible benchmark results
- [ ] Persistence survives restart
- [ ] HNSW is either deferred or gated behind measured evidence
- [ ] Native engine is clearly documented as non-distributed

---

# Phase 4: Benchmark Harness 🔄 PLANNED

## Goal

Create a reproducible benchmark system that guides backend and index decisions instead of relying on assumptions.

## 4.1 Datasets

- [ ] Synthetic clustered dataset
- [ ] Synthetic random dataset
- [ ] Synthetic adversarial/high-dimensional dataset
- [ ] Real embedding dataset option
- [ ] Small local dataset for CI
- [ ] Larger optional dataset for manual runs

## 4.2 Metrics

- [ ] Recall@k
- [ ] Latency p50/p95/p99
- [ ] Throughput
- [ ] Index build time
- [ ] Memory usage
- [ ] Disk usage
- [ ] Filter selectivity behavior
- [ ] Ingestion throughput
- [ ] Query cost estimate

## 4.3 Benchmark Modes

- [ ] Native exact
- [ ] pgvector exact
- [ ] pgvector HNSW
- [ ] Qdrant
- [ ] FAISS baseline
- [ ] ZepGPU batch exact search
- [ ] Rerank pipeline

## 4.4 Reports

- [ ] JSON benchmark output
- [ ] Markdown report output
- [ ] CSV output
- [ ] Comparison table
- [ ] Recommended backend/index config

## Acceptance Criteria

- [ ] Benchmark can be reproduced from CLI
- [ ] Benchmark output includes version and config metadata
- [ ] Ground truth is exact and auditable
- [ ] Backend comparisons use same dataset/query set
- [ ] Results can guide go/no-go decisions

---

# Phase 5: ZepGPU Integration 🔄 PLANNED

## Goal

Connect Deepiri Vector to ZepGPU for batch embedding, index training, exact batch search, and reranking.

## 5.1 ZepGPU Job Types

- [ ] `vector.embed_batch`
- [ ] `vector.search_batch_exact`
- [ ] `vector.rerank_batch`
- [ ] `vector.train_index`
- [ ] `vector.build_index`
- [ ] `vector.benchmark_ground_truth`

## 5.2 Job Orchestration

- [ ] Submit batch jobs to ZepGPU
- [ ] Track job status
- [ ] Store job artifacts in S3/MinIO
- [ ] Retry failed jobs
- [ ] Cancel jobs
- [ ] Stream job logs
- [ ] Link job output back to collection/index version

## 5.3 Embedding Backfill

- [ ] Create new embedding version
- [ ] Schedule batch embedding
- [ ] Store new vectors separately
- [ ] Compare old vs new retrieval
- [ ] Promote read alias
- [ ] Support rollback

## Acceptance Criteria

- [ ] ZepGPU can run at least one vector batch workflow
- [ ] Batch job output is stored and linked to collection metadata
- [ ] Queue overhead is measured
- [ ] Transfer overhead is measured
- [ ] GPU benefit is reported honestly
- [ ] No GPU performance claims without benchmark evidence

---

# Phase 6: Service/API Layer 🔄 PLANNED

## Goal

Expose Deepiri Vector through a stable REST/gRPC API and SDKs.

## 6.1 API Server

- [ ] FastAPI or Axum service
- [ ] Collection routes
- [ ] Record routes
- [ ] Search routes
- [ ] Backend routes
- [ ] Benchmark routes
- [ ] Migration routes
- [ ] Job routes
- [ ] Health routes
- [ ] Metrics routes

## 6.2 SDKs

- [ ] Python SDK
- [ ] TypeScript SDK
- [ ] CLI client
- [ ] Typed request/response models
- [ ] Examples for RAG ingestion
- [ ] Examples for migration workflow
- [ ] Examples for benchmark workflow

## 6.3 Auth and Multi-Tenancy

- [ ] API key authentication
- [ ] JWT support
- [ ] Namespace isolation
- [ ] Tenant-scoped collections
- [ ] Audit logging
- [ ] Basic rate limits

## Acceptance Criteria

- [ ] API docs generate correctly
- [ ] Python SDK can ingest/search
- [ ] TypeScript SDK can search
- [ ] CLI supports create/upsert/search/benchmark
- [ ] Namespaces isolate data
- [ ] Auth errors are clear

---

# Phase 7: Persistence and Storage 🔄 PLANNED

## Goal

Add safe native persistence for small-to-medium local use without pretending to be a full distributed database.

## 7.1 Catalog Persistence

- [ ] PostgreSQL catalog tables
- [ ] Collection metadata
- [ ] Backend config
- [ ] Index metadata
- [ ] Embedding versions
- [ ] Jobs
- [ ] Benchmark runs

## 7.2 Native Storage

- [ ] Manifest file
- [ ] Immutable vector segments
- [ ] Payload segments
- [ ] Basic WAL
- [ ] Snapshot export/import
- [ ] Compaction plan
- [ ] Checksums

## 7.3 Recovery

- [ ] Restart recovery test
- [ ] Interrupted write test
- [ ] Snapshot restore test
- [ ] Manifest compatibility test
- [ ] Versioned storage format

## Acceptance Criteria

- [ ] Native engine survives restart
- [ ] Native records can be restored from snapshot
- [ ] Corrupted segments are detected
- [ ] Storage format is versioned
- [ ] WAL/manifest design is documented

---

# Phase 8: Metadata Filtering and Hybrid Search 🔄 PLANNED

## Goal

Support real retrieval workloads where vector search is combined with metadata filters, sparse search, and reranking.

## 8.1 Metadata Filtering

- [ ] Equality filters
- [ ] Range filters
- [ ] Boolean filters
- [ ] Array contains filters
- [ ] Namespace filter
- [ ] Embedding version filter
- [ ] Filter validation
- [ ] Filter-to-backend translation

## 8.2 Filter-Aware Planning

- [ ] Pre-filter allowed ID set
- [ ] Post-filter fallback
- [ ] Exact fallback for selective filters
- [ ] Configurable flat-search cutoff
- [ ] Filter selectivity estimates

## 8.3 Hybrid Search

- [ ] Sparse backend integration
- [ ] BM25 adapter or search-engine adapter
- [ ] Reciprocal Rank Fusion
- [ ] Dense candidate generation
- [ ] Sparse candidate generation
- [ ] Rerank stage

## Acceptance Criteria

- [ ] Filtered search returns stable top-k when possible
- [ ] Exact fallback works for small filtered sets
- [ ] RRF hybrid search works
- [ ] Hybrid search has benchmark coverage
- [ ] Backend filter limitations are documented

---

# Phase 9: Production Hardening 🔄 PLANNED

## Goal

Make the system reliable enough for early design partners and internal Deepiri usage.

## 9.1 Observability

- [ ] Query latency metrics
- [ ] Index build metrics
- [ ] Backend health metrics
- [ ] ZepGPU job metrics
- [ ] Recall benchmark metrics
- [ ] Error metrics
- [ ] Audit logs

## 9.2 Security

- [ ] API key auth
- [ ] Tenant isolation tests
- [ ] Payload size limits
- [ ] Input validation
- [ ] Secret management
- [ ] Deletion semantics
- [ ] Snapshot access control

## 9.3 Reliability

- [ ] Retry policies
- [ ] Idempotent upsert
- [ ] Graceful backend failure handling
- [ ] Backup/restore guide
- [ ] Migration rollback
- [ ] Versioned API compatibility

## 9.4 Deployment

- [ ] Docker Compose
- [ ] Kubernetes manifests
- [ ] Health checks
- [ ] Readiness probes
- [ ] Config docs
- [ ] Example production deployment

## Acceptance Criteria

- [ ] End-to-end integration tests pass
- [ ] Deployment works locally through Docker Compose
- [ ] Basic Kubernetes deployment works
- [ ] Auth and namespace isolation tests pass
- [ ] Backup/restore path is documented
- [ ] Design partner checklist is ready

---

# Database Schema Design

## Catalog Tables

```sql
CREATE TABLE vector_namespaces (
    id UUID PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    owner_id UUID,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

```sql
CREATE TABLE vector_collections (
    id UUID PRIMARY KEY,
    namespace_id UUID REFERENCES vector_namespaces(id),
    name VARCHAR(255) NOT NULL,
    backend VARCHAR(50) NOT NULL,
    metric VARCHAR(50) NOT NULL,
    dimension INTEGER NOT NULL,
    active_embedding_version_id UUID,
    metadata_json JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(namespace_id, name)
);
```

```sql
CREATE TABLE embedding_versions (
    id UUID PRIMARY KEY,
    collection_id UUID REFERENCES vector_collections(id),
    model_id VARCHAR(255) NOT NULL,
    model_version VARCHAR(255),
    dimension INTEGER NOT NULL,
    metric VARCHAR(50) NOT NULL,
    normalized BOOLEAN DEFAULT false,
    status VARCHAR(50) DEFAULT 'building',
    created_at TIMESTAMP DEFAULT NOW(),
    promoted_at TIMESTAMP
);
```

```sql
CREATE TABLE vector_backend_configs (
    id UUID PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    backend_type VARCHAR(50) NOT NULL,
    connection_uri TEXT,
    capability_json JSONB DEFAULT '{}',
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT NOW()
);
```

```sql
CREATE TABLE vector_index_versions (
    id UUID PRIMARY KEY,
    collection_id UUID REFERENCES vector_collections(id),
    embedding_version_id UUID REFERENCES embedding_versions(id),
    backend VARCHAR(50) NOT NULL,
    index_type VARCHAR(50) NOT NULL,
    index_config JSONB DEFAULT '{}',
    artifact_uri TEXT,
    status VARCHAR(50) DEFAULT 'building',
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);
```

```sql
CREATE TABLE vector_jobs (
    id UUID PRIMARY KEY,
    collection_id UUID REFERENCES vector_collections(id),
    job_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    zepgpu_task_id UUID,
    input_uri TEXT,
    output_uri TEXT,
    error TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);
```

```sql
CREATE TABLE vector_benchmark_runs (
    id UUID PRIMARY KEY,
    collection_id UUID REFERENCES vector_collections(id),
    benchmark_name VARCHAR(255) NOT NULL,
    backend VARCHAR(50),
    index_type VARCHAR(50),
    config_json JSONB DEFAULT '{}',
    results_json JSONB DEFAULT '{}',
    artifact_uri TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

# Vector Record Model

```json
{
  "id": "doc-123#chunk-7",
  "namespace": "tenant-a",
  "collection": "support-docs",
  "vectors": {
    "text_v3": [0.021, -0.311, 0.084]
  },
  "payload": {
    "document_id": "doc-123",
    "language": "en",
    "created_at": "2026-06-17T00:00:00Z"
  },
  "embedding": {
    "model_id": "encoder-x",
    "model_version": "3",
    "dimension": 768,
    "normalized": true
  }
}
```

---

# API Endpoints

## Collections

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/vector/collections` | Create collection |
| `GET` | `/api/v1/vector/collections` | List collections |
| `GET` | `/api/v1/vector/collections/{collection_id}` | Get collection |
| `DELETE` | `/api/v1/vector/collections/{collection_id}` | Delete/archive collection |
| `POST` | `/api/v1/vector/collections/{collection_id}/aliases` | Update read/write aliases |

## Records

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/vector/collections/{collection_id}/records:upsert` | Upsert records |
| `POST` | `/api/v1/vector/collections/{collection_id}/records:delete` | Delete records |
| `GET` | `/api/v1/vector/collections/{collection_id}/records/{record_id}` | Get record |
| `POST` | `/api/v1/vector/collections/{collection_id}/records:batch` | Batch get records |

## Search

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/vector/collections/{collection_id}:search` | Search vectors |
| `POST` | `/api/v1/vector/collections/{collection_id}:hybrid-search` | Sparse+dense hybrid search |
| `POST` | `/api/v1/vector/collections/{collection_id}:rerank` | Rerank candidates |
| `POST` | `/api/v1/vector/collections/{collection_id}:explain` | Explain query plan |

## Embedding Versions

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/vector/collections/{collection_id}/embedding-versions` | Create embedding version |
| `GET` | `/api/v1/vector/collections/{collection_id}/embedding-versions` | List embedding versions |
| `POST` | `/api/v1/vector/embedding-versions/{version_id}:promote` | Promote embedding version |
| `POST` | `/api/v1/vector/embedding-versions/{version_id}:rollback` | Roll back version alias |

## Backends

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/vector/backends` | List configured backends |
| `GET` | `/api/v1/vector/backends/{name}/capabilities` | Get backend capabilities |
| `GET` | `/api/v1/vector/backends/{name}/health` | Backend health check |

## Benchmarks

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/vector/benchmarks` | Start benchmark run |
| `GET` | `/api/v1/vector/benchmarks/{id}` | Get benchmark status |
| `GET` | `/api/v1/vector/benchmarks/{id}/results` | Get benchmark results |
| `GET` | `/api/v1/vector/benchmarks` | List benchmark runs |

## Jobs

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/vector/jobs/embed-batch` | Start ZepGPU embedding batch |
| `POST` | `/api/v1/vector/jobs/build-index` | Start index build job |
| `POST` | `/api/v1/vector/jobs/rerank` | Start rerank job |
| `GET` | `/api/v1/vector/jobs/{id}` | Get job status |
| `DELETE` | `/api/v1/vector/jobs/{id}` | Cancel job |

## System

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/vector/health` | Health check |
| `GET` | `/api/v1/vector/stats` | System stats |
| `GET` | `/api/v1/vector/metrics` | Prometheus metrics |

---

# File Structure

```text
deepiri-vector/
├── README.md
├── docs/
│   ├── ARCHITECTURE.md
│   ├── BENCHMARKING.md
│   ├── BACKENDS.md
│   ├── MIGRATIONS.md
│   └── ZEPGPU_INTEGRATION.md
├── pyproject.toml
├── Cargo.toml
├── crates/
│   ├── dv-types/
│   ├── dv-metrics/
│   ├── dv-topk/
│   ├── dv-index-api/
│   ├── dv-index-flat/
│   ├── dv-index-hnsw/
│   ├── dv-storage/
│   ├── dv-metadata/
│   ├── dv-query/
│   ├── dv-bench/
│   └── dv-bindings-python/
├── deepiri_vector/
│   ├── __init__.py
│   ├── config.py
│   ├── api/
│   │   ├── server.py
│   │   ├── dependencies.py
│   │   └── routes/
│   │       ├── collections.py
│   │       ├── records.py
│   │       ├── search.py
│   │       ├── backends.py
│   │       ├── benchmarks.py
│   │       ├── jobs.py
│   │       └── health.py
│   ├── catalog/
│   │   ├── models.py
│   │   ├── repositories.py
│   │   └── migrations.py
│   ├── backends/
│   │   ├── base.py
│   │   ├── capabilities.py
│   │   ├── native.py
│   │   ├── pgvector.py
│   │   └── qdrant.py
│   ├── native/
│   │   ├── exact.py
│   │   ├── metrics.py
│   │   ├── storage.py
│   │   ├── segment.py
│   │   └── manifest.py
│   ├── planner/
│   │   ├── planner.py
│   │   ├── filters.py
│   │   └── explain.py
│   ├── jobs/
│   │   ├── zepgpu_client.py
│   │   ├── embedding.py
│   │   ├── indexing.py
│   │   └── reranking.py
│   ├── benchmarks/
│   │   ├── datasets.py
│   │   ├── harness.py
│   │   ├── metrics.py
│   │   └── reports.py
│   ├── sdk/
│   │   ├── client.py
│   │   └── models.py
│   └── utils/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── conformance/
│   └── benchmarks/
├── examples/
│   ├── quickstart.py
│   ├── rag_ingest.py
│   ├── pgvector_backend.py
│   ├── qdrant_backend.py
│   └── zepgpu_embedding_migration.py
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── prometheus.yml
├── k8s/
│   └── deployment.yaml
└── scripts/
    ├── run_benchmarks.py
    ├── seed_demo_data.py
    └── export_report.py
```

---

# Configuration

## Environment Variables

```env
# API
DEEPIRI_VECTOR_HOST=0.0.0.0
DEEPIRI_VECTOR_PORT=8100
DEEPIRI_VECTOR_ENV=development

# Catalog Database
DATABASE_URL=postgresql+asyncpg://deepiri:deepiri@localhost:5432/deepiri_vector
DATABASE_POOL_SIZE=10

# Default Backend
VECTOR_DEFAULT_BACKEND=native
VECTOR_DEFAULT_METRIC=cosine
VECTOR_DEFAULT_DIMENSION=768

# pgvector
PGVECTOR_DATABASE_URL=postgresql+asyncpg://deepiri:deepiri@localhost:5432/deepiri_vector
PGVECTOR_SCHEMA=public

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# Native Engine
NATIVE_VECTOR_DATA_DIR=./data/native
NATIVE_ENABLE_WAL=true
NATIVE_MAX_MEMORY_MB=2048

# ZepGPU
ZEPGPU_API_URL=http://localhost:8000
ZEPGPU_API_KEY=
ZEPGPU_ENABLE_BATCH_JOBS=true

# Object Storage
S3_ENDPOINT_URL=http://localhost:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET_NAME=deepiri-vector-artifacts

# Auth
JWT_SECRET_KEY=change-me
API_KEY_SECRET=change-me

# Benchmarks
BENCHMARK_DATA_DIR=./data/benchmarks
BENCHMARK_OUTPUT_DIR=./reports/benchmarks
```

---

# Testing Strategy

## Unit Tests

- [ ] Metric correctness
- [ ] Cosine normalization
- [ ] L2 ranking
- [ ] Dot-product ranking
- [ ] Top-k heap behavior
- [ ] Filter parsing
- [ ] Payload validation
- [ ] Schema validation
- [ ] Planner rules
- [ ] Backend capability validation

## Adapter Conformance Tests

Each backend must pass the same tests:

- [ ] Create collection
- [ ] Upsert records
- [ ] Search exact or supported ANN mode
- [ ] Delete records
- [ ] Apply metadata filters
- [ ] Return consistent result shape
- [ ] Handle unsupported capabilities clearly
- [ ] Health check works

## Integration Tests

- [ ] Native backend end-to-end
- [ ] pgvector backend with Docker Compose
- [ ] Qdrant backend with Docker Compose
- [ ] ZepGPU batch job submission
- [ ] Embedding migration workflow
- [ ] Benchmark workflow
- [ ] API auth and namespace isolation

## Regression Tests

- [ ] Dimension mismatch
- [ ] Zero-vector cosine rejection
- [ ] Non-finite vector rejection
- [ ] Duplicate IDs
- [ ] Delete then search
- [ ] Snapshot restore
- [ ] Backend unavailable
- [ ] Failed ZepGPU job retry
- [ ] Embedding rollback

---

# Benchmark Strategy

## Benchmark Goals

The benchmark harness should answer:

- Which backend is best for a collection size?
- When is exact search good enough?
- When does HNSW help?
- When does filtering make ANN worse?
- Does ZepGPU provide end-to-end benefit?
- Which embedding version performs better?
- Which reranker improves retrieval quality?

## Benchmark Matrix

| Benchmark | Native Exact | pgvector | Qdrant | ZepGPU | Purpose |
|---|---:|---:|---:|---:|---|
| Small collection exact | Yes | Yes | Yes | Optional | Verify baseline |
| Medium collection search | Yes | Yes | Yes | Optional | Compare latency |
| Filtered search | Yes | Yes | Yes | Optional | Test filter planner |
| Batch exact search | Yes | Optional | Optional | Yes | Test GPU batch value |
| Embedding migration | No | Yes | Yes | Yes | Test version workflow |
| Reranking | Optional | Optional | Optional | Yes | Test quality improvement |
| HNSW recall | Future | Yes | Yes | No | Compare ANN behavior |

## Required Metrics

- recall@1
- recall@5
- recall@10
- recall@100
- p50 latency
- p95 latency
- p99 latency
- throughput
- indexing time
- memory usage
- disk usage
- filter selectivity
- queue overhead
- transfer overhead
- total job cost

---

# Deployment Strategy

## Local Development

- Docker Compose with:
  - Deepiri Vector API
  - PostgreSQL + pgvector
  - Qdrant
  - MinIO
  - Redis
  - optional ZepGPU service

## MVP Deployment

- One API service
- PostgreSQL catalog
- External vector backend
- Optional native local storage
- Optional ZepGPU integration

## Production Deployment

- Kubernetes deployment
- PostgreSQL managed database
- Qdrant managed/self-hosted backend
- S3-compatible artifact storage
- Prometheus/Grafana
- API key/JWT auth
- Horizontal API scaling
- Backend-specific scaling

## Not in Initial Deployment

- Deepiri-owned distributed vector shards
- Multi-region replication
- Online shard rebalancing
- Native billion-vector disk engine
- Custom consensus layer

---

# Risks and Tradeoffs

## 1. Backend Abstraction Risk

Different vector backends do not expose identical behavior.

Mitigation:

- use capability registry
- expose backend-specific limits
- avoid pretending every backend supports every feature

## 2. Native Engine Scope Creep

Building a native engine can become a full database project.

Mitigation:

- exact search first
- no distributed claims
- HNSW only after benchmarks
- storage guarantees clearly scoped

## 3. ZepGPU Overhead

GPU batch workflows may not outperform CPU or backend-native execution after transfer overhead.

Mitigation:

- benchmark end-to-end
- report queue and transfer overhead
- remove GPU claims if no measurable benefit

## 4. Filtering Complexity

Filtered ANN can return poor results or fewer than top-k.

Mitigation:

- exact fallback for selective filters
- benchmark filtered recall
- do not claim filter-aware HNSW until proven

## 5. Migration Complexity

Embedding-version migration requires careful aliasing, rollback, and evaluation.

Mitigation:

- make embedding versions first-class
- support read aliases
- support rollback
- benchmark old vs new retrieval before promotion

## 6. Too Many Deployment Modes

Supporting native, pgvector, Qdrant, and ZepGPU can dilute testing.

Mitigation:

- conformance tests
- strict capability registry
- phase-gated backend support
- documented support levels

---

# Open Research Questions

1. Which first workload has enough batch size for ZepGPU to help?
2. What filtered-count threshold should trigger exact search?
3. Should native storage use Arrow, Lance, SQLite, or a custom segment format?
4. Is native HNSW strategically necessary, or should Deepiri embed hnswlib/USearch?
5. How should backend-neutral APIs expose backend-specific capabilities?
6. How much does retaining original vectors improve reranking and migration?
7. What is the smallest safe WAL/manifest design?
8. Should namespace mean authorization boundary, physical partition, or both?
9. What consistency guarantees do early users actually need?
10. Which sparse backend should power hybrid search?
11. Should RRF be the default hybrid fusion method?
12. How should GDPR/CCPA deletion propagate to snapshots and derived indexes?
13. What evidence would justify disk-native ANN?
14. Should reranking happen in the API service, backend, or ZepGPU?
15. What license and governance model best supports open-source adoption?

---

# Go / No-Go Gates

## Continue Native Engine Work If

- [ ] Native exact search is useful for local/dev workflows
- [ ] Native exact search is competitive for small collections
- [ ] Native HNSW prototype beats or matches embedded alternatives for target workloads
- [ ] Native storage can be made safe without dominating roadmap
- [ ] Users want an embedded local backend

## Stop Native Engine Work If

- [ ] pgvector or Qdrant handles target workloads well enough
- [ ] native performance remains materially worse
- [ ] crash-safe storage consumes too much engineering time
- [ ] users only care about adapters/migrations

## Continue ZepGPU Integration If

- [ ] batch embedding shows measurable throughput or cost benefit
- [ ] exact batch search is useful for benchmark ground truth
- [ ] reranking workflows benefit from scheduled GPU execution
- [ ] transfer and queue overhead are acceptable

## Stop ZepGPU-Specific Claims If

- [ ] queue overhead dominates runtime
- [ ] data transfer eliminates speedup
- [ ] backend-native execution is simpler and faster
- [ ] no user workload benefits

## Continue Toward Distributed Service If

- [ ] at least three design partners need it
- [ ] backend adapters are not enough
- [ ] a dedicated systems team exists
- [ ] consistency and scale requirements are clear

## Do Not Build Distributed Service If

- [ ] requirements are unclear
- [ ] no production demand exists
- [ ] single-node/adapters solve the problem
- [ ] team capacity is limited

---



# Priority Queue

| Priority | Task | Phase | Status |
|---|---|---|---|
| P0 | Create repo skeleton | Phase 1 | Planned |
| P0 | Implement core schemas | Phase 1 | Planned |
| P0 | Implement native exact search | Phase 1 | Planned |
| P0 | Add pgvector adapter | Phase 2 | Planned |
| P0 | Add Qdrant adapter | Phase 2 | Planned |
| P1 | Add benchmark harness | Phase 4 | Planned |
| P1 | Add embedding version metadata | Phase 5 | Planned |
| P1 | Add ZepGPU batch experiment | Phase 5 | Planned |
| P2 | Add native persistence | Phase 7 | Planned |
| P2 | Add hybrid search | Phase 8 | Planned |
| P3 | Add HNSW prototype | Phase 3 | Deferred |
| P3 | Add static sharding | Future | Deferred |
| P3 | Add replication | Future | Deferred |

---

# Milestones

| Milestone | Target | Status |
|---|---|---|
| M1: Core API + Native Exact | Week 1-2 | Planned |
| M2: pgvector Adapter | Week 3-4 | Planned |
| M3: Qdrant Adapter | Week 5-6 | Planned |
| M4: Benchmark Harness | Week 7-8 | Planned |
| M5: Embedding Version Migration | Week 9-10 | Planned |
| M6: ZepGPU Batch Experiment | Week 11-12 | Planned |
| M7: Service API + SDKs | Week 13-16 | Planned |
| M8: Native Persistence | Week 17-20 | Planned |
| M9: Metadata Filtering + Hybrid | Week 21-24 | Planned |
| M10: Production Hardening | Week 25-30 | Planned |

---

# Success Criteria

## Functional

- [ ] Can create collections
- [ ] Can upsert vectors
- [ ] Can search vectors
- [ ] Can use pgvector backend
- [ ] Can use Qdrant backend
- [ ] Can use native exact backend
- [ ] Can run benchmarks
- [ ] Can run at least one ZepGPU vector job

## Reliable

- [ ] Adapter conformance tests pass
- [ ] Native persistence survives restart
- [ ] Backend failures return clear errors
- [ ] Embedding migration supports rollback

## Performant

- [ ] Exact search benchmarked
- [ ] Backend search benchmarked
- [ ] Batch ZepGPU overhead measured
- [ ] HNSW not added without recall/latency data

## Secure

- [ ] Namespace isolation works
- [ ] API keys/JWT supported
- [ ] Payload limits enforced
- [ ] Deletion semantics documented

## Observable

- [ ] Query metrics available
- [ ] Backend health metrics available
- [ ] Benchmark results exportable
- [ ] ZepGPU job metrics tracked

## Documented

- [ ] Quickstart complete
- [ ] API docs complete
- [ ] Backend capability docs complete
- [ ] Benchmark guide complete
- [ ] Migration guide complete

---


