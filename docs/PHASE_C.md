# Phase C — Operate like Milvus/Qdrant

Stacked on Phase B. Exit criteria: multi-node operable deployment an SRE would keep alive.

## Acceptance map

| ID | Delivered |
|---|---|
| **C10** | Primary + **replica** endpoints per shard (`ShardClusterConfig.replicas`); sync replicate on sharded upsert; cluster **membership** API; failover query list |
| **C11** | Shard query carries **filter + metadata**; client **retries** + **circuit breaker**; primary→replica failover |
| **C12** | `GET /metrics` Prometheus text (QPS counters, p99 sample, WAL lag gauge); request tracing via `TraceLayer` |
| **C13** | **Namespaces** (`x-namespace` / tenant API keys); `deploy/Dockerfile`, `docker-compose.yml`, Helm chart |
| **C14** | Snapshot create/list/restore/delete under `__snapshots__/` |

## Replication (C10)

```bash
# Register backup for shard 0 of logical collection "corp"
curl -X POST localhost:6333/v1/shards/corp/replicas \
  -H 'content-type: application/json' \
  -d '{"shard_id":0,"url":"http://replica:6333"}'

# Membership
curl -X PUT localhost:6333/v1/cluster/membership \
  -H 'content-type: application/json' \
  -d '{"nodes":[{"id":"n1","advertise_url":"http://n1:6333","role":"data"}]}'
```

Replica nodes must run with `--shard-collection <physical>` (or host the physical collection) and accept `POST /topolsea/v1/replicate/upsert`.

## Hardened shards (C11)

Remote `ShardQueryRequest` now includes optional `filter`. Fan-out uses retries (2) + circuit breaker + ordered failover endpoints.

## Metrics (C12)

```bash
curl -s localhost:6333/metrics | head
# topolsea_http_requests_total, topolsea_search_total, topolsea_http_latency_p99_micros, ...
```

## Namespaces + packaging (C13)

```bash
# Tenant key forces namespace
export TOPOLSEA_TENANT_KEYS='{"acme-key":"acme"}'
# Or header with global key:
curl -H 'x-api-key: secret' -H 'x-namespace: acme' ...
```

```bash
docker compose -f deploy/docker-compose.yml up --build
helm install topolsea deploy/helm/topolsea
```

## Snapshots (C14)

```bash
curl -X POST localhost:6333/v1/snapshots -d '{"name":"nightly"}'
curl localhost:6333/v1/snapshots
curl -X POST localhost:6333/v1/snapshots/nightly/restore
curl -X DELETE localhost:6333/v1/snapshots/nightly
```
