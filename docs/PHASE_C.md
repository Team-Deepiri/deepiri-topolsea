# Phase C — Operate like Milvus/Qdrant

Stacked on Phase B. Exit criteria: multi-node operable deployment an SRE would keep alive.

## Acceptance map

| ID | Delivered |
|---|---|
| **C10** | Primary + **replica** endpoints; sync replicate upsert **and delete**; `require_replica_ack` durability; membership + **heartbeat**; failover query list |
| **C11** | Shard query carries **filter + metadata**; client **retries** + **circuit breaker**; primary→replica failover; `/topolsea/v1/shard/health`; configurable replica/fan-out timeout |
| **C12** | `GET /metrics` Prometheus (QPS, p50/p99, per-route, WAL lag, replicate fail); TraceLayer + `x-request-id` / `traceparent`; auto-flush WAL lag sampling |
| **C13** | Namespaces (`x-namespace` / tenant keys / `/v1/ns/:ns/...`); `/livez` `/readyz`; Dockerfile (no curl healthcheck); compose; Helm + **ServiceMonitor** |
| **C14** | Snapshots create/list/meta/restore/delete; **partial collection** snapshots; scoped restore (default) vs `replace_all` |

## Replication (C10)

```bash
# Register backup for shard 0 of logical collection "corp"
curl -X POST localhost:6333/v1/shards/corp/replicas \
  -H 'content-type: application/json' \
  -d '{"shard_id":0,"url":"http://replica:6333"}'

# Fail upserts when replica ack fails
curl -X PUT localhost:6333/v1/shards/corp/replica-policy \
  -H 'content-type: application/json' \
  -d '{"require_replica_ack":true,"replica_timeout_ms":5000}'

# Membership + heartbeat
curl -X PUT localhost:6333/v1/cluster/membership \
  -H 'content-type: application/json' \
  -d '{"nodes":[{"id":"n1","advertise_url":"http://n1:6333","role":"data"}]}'
curl -X POST localhost:6333/v1/cluster/heartbeat \
  -H 'content-type: application/json' \
  -d '{"id":"n1","healthy":true}'
```

Replica nodes must host the physical collection and accept:

- `POST /topolsea/v1/replicate/upsert`
- `POST /topolsea/v1/replicate/delete`
- `GET /topolsea/v1/shard/health`

## Hardened shards (C11)

Remote `ShardQueryRequest` includes optional `filter`. Fan-out uses retries (2) + circuit breaker + ordered failover endpoints. Write replication uses `replica_timeout_ms`; query fan-out uses `query_timeout_ms` (default 30s, independently tunable via replica-policy).

```bash
curl -X PUT localhost:6333/v1/shards/corp/replica-policy \
  -H 'content-type: application/json' \
  -d '{"require_replica_ack":true,"replica_timeout_ms":5000,"query_timeout_ms":30000}'
```

## Metrics (C12)

```bash
curl -s localhost:6333/metrics | head
# topolsea_http_requests_total, topolsea_search_total, topolsea_http_latency_p99_micros,
# topolsea_wal_lag_last, topolsea_replicate_fail_total, topolsea_http_route_p99_micros{route=...}
```

Responses include `x-request-id` and W3C-style `traceparent` for collector correlation.

## Namespaces + packaging (C13)

```bash
export TOPOLSEA_TENANT_KEYS='{"acme-key":"acme"}'
curl -H 'x-api-key: secret' -H 'x-namespace: acme' ...
# Explicit path namespace:
curl -H 'x-api-key: secret' localhost:6333/v1/ns/acme/collections
```

```bash
docker compose -f deploy/docker-compose.yml up --build
helm install topolsea deploy/helm/topolsea
# ServiceMonitor scrapes /metrics when serviceMonitor.enabled=true
```

Probes: `GET /livez`, `GET /readyz` (Helm uses these).

## Snapshots (C14)

```bash
curl -X POST localhost:6333/v1/snapshots -d '{"name":"nightly"}'
curl -X POST localhost:6333/v1/snapshots -d '{"name":"partial","collections":["acme/docs"]}'
curl localhost:6333/v1/snapshots
curl localhost:6333/v1/snapshots/nightly
curl -X POST localhost:6333/v1/snapshots/nightly/restore -d '{"replace_all":false}'
curl -X DELETE localhost:6333/v1/snapshots/nightly
```
