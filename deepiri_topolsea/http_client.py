//! Thin HTTP client for a remote `topolsea-server` (Phase A3).
//!
//! Use alongside the embedded PyO3 / pure-Python client when the DB runs as a service.

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urljoin

from deepiri_topolsea.types import QueryResult


class HttpClient:
    """REST client for Topolsea Phase A server."""

    def __init__(self, base_url: str = "http://127.0.0.1:6333", api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.api_key = api_key

    def _headers(self, content_type: bool = True) -> dict[str, str]:
        headers: dict[str, str] = {}
        if content_type:
            headers["content-type"] = "application/json"
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> Any:
        url = urljoin(self.base_url, path.lstrip("/"))
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers=self._headers(content_type=body is not None),
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code}: {detail}") from e

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def list_collections(self) -> list[str]:
        return list(self._request("GET", "/v1/collections")["collections"])

    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: str = "cosine",
        index: str = "hnsw",
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/collections",
            {"name": name, "dimension": dimension, "metric": metric, "index": index},
        )

    def get_or_create_collection(
        self,
        name: str,
        dimension: int,
        metric: str = "cosine",
        index: str = "hnsw",
    ) -> HttpCollection:
        try:
            self.create_collection(name, dimension, metric, index)
        except RuntimeError as e:
            if "409" not in str(e) and "already exists" not in str(e).lower():
                # Collection may already exist — probe via GET.
                try:
                    self._request("GET", f"/v1/collections/{name}")
                except RuntimeError:
                    raise e from e
        return HttpCollection(self, name, dimension, metric)

    def delete_collection(self, name: str) -> None:
        self._request("DELETE", f"/v1/collections/{name}")


class HttpCollection:
    """Collection handle backed by HTTP."""

    def __init__(self, client: HttpClient, name: str, dimension: int, metric: str) -> None:
        self._client = client
        self.name = name
        self.dimension = dimension
        self.metric = metric

    def count(self) -> int:
        info = self._client._request("GET", f"/v1/collections/{self.name}")
        return int(info["vectors"])

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any] | None] | None = None,
    ) -> None:
        body: dict[str, Any] = {"ids": ids, "vectors": vectors}
        if metadatas is not None:
            body["metadatas"] = metadatas
        self._client._request("PUT", f"/v1/collections/{self.name}/upsert", body)

    def query(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
        ef: int = 64,
    ) -> list[QueryResult]:
        body: dict[str, Any] = {"vector": query_vector, "top_k": top_k, "ef": ef}
        if filter is not None:
            body["filter"] = filter
        raw = self._client._request("POST", f"/v1/collections/{self.name}/search", body)
        return [
            QueryResult(
                id=item.get("id"),
                distance=float(item["distance"]),
                score=float(item["score"]),
                metadata=item.get("metadata"),
            )
            for item in raw["hits"]
        ]

    def persist(self) -> None:
        self._client._request("POST", f"/v1/collections/{self.name}/persist")

    def explain_query(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
        ef: int = 64,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"vector": query_vector, "top_k": top_k, "ef": ef}
        if filter is not None:
            body["filter"] = filter
        return self._client._request("POST", f"/v1/collections/{self.name}/explain", body)
