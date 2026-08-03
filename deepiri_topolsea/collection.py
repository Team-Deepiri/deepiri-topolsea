"""Collection API for upsert, query, and delete."""

from __future__ import annotations

from typing import Any

from deepiri_topolsea.types import QueryResult


class Collection:
    """A named vector collection."""

    def __init__(
        self,
        native: Any,
        name: str,
        dimension: int,
        metric: str,
    ) -> None:
        self._native = native
        self.name = name
        self.dimension = dimension
        self.metric = metric

    def count(self) -> int:
        return self._native.count()

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        meta_arg = metadatas if metadatas is not None else None
        self._native.upsert(ids, vectors, meta_arg)

    def delete(self, ids: list[str]) -> None:
        self._native.delete(ids)

    def query(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
        ef: int = 64,
    ) -> list[QueryResult]:
        raw = self._native.query(query_vector, top_k, filter, ef)
        return [
            QueryResult(
                id=item["id"],
                distance=float(item["distance"]),
                score=float(item["score"]),
                metadata=item.get("metadata"),
            )
            for item in raw
        ]

    def query_batch(
        self,
        query_vectors: list[list[float]],
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
        ef: int = 64,
    ) -> list[list[QueryResult]]:
        raw = self._native.query_batch(query_vectors, top_k, filter, ef)
        return [
            [
                QueryResult(
                    id=item["id"],
                    distance=float(item["distance"]),
                    score=float(item["score"]),
                    metadata=item.get("metadata"),
                )
                for item in batch
            ]
            for batch in raw
        ]

    def persist(self) -> None:
        self._native.persist()

    def explain_query(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
        ef: int = 64,
    ) -> dict[str, Any]:
        raw = self._native.explain_query(query_vector, top_k, filter, ef)
        return {
            "results": [
                QueryResult(
                    id=item["id"],
                    distance=float(item["distance"]),
                    score=float(item["score"]),
                    metadata=item.get("metadata"),
                )
                for item in raw["results"]
            ],
            "explain": raw["explain"],
        }

    def zcolumn_stats(self) -> dict[str, Any] | None:
        stats = self._native.zcolumn_stats()
        return stats if stats is not None else None
