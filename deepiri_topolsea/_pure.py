"""Pure-Python fallback when native extension is not built."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class PureDatabase:
    def __init__(self, data_dir: str) -> None:
        self.root = Path(data_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._collections: dict[str, PureCollection] = {}

    def list_collections(self) -> list[str]:
        names = [
            p.name for p in self.root.iterdir() if p.is_dir() and (p / "manifest.json").exists()
        ]
        return sorted(names)

    def get_or_create_collection(self, name: str, dimension: int, metric: str) -> PureCollection:
        if name not in self._collections:
            self._collections[name] = PureCollection(self.root / name, dimension, metric)
        return self._collections[name]


class PureCollection:
    def __init__(self, path: Path, dimension: int, metric: str) -> None:
        self.path = path
        self.dimension = dimension
        self.metric = metric
        self.vectors: dict[str, list[float]] = {}
        self.metadatas: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        manifest = self.path / "manifest.json"
        if manifest.exists():
            data = json.loads(manifest.read_text())
            self.dimension = data["config"]["dimension"]
            self.metric = data["config"]["metric"]
        meta_path = self.path / "metadata.json"
        if meta_path.exists():
            raw = json.loads(meta_path.read_text())
            raw.pop("__id_map__", None)
            self.metadatas = {k: v for k, v in raw.items() if isinstance(v, dict)}
        vec_path = self.path / "vectors_pure.json"
        if vec_path.exists():
            self.vectors = json.loads(vec_path.read_text())

    def count(self) -> int:
        return len(self.vectors)

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        for i, (vid, vec) in enumerate(zip(ids, vectors)):
            if len(vec) != self.dimension:
                raise ValueError(f"dimension mismatch for {vid}")
            self.vectors[vid] = vec
            if metadatas and i < len(metadatas) and metadatas[i]:
                self.metadatas[vid] = metadatas[i]

    def delete(self, ids: list[str]) -> None:
        for vid in ids:
            self.vectors.pop(vid, None)
            self.metadatas.pop(vid, None)

    def query(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
        ef: int = 64,
    ) -> list[dict[str, Any]]:
        del ef
        scored: list[tuple[str, float]] = []
        for vid, vec in self.vectors.items():
            if filter:
                meta = self.metadatas.get(vid, {})
                if not all(meta.get(k) == v for k, v in filter.items()):
                    continue
            scored.append((vid, _distance(self.metric, query_vector, vec)))
        scored.sort(key=lambda x: x[1])
        out = []
        for vid, dist in scored[:top_k]:
            out.append(
                {
                    "id": vid,
                    "distance": dist,
                    "score": 1.0 / (1.0 + dist),
                    "metadata": self.metadatas.get(vid),
                }
            )
        return out

    def persist(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "config": {
                "name": self.path.name,
                "dimension": self.dimension,
                "metric": self.metric,
                "index_kind": "flat",
            },
            "next_id": len(self.vectors),
        }
        (self.path / "manifest.json").write_text(json.dumps(manifest, indent=2))
        meta = dict(self.metadatas)
        (self.path / "metadata.json").write_text(json.dumps(meta, indent=2))
        (self.path / "vectors_pure.json").write_text(json.dumps(self.vectors))


def _distance(metric: str, a: list[float], b: list[float]) -> float:
    if metric in ("l2", "euclidean"):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    if metric == "cosine":
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na * nb < 1e-8:
            return 1.0
        return 1.0 - dot / (na * nb)
    # dot product
    return -sum(x * y for x, y in zip(a, b))
