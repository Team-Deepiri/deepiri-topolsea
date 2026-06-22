"""Shared types for the Topolsea Python SDK."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class DistanceMetric(str, Enum):
    L2 = "l2"
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"


@dataclass
class QueryResult:
    id: str | None
    distance: float
    score: float
    metadata: dict[str, Any] | None = None


SearchResult = QueryResult
