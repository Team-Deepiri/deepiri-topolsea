"""Deepiri Topolsea — topological vector database Python SDK."""

from deepiri_topolsea.client import Client
from deepiri_topolsea.collection import Collection
from deepiri_topolsea.types import QueryResult, SearchResult

__all__ = ["Client", "Collection", "QueryResult", "SearchResult"]
__version__ = "0.1.0"
