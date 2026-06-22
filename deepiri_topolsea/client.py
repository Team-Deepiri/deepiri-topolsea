"""Pure-Python client wrapping the native Rust engine when available."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from deepiri_topolsea.types import DistanceMetric

if TYPE_CHECKING:
    from deepiri_topolsea.collection import Collection


class Client:
    """Entry point for Topolsea vector database operations."""

    def __init__(self, data_dir: str | Path = "./data") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        try:
            from topolsea_native import PyClient  # type: ignore[import-not-found]

            self._native = PyClient(str(self.data_dir))
            self._use_native = True
        except ImportError:
            from deepiri_topolsea._pure import PureDatabase

            self._native = PureDatabase(str(self.data_dir))
            self._use_native = False

    def list_collections(self) -> list[str]:
        return self._native.list_collections()

    def get_or_create_collection(
        self,
        name: str,
        dimension: int,
        metric: DistanceMetric | str = "cosine",
    ) -> Collection:
        from deepiri_topolsea.collection import Collection

        metric_str = metric.value if isinstance(metric, DistanceMetric) else metric
        if self._use_native:
            native_col = self._native.get_or_create_collection(name, dimension, metric_str)
            return Collection(native=native_col, name=name, dimension=dimension, metric=metric_str)
        pure_col = self._native.get_or_create_collection(name, dimension, metric_str)
        return Collection(native=pure_col, name=name, dimension=dimension, metric=metric_str)
