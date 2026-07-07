"""Integration tests for Z-Column index."""

from __future__ import annotations

import tempfile

import pytest

from deepiri_topolsea import Client

topolsea_native = pytest.importorskip("topolsea_native")


def test_zcolumn_create_search_persist() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        client = Client(tmp)
        col = client.get_or_create_collection(
            "ztest", dimension=4, metric="cosine", index="zcolumn"
        )
        col.upsert(
            ids=["a", "b", "c"],
            vectors=[
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
        )
        results = col.query(query_vector=[1.0, 0.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0].id == "a"
        col.persist()

        client2 = Client(tmp)
        col2 = client2.get_or_create_collection(
            "ztest", dimension=4, metric="cosine", index="zcolumn"
        )
        assert col2.count() == 3
        results2 = col2.query(query_vector=[1.0, 0.0, 0.0, 0.0], top_k=2)
        assert results2[0].id == "a"
