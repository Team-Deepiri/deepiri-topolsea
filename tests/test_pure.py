"""Tests for the pure-Python fallback."""

from deepiri_topolsea._pure import PureDatabase, _distance


def test_pure_upsert_and_query(tmp_path):
    db = PureDatabase(str(tmp_path))
    col = db.get_or_create_collection("docs", dimension=3, metric="l2")
    col.upsert(
        ids=["a", "b"],
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        metadatas=[{"topic": "rust"}, {"topic": "python"}],
    )
    results = col.query([0.9, 0.1, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "a"


def test_pure_filter(tmp_path):
    db = PureDatabase(str(tmp_path))
    col = db.get_or_create_collection("docs", dimension=2, metric="cosine")
    col.upsert(
        ids=["a", "b"],
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[{"lang": "rust"}, {"lang": "py"}],
    )
    results = col.query([1.0, 0.0], top_k=5, filter={"lang": "rust"})
    assert len(results) == 1
    assert results[0]["id"] == "a"


def test_distance_cosine():
    d = _distance("cosine", [1.0, 0.0], [1.0, 0.0])
    assert d < 1e-6
