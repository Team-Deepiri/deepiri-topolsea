"""Tests for the public Client API."""

from deepiri_topolsea import Client


def test_client_pure_mode(tmp_path):
    client = Client(tmp_path)
    col = client.get_or_create_collection("embeddings", dimension=8, metric="cosine")
    col.upsert(
        ids=["doc1"],
        vectors=[[0.1] * 8],
        metadatas=[{"source": "test"}],
    )
    assert col.count() == 1
    hits = col.query([0.1] * 8, top_k=1)
    assert hits[0].id == "doc1"
    col.persist()
