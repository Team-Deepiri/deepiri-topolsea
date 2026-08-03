"""Smoke tests for the HTTP client (requires a running server — skipped by default)."""

from __future__ import annotations

import pytest

from deepiri_topolsea.http_client import HttpClient


@pytest.mark.skip(reason="requires topolsea-server; covered by Rust HTTP smoke test")
def test_http_client_health() -> None:
    client = HttpClient("http://127.0.0.1:6333")
    assert client.health()["status"] == "ok"
