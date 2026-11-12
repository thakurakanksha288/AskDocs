"""API integration tests using TestClient with mocked services."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def test_root_endpoint():
    with (
        patch("app.services.cache.cache_service") as mock_cache,
        patch("app.services.vector_store.vector_store") as mock_vs,
        patch("app.db.session.engine"),
    ):
        mock_cache.is_available = False
        mock_cache.connect = AsyncMock()
        mock_cache.close = AsyncMock()
        mock_vs.load = MagicMock()
        mock_vs.total_vectors = 0

        from fastapi.testclient import TestClient
        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/")
            assert resp.status_code == 200
            assert "AskDocs" in resp.json()["message"]


def test_ask_validation_too_short():
    with (
        patch("app.services.cache.cache_service") as mock_cache,
        patch("app.services.vector_store.vector_store") as mock_vs,
        patch("app.db.session.engine"),
    ):
        mock_cache.is_available = False
        mock_cache.connect = AsyncMock()
        mock_cache.close = AsyncMock()
        mock_vs.load = MagicMock()
        mock_vs.total_vectors = 0

        from fastapi.testclient import TestClient
        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/ask", json={"question": "Hi"})
            assert resp.status_code == 422
