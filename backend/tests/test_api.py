"""Tests for SCRAG FastAPI REST API endpoints."""

import pytest
from fastapi.testclient import TestClient

from scrag.api.server import app

client = TestClient(app)


def test_health_endpoint():
    """Verify health check returns valid SCRAG status."""
    with TestClient(app) as tc:
        response = tc.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["project"] == "Self-Correcting RAG (SCRAG)"
        assert "execution_mode" in data
        assert "generator_model" in data


def test_index_endpoint():
    """Verify document indexing endpoint."""
    with TestClient(app) as tc:
        payload = {
            "text": "SCRAG is an advanced self-correcting RAG pipeline that combines CRAG and Self-RAG.",
            "source": "api_test.txt",
        }
        response = tc.post("/api/v1/index", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["indexed_chunks"] >= 1
