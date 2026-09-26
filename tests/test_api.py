"""Tests for the FastAPI Scoring API layer."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with lifespan context active."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    """Verify /health returns 200 and reports model loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "threshold" in data
    assert "version" in data


def test_score_normal_transaction(client: TestClient):
    """Verify scoring a normal transaction produces a low risk score."""
    payload = {
        "transaction_id": "test-tx-normal-001",
        "amount": 25.50,
        "features": [0.0] * 28,
        "timestamp": "2026-09-23T12:00:00Z",
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["transaction_id"] == "test-tx-normal-001"
    assert "risk_score" in data
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["is_suspicious"] is False
    assert data["threshold"] == 0.80
    assert "scored_at" in data
    assert "latency_ms" in data
    # Verify sub-100ms latency requirement
    assert data["latency_ms"] < 100.0


def test_score_suspicious_transaction(client: TestClient):
    """Verify scoring an anomalous transaction produces a high risk score and flag."""
    payload = {
        "transaction_id": "test-tx-fraud-001",
        "amount": 8900.00,
        "features": [4.0 if i % 2 == 0 else -4.0 for i in range(28)],
        "timestamp": "2026-09-23T12:05:00Z",
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["transaction_id"] == "test-tx-fraud-001"
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["is_suspicious"] is True
    assert data["risk_score"] >= 0.80


def test_score_auto_generates_transaction_id(client: TestClient):
    """Verify transaction_id is auto-generated if omitted."""
    payload = {
        "amount": 100.0,
        "features": [0.1] * 28,
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] is not None
    assert len(data["transaction_id"]) > 0


def test_score_invalid_features_count(client: TestClient):
    """Verify sending incorrect number of PCA features returns 422."""
    payload = {
        "amount": 50.0,
        "features": [1.0, 2.0, 3.0],  # only 3 features instead of 28
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_score_negative_amount_validation(client: TestClient):
    """Verify negative amount is rejected with 422."""
    payload = {
        "amount": -50.0,
        "features": [0.0] * 28,
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 422


def test_metrics_endpoint(client: TestClient):
    """Verify /metrics tracks scored requests, flags, and latency."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["total_requests"] >= 2
    assert data["suspicious_count"] >= 1
    assert 0.0 <= data["flag_rate_pct"] <= 100.0
    assert data["avg_latency_ms"] < 100.0


def test_dashboard_endpoints(client: TestClient):
    """Verify /dashboard returns HTML and /dashboard/data returns live JSON."""
    html_res = client.get("/dashboard")
    assert html_res.status_code == 200
    assert "FraudLens" in html_res.text
    assert "<!DOCTYPE html>" in html_res.text

    data_res = client.get("/dashboard/data")
    assert data_res.status_code == 200
    d = data_res.json()
    assert "total_requests" in d
    assert "recent_transactions" in d

