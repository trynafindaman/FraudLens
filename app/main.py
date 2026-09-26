"""FastAPI service for Real-Time Fraud Detection Scoring.

Provides:
- POST /score: score incoming transaction for risk in sub-100ms.
- GET /health: liveness and readiness check.
- GET /metrics: observability endpoint for request count, flag rate, and latency.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse

from app.schemas import (
    HealthResponse,
    MetricsResponse,
    TransactionScoreRequest,
    TransactionScoreResponse,
)
from src.data_prep import (
    SAMPLE_DATA_PATH,
    generate_synthetic_sample,
    get_stratified_split,
    load_data,
    preprocess_data,
)
from src.model import DEFAULT_MODEL_PATH, FraudDetector

# Configure logger
logger = logging.getLogger("fraud_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Global in-memory state
detector: FraudDetector | None = None
stats = {
    "total_requests": 0,
    "suspicious_count": 0,
    "total_latency_ms": 0.0,
    "recent_transactions": [],
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager to load or train model on startup."""
    global detector
    model_path = Path(os.getenv("FRAUD_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    threshold = float(os.getenv("FRAUD_THRESHOLD", "0.80"))

    if model_path.exists():
        logger.info("Loading pre-trained fraud detection model from %s ...", model_path)
        detector = FraudDetector.load(model_path)
        detector.threshold = threshold
    else:
        logger.warning("No pre-trained model artifact found at %s. Bootstrapping with sample data...", model_path)
        if not SAMPLE_DATA_PATH.exists():
            generate_synthetic_sample(output_path=SAMPLE_DATA_PATH, n_samples=2000, fraud_ratio=0.01)
        df = load_data(SAMPLE_DATA_PATH)
        X, y, scaler = preprocess_data(df, fit_scaler=True)
        X_train, _, _, _ = get_stratified_split(X, y, test_size=0.2)
        detector = FraudDetector(contamination=0.01, n_estimators=50, random_state=42, threshold=threshold)
        detector.fit(X_train, scaler=scaler)
        detector.save(model_path)
        logger.info("Bootstrapped model saved to %s.", model_path)

    logger.info("Fraud Detection API ready. Configured threshold: %.2f", detector.threshold)
    yield
    logger.info("Shutting down Fraud Detection API.")


app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="Scores transactions for fraud risk in real-time using an Isolation Forest anomaly detection model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, summary="Liveness & Readiness Check")
async def health_check() -> HealthResponse:
    """Check API operational health and model readiness."""
    is_ready = detector is not None and detector.is_fitted
    return HealthResponse(
        status="ok" if is_ready else "degraded",
        model_loaded=is_ready,
        threshold=detector.threshold if detector else 0.80,
        version="1.0.0",
    )


@app.post(
    "/score",
    response_model=TransactionScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Score a Transaction for Fraud Risk",
)
async def score_transaction(payload: TransactionScoreRequest) -> TransactionScoreResponse:
    """Score a single incoming transaction and return risk score and suspicious flag.

    Target latency: sub-100ms.
    """
    if detector is None or not detector.is_fitted:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or ready for inference.",
        )

    start_time = time.perf_counter()

    try:
        # Run inference using the pre-fitted FraudDetector
        result = detector.predict_single(
            amount=payload.amount,
            features=payload.features,
        )
    except Exception as exc:
        logger.error("Inference failure for transaction %s: %s", payload.transaction_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference error: {str(exc)}",
        ) from exc

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # Update runtime observability metrics
    stats["total_requests"] += 1
    if result["is_suspicious"]:
        stats["suspicious_count"] += 1
    stats["total_latency_ms"] += latency_ms

    scored_at = datetime.now(timezone.utc).isoformat()

    tx_record = {
        "transaction_id": str(payload.transaction_id)[:8],
        "amount": round(payload.amount, 2),
        "risk_score": round(result["risk_score"], 4),
        "is_suspicious": result["is_suspicious"],
        "latency_ms": round(latency_ms, 2),
        "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
    }
    stats["recent_transactions"].insert(0, tx_record)
    if len(stats["recent_transactions"]) > 20:
        stats["recent_transactions"].pop()

    logger.info(
        "Scored tx=%s | amount=$%.2f | score=%.4f | suspicious=%s | latency=%.2fms",
        payload.transaction_id,
        payload.amount,
        result["risk_score"],
        result["is_suspicious"],
        latency_ms,
    )

    response = TransactionScoreResponse(
        transaction_id=str(payload.transaction_id),
        risk_score=result["risk_score"],
        is_suspicious=result["is_suspicious"],
        threshold=result["threshold"],
        scored_at=scored_at,
        latency_ms=round(latency_ms, 3),
    )
    return response


@app.get("/metrics", response_model=MetricsResponse, summary="Observability Metrics")
async def get_metrics() -> MetricsResponse:
    """Retrieve runtime observability metrics including flag rate and average latency."""
    total = stats["total_requests"]
    suspicious = stats["suspicious_count"]
    flag_rate = (suspicious / total * 100.0) if total > 0 else 0.0
    avg_latency = (stats["total_latency_ms"] / total) if total > 0 else 0.0

    return MetricsResponse(
        total_requests=total,
        suspicious_count=suspicious,
        flag_rate_pct=round(flag_rate, 2),
        avg_latency_ms=round(avg_latency, 3),
    )


@app.get("/dashboard/data")
async def get_dashboard_data() -> JSONResponse:
    """Return live metrics and recent transactions for dashboard polling."""
    total = stats["total_requests"]
    suspicious = stats["suspicious_count"]
    flag_rate = (suspicious / total * 100.0) if total > 0 else 0.0
    avg_latency = (stats["total_latency_ms"] / total) if total > 0 else 0.0

    return JSONResponse({
        "total_requests": total,
        "suspicious_count": suspicious,
        "flag_rate_pct": round(flag_rate, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "threshold": detector.threshold if detector else 0.80,
        "recent_transactions": stats["recent_transactions"],
    })


@app.get("/dashboard", response_class=HTMLResponse, summary="Real-Time Metrics Dashboard")
async def get_dashboard() -> str:
    """Serve a clean real-time HTML dashboard with live KPI cards and transactions feed."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>FraudLens — Real-Time Streaming Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0D1117;
      --card-bg: #161B22;
      --border: #30363D;
      --text: #C9D1D9;
      --text-dim: #8B949E;
      --red: #F85149;
      --green: #3FB950;
      --blue: #58A6FF;
      --mono: 'JetBrains Mono', monospace;
      --sans: 'Inter', sans-serif;
    }
    body {
      margin: 0; padding: 24px; background: var(--bg); color: var(--text);
      font-family: var(--sans); line-height: 1.5;
    }
    .header {
      display: flex; align-items: baseline; justify-content: space-between;
      border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px;
    }
    .header h1 { margin: 0; font-size: 24px; font-weight: 700; color: #FFF; }
    .badge {
      font-family: var(--mono); font-size: 12px; padding: 4px 10px;
      border-radius: 12px; background: rgba(63, 185, 80, 0.15); color: var(--green); border: 1px solid var(--green);
    }
    .kpi-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 28px;
    }
    .kpi-card {
      background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 18px;
    }
    .kpi-label { font-size: 13px; color: var(--text-dim); text-transform: uppercase; font-family: var(--mono); }
    .kpi-val { font-size: 32px; font-weight: 700; margin-top: 6px; font-family: var(--mono); color: #FFF; }
    .kpi-val.danger { color: var(--red); }
    .table-card {
      background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 20px;
    }
    .table-title { margin: 0 0 16px; font-size: 16px; font-weight: 600; color: #FFF; }
    table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 13px; }
    th { text-align: left; padding: 10px 12px; color: var(--text-dim); border-bottom: 1px solid var(--border); font-size: 11px; text-transform: uppercase; }
    td { padding: 10px 12px; border-bottom: 1px solid var(--border); }
    tr:last-child td { border-bottom: none; }
    .tag { padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
    .tag.normal { background: rgba(63, 185, 80, 0.15); color: var(--green); }
    .tag.suspicious { background: rgba(248, 81, 73, 0.2); color: var(--red); border: 1px solid var(--red); }
  </style>
</head>
<body>
  <div class="header">
    <h1>🛡️ FraudLens Real-Time Detection API</h1>
    <span class="badge">● LIVE STREAMING</span>
  </div>

  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">Total Transactions Scored</div>
      <div class="kpi-val" id="totalRequests">0</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Fraudulent / Suspicious</div>
      <div class="kpi-val danger" id="suspiciousCount">0</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Current Flag Rate</div>
      <div class="kpi-val" id="flagRate">0.0%</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Average Scoring Latency</div>
      <div class="kpi-val" id="avgLatency">0.0 ms</div>
    </div>
  </div>

  <div class="table-card">
    <div class="table-title">Recent Real-Time Transaction Stream</div>
    <table>
      <thead>
        <tr>
          <th>Time</th>
          <th>Tx ID</th>
          <th>Amount ($)</th>
          <th>Risk Score</th>
          <th>Status</th>
          <th>Latency</th>
        </tr>
      </thead>
      <tbody id="txBody">
        <tr><td colspan="6" style="text-align:center; color: var(--text-dim);">Waiting for stream traffic...</td></tr>
      </tbody>
    </table>
  </div>

  <script>
    async function updateDashboard() {
      try {
        const res = await fetch('/dashboard/data');
        const data = await res.json();

        document.getElementById('totalRequests').innerText = data.total_requests.toLocaleString();
        document.getElementById('suspiciousCount').innerText = data.suspicious_count.toLocaleString();
        document.getElementById('flagRate').innerText = data.flag_rate_pct.toFixed(2) + '%';
        document.getElementById('avgLatency').innerText = data.avg_latency_ms.toFixed(2) + ' ms';

        const tbody = document.getElementById('txBody');
        if (data.recent_transactions && data.recent_transactions.length > 0) {
          tbody.innerHTML = data.recent_transactions.map(tx => {
            const statusClass = tx.is_suspicious ? 'suspicious' : 'normal';
            const statusLabel = tx.is_suspicious ? 'FLAGGED' : 'LEGIT';
            return `<tr>
              <td>${tx.time}</td>
              <td><code>${tx.transaction_id}</code></td>
              <td>$${tx.amount.toFixed(2)}</td>
              <td><strong>${tx.risk_score.toFixed(4)}</strong></td>
              <td><span class="tag ${statusClass}">${statusLabel}</span></td>
              <td>${tx.latency_ms.toFixed(1)} ms</td>
            </tr>`;
          }).join('');
        }
      } catch (err) {
        console.error("Dashboard poll failed:", err);
      }
    }
    setInterval(updateDashboard, 1500);
    updateDashboard();
  </script>
</body>
</html>"""
