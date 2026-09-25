"""Pydantic request and response schemas for Fraud Detection Scoring API."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class TransactionScoreRequest(BaseModel):
    """Transaction scoring payload matching design-doc specifications."""

    transaction_id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the transaction (UUID). Auto-generated if omitted.",
        examples=["9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d"],
    )
    amount: float = Field(
        ...,
        ge=0.0,
        description="Transaction amount in original currency.",
        examples=[4820.00],
    )
    features: List[float] = Field(
        ...,
        description="28 PCA-anonymized numeric features (V1 through V28).",
        examples=[[-1.3598, -0.0727, 2.5363, 1.3781] + [0.0] * 24],
    )
    timestamp: Optional[str] = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Transaction occurrence timestamp in ISO 8601 format.",
        examples=["2026-09-22T10:14:02Z"],
    )

    @field_validator("features")
    @classmethod
    def validate_features_length(cls, v: List[float]) -> List[float]:
        """Ensure exactly 28 PCA features are provided."""
        if len(v) != 28:
            raise ValueError(f"Features array must contain exactly 28 PCA values (V1..V28), got {len(v)}.")
        return v


class TransactionScoreResponse(BaseModel):
    """Scoring result returned by the API."""

    transaction_id: str = Field(description="Unique identifier of scored transaction.")
    risk_score: float = Field(
        description="Calculated fraud risk score ranging from 0.0 (safe) to 1.0 (anomalous).",
        examples=[0.93],
    )
    is_suspicious: bool = Field(
        description="Flag indicating whether risk_score exceeds the decision threshold.",
        examples=[True],
    )
    threshold: float = Field(
        description="Configured decision threshold used to flag suspicious transactions.",
        examples=[0.85],
    )
    scored_at: str = Field(
        description="ISO 8601 timestamp when scoring was performed.",
        examples=["2026-09-22T10:14:02.123456Z"],
    )
    latency_ms: float = Field(
        description="Inference time in milliseconds.",
        examples=[1.42],
    )


class HealthResponse(BaseModel):
    """Liveness and readiness check response."""

    status: str = Field(description="Service health status.", examples=["ok"])
    model_loaded: bool = Field(description="Whether the fraud detection model is active.", examples=[True])
    threshold: float = Field(description="Configured decision threshold.", examples=[0.80])
    version: str = Field(description="API service version.", examples=["1.0.0"])


class MetricsResponse(BaseModel):
    """Observability metrics tracking request count, flags, and latency."""

    total_requests: int
    suspicious_count: int
    flag_rate_pct: float
    avg_latency_ms: float
