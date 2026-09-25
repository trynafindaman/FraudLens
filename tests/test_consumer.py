"""Tests for Kafka consumer and scoring loop."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from consumer import append_to_sink, process_transaction, run_consumer_loop


@pytest.fixture
def mock_transaction():
    """Sample incoming transaction."""
    return {
        "transaction_id": "tx-test-consumer-001",
        "amount": 250.00,
        "features": [0.1] * 28,
        "timestamp": "2026-09-25T10:00:00Z",
        "actual_class": 0,
    }


def test_append_to_sink(tmp_path: Path):
    """Verify sink appends json records."""
    sink_file = tmp_path / "sink.jsonl"
    record = {"transaction_id": "tx-1", "risk_score": 0.45, "is_suspicious": False}
    append_to_sink(record, sink_file)
    append_to_sink({"transaction_id": "tx-2", "risk_score": 0.95, "is_suspicious": True}, sink_file)

    lines = sink_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["transaction_id"] == "tx-1"
    assert json.loads(lines[1])["is_suspicious"] is True


@patch("consumer.score_via_api")
def test_process_transaction(mock_score, mock_transaction, tmp_path: Path):
    """Verify process_transaction calls scoring, publishes output, and saves sink."""
    mock_score.return_value = {
        "transaction_id": "tx-test-consumer-001",
        "risk_score": 0.88,
        "is_suspicious": True,
        "threshold": 0.80,
        "scored_at": "2026-09-25T10:00:01Z",
        "latency_ms": 12.5,
    }
    mock_producer = MagicMock()
    sink_file = tmp_path / "test_sink.jsonl"

    result = process_transaction(
        message=mock_transaction,
        api_url="http://localhost:8000/score",
        producer=mock_producer,
        topic_out="transactions-scored",
        sink_path=sink_file,
    )

    assert result["risk_score"] == 0.88
    assert result["is_suspicious"] is True
    assert mock_producer.send.called
    assert sink_file.exists()


@patch("consumer.score_via_api")
def test_run_consumer_loop(mock_score, mock_transaction, tmp_path: Path):
    """Verify run_consumer_loop processes all messages and returns metrics."""
    mock_score.return_value = {
        "risk_score": 0.20,
        "is_suspicious": False,
        "threshold": 0.80,
        "scored_at": "2026-09-25T10:00:01Z",
        "latency_ms": 5.0,
    }
    messages = [mock_transaction, mock_transaction]
    sink_file = tmp_path / "test_sink.jsonl"

    stats = run_consumer_loop(
        consumer=messages,
        api_url="http://localhost:8000/score",
        sink_path=sink_file,
        max_events=2,
    )

    assert stats["processed"] == 2
    assert stats["suspicious"] == 0