"""Tests for Kafka transaction stream producer."""

from unittest.mock import MagicMock
import pytest
import pandas as pd

from producer import format_transaction, stream_transactions
from src.data_prep import generate_synthetic_sample


@pytest.fixture
def sample_df(tmp_path):
    """Fixture providing a small transactions DataFrame."""
    csv_file = tmp_path / "test_stream.csv"
    df = generate_synthetic_sample(output_path=csv_file, n_samples=30, fraud_ratio=0.1, random_state=42)
    return df


def test_format_transaction_structure(sample_df):
    """Verify formatted transaction satisfies Section 8 Data Model."""
    row = sample_df.iloc[0]
    msg = format_transaction(row, tx_id="tx-custom-123")

    assert msg["transaction_id"] == "tx-custom-123"
    assert isinstance(msg["amount"], float)
    assert msg["amount"] >= 0.0
    assert isinstance(msg["features"], list)
    assert len(msg["features"]) == 28
    assert all(isinstance(v, float) for v in msg["features"])
    assert "timestamp" in msg
    assert msg["actual_class"] in [0, 1]


def test_format_transaction_injected_fraud(sample_df):
    """Verify injected fraud transactions have anomalous values and class 1."""
    row = sample_df.iloc[0]
    msg = format_transaction(row, is_injected_fraud=True)

    assert msg["actual_class"] == 1
    assert msg["amount"] == 9999.99
    assert len(msg["features"]) == 28
    assert msg["features"][0] == 5.0
    assert msg["features"][1] == -5.0


def test_stream_transactions_dry_run(sample_df):
    """Verify dry_run streams the exact count of requested events without error."""
    count = stream_transactions(
        df=sample_df,
        producer=None,
        topic="transactions-in",
        delay=0.0,
        max_events=12,
        dry_run=True,
    )
    assert count == 12


def test_stream_transactions_with_mock_producer(sample_df):
    """Verify producer sends messages to Kafka topic and flushes."""
    mock_producer = MagicMock()

    count = stream_transactions(
        df=sample_df,
        producer=mock_producer,
        topic="transactions-in",
        delay=0.0,
        max_events=7,
        dry_run=False,
    )

    assert count == 7
    assert mock_producer.send.call_count == 7
    mock_producer.flush.assert_called_once()

    # Check first call arguments
    call_args = mock_producer.send.call_args_list[0]
    topic_arg = call_args[0][0]
    assert topic_arg == "transactions-in"
    kwargs = call_args[1]
    assert "key" in kwargs
    assert "value" in kwargs
    assert len(kwargs["value"]["features"]) == 28


def test_stream_transactions_injected_fraud_interval(sample_df):
    """Verify synthetic fraud injection triggers at specified intervals."""
    mock_producer = MagicMock()

    stream_transactions(
        df=sample_df,
        producer=mock_producer,
        topic="transactions-in",
        delay=0.0,
        max_events=6,
        dry_run=False,
        inject_fraud_every=3,
    )

    # 3rd and 6th messages should have injected fraud
    messages = [call[1]["value"] for call in mock_producer.send.call_args_list]
    assert messages[0]["amount"] != 9999.99
    assert messages[1]["amount"] != 9999.99
    assert messages[2]["amount"] == 9999.99
    assert messages[2]["actual_class"] == 1
    assert messages[5]["amount"] == 9999.99
    assert messages[5]["actual_class"] == 1
