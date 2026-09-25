"""Kafka Producer for Real-Time Fraud Detection stream simulation.

Replays transactions from the Kaggle dataset row-by-row onto the Kafka topic
'transactions-in' with a configurable arrival delay.

Usage:
    python producer.py --dry-run --max-events 10
    python producer.py --use-sample --delay 0.05
    python producer.py --bootstrap-servers localhost:9092 --topic transactions-in
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import pandas as pd
from dotenv import load_dotenv

from src.data_prep import (
    AMOUNT_COL,
    CLASS_COL,
    DEFAULT_DATA_PATH,
    SAMPLE_DATA_PATH,
    V_COLS,
    generate_synthetic_sample,
    load_data,
)

# Load environment configuration
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Producer] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("producer")


def format_transaction(
    row: pd.Series | Dict[str, Any],
    tx_id: Optional[str] = None,
    is_injected_fraud: bool = False,
) -> Dict[str, Any]:
    """Format a dataset row into the standardized Kafka transaction payload.

    Args:
        row: Series or dictionary containing Kaggle transaction features.
        tx_id: Optional UUID string override.
        is_injected_fraud: Whether to inject obviously anomalous feature values.

    Returns:
        Structured dictionary matching Section 8 Data Model.
    """
    transaction_id = tx_id or str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    if is_injected_fraud:
        # Obvious fraud injection for demo / verification
        amount = 9999.99
        features = [5.0 if i % 2 == 0 else -5.0 for i in range(28)]
        actual_class = 1
    else:
        amount = float(row[AMOUNT_COL])
        features = [float(row[col]) for col in V_COLS]
        actual_class = int(row[CLASS_COL]) if CLASS_COL in row else 0

    return {
        "transaction_id": transaction_id,
        "amount": round(amount, 2),
        "features": features,
        "timestamp": timestamp,
        "actual_class": actual_class,
    }


def create_producer(bootstrap_servers: str):
    """Instantiate a KafkaProducer client.

    Args:
        bootstrap_servers: Comma-separated list of host:port Kafka brokers.

    Returns:
        Configured KafkaProducer instance.
    """
    try:
        from kafka import KafkaProducer

        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers.split(","),
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",
            retries=3,
        )
        logger.info("Connected to Kafka brokers at %s", bootstrap_servers)
        return producer
    except Exception as exc:
        logger.error("Failed to connect to Kafka at %s: %s", bootstrap_servers, exc)
        raise


def stream_transactions(
    df: pd.DataFrame,
    producer: Any = None,
    topic: str = "transactions-in",
    delay: float = 0.1,
    max_events: Optional[int] = None,
    dry_run: bool = False,
    inject_fraud_every: int = 0,
) -> int:
    """Stream dataset rows onto Kafka topic or stdout.

    Args:
        df: DataFrame of transactions to replay.
        producer: KafkaProducer client or None if dry_run.
        topic: Kafka topic name.
        delay: Delay in seconds between replayed transactions.
        max_events: Maximum number of events to publish (None for entire df).
        dry_run: If True, validates and logs messages without connecting to Kafka.
        inject_fraud_every: If > 0, injects synthetic extreme fraud every N events.

    Returns:
        Number of events produced.
    """
    events_count = 0
    total_available = len(df)
    limit = max_events if max_events is not None else total_available

    logger.info(
        "Starting stream: %d events scheduled (delay=%.3fs, dry_run=%s, topic=%s)",
        limit,
        delay,
        dry_run,
        topic,
    )

    try:
        for idx in range(limit):
            row = df.iloc[idx % total_available]
            is_injected = (inject_fraud_every > 0) and ((idx + 1) % inject_fraud_every == 0)

            message = format_transaction(row, is_injected_fraud=is_injected)

            if dry_run:
                logger.info(
                    "[DRY-RUN] [%d/%d] tx=%s amount=$%.2f fraud=%s (class=%d)",
                    idx + 1,
                    limit,
                    message["transaction_id"][:8],
                    message["amount"],
                    "INJECTED" if is_injected else ("YES" if message["actual_class"] == 1 else "NO"),
                    message["actual_class"],
                )
            else:
                if producer is None:
                    raise RuntimeError("KafkaProducer instance is required when dry_run=False.")
                key = message["transaction_id"]
                producer.send(topic, key=key, value=message)
                logger.info(
                    "[SENT] [%d/%d] tx=%s amount=$%.2f",
                    idx + 1,
                    limit,
                    key[:8],
                    message["amount"],
                )

            events_count += 1

            if delay > 0 and idx < limit - 1:
                time.sleep(delay)

        if not dry_run and producer is not None:
            producer.flush()
            logger.info("Flushed all messages to topic '%s'.", topic)

    except KeyboardInterrupt:
        logger.info("Stream interrupted by user after %d messages.", events_count)

    logger.info("Streaming completed. Total events produced: %d", events_count)
    return events_count


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for transaction producer."""
    parser = argparse.ArgumentParser(
        description="Stream transactions from dataset to Kafka topic 'transactions-in'."
    )
    parser.add_argument(
        "--bootstrap-servers",
        type=str,
        default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        help="Kafka bootstrap servers (default: localhost:9092 or $KAFKA_BOOTSTRAP_SERVERS).",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=os.getenv("KAFKA_TOPIC_IN", "transactions-in"),
        help="Target Kafka topic (default: transactions-in or $KAFKA_TOPIC_IN).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help=f"Path to CSV dataset (default: {DEFAULT_DATA_PATH}).",
    )
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use data/sample_creditcard.csv.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=float(os.getenv("PRODUCE_DELAY", "0.1")),
        help="Delay in seconds between replayed transactions (default: 0.1s).",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Maximum events to publish (default: all rows in dataset).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate stream and print formatted messages without requiring Kafka broker.",
    )
    parser.add_argument(
        "--inject-fraud-every",
        type=int,
        default=int(os.getenv("INJECT_FRAUD_EVERY", "0")),
        help="Inject an obvious synthetic fraudulent transaction every N events (0 to disable).",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function for producer."""
    args = parse_args()

    # Determine data source
    data_path = Path(SAMPLE_DATA_PATH if args.use_sample else args.data_path)
    if not data_path.exists():
        if data_path == DEFAULT_DATA_PATH:
            logger.info("Full Kaggle dataset not found at %s. Bootstrapping sample dataset...", data_path)
            generate_synthetic_sample(output_path=SAMPLE_DATA_PATH, n_samples=2000, fraud_ratio=0.01)
            data_path = SAMPLE_DATA_PATH
        else:
            logger.error("Dataset not found at %s", data_path)
            sys.exit(1)

    # Load dataset
    df = load_data(data_path)

    # Create Kafka producer if not in dry-run mode
    producer = None
    if not args.dry_run:
        try:
            producer = create_producer(args.bootstrap_servers)
        except Exception:
            logger.warning(
                "Unable to connect to Kafka at '%s'. "
                "Running in --dry-run mode for local demonstration.",
                args.bootstrap_servers,
            )
            args.dry_run = True

    # Replay stream
    stream_transactions(
        df=df,
        producer=producer,
        topic=args.topic,
        delay=args.delay,
        max_events=args.max_events,
        dry_run=args.dry_run,
        inject_fraud_every=args.inject_fraud_every,
    )


if __name__ == "__main__":
    main()
