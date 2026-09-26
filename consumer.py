"""Kafka Consumer for Real-Time Fraud Detection stream processing.

Subscribes to 'transactions-in', forwards each transaction to the FastAPI
/score endpoint for real-time scoring, publishes results to 'transactions-scored',
and writes an audit record to a sink file (data/scored_transactions.jsonl).

Usage:
    python consumer.py --dry-run --max-events 10
    python consumer.py --bootstrap-servers localhost:9092
    python consumer.py --api-url http://localhost:8000/score
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
import httpx

from src.data_prep import (
    DEFAULT_DATA_DIR,
    SAMPLE_DATA_PATH,
    generate_synthetic_sample,
    load_data,
)
from producer import format_transaction

# Load environment configuration
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Consumer] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("consumer")

DEFAULT_SINK_PATH = DEFAULT_DATA_DIR / "scored_transactions.jsonl"


def score_via_api(
    api_url: str,
    transaction: Dict[str, Any],
    client: Optional[httpx.Client] = None,
    timeout: float = 2.0,
) -> Dict[str, Any]:
    """Send transaction features to the FastAPI /score endpoint.

    Args:
        api_url: URL to the scoring endpoint (e.g. http://localhost:8000/score).
        transaction: Raw incoming transaction dictionary.
        client: Reusable httpx.Client instance (optional).
        timeout: Request timeout in seconds.

    Returns:
        Dictionary with 'risk_score', 'is_suspicious', 'threshold', and 'latency_ms'.
    """
    payload = {
        "transaction_id": transaction.get("transaction_id"),
        "amount": transaction["amount"],
        "features": transaction["features"],
        "timestamp": transaction.get("timestamp"),
    }

    own_client = False
    if client is None:
        client = httpx.Client(timeout=timeout)
        own_client = True

    try:
        response = client.post(api_url, json=payload)
        response.raise_for_status()
        return response.json()
    finally:
        if own_client:
            client.close()


def append_to_sink(record: Dict[str, Any], sink_path: Path | str) -> None:
    """Persist scored transaction record to a local JSONL sink file.

    Args:
        record: Complete scored transaction dictionary.
        sink_path: Filepath to the JSONL sink.
    """
    path = Path(sink_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def process_transaction(
    message: Dict[str, Any],
    api_url: str,
    http_client: Optional[httpx.Client] = None,
    producer: Any = None,
    topic_out: Optional[str] = None,
    sink_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Score transaction, publish to output topic, and persist to sink.

    Args:
        message: Decoded incoming transaction dictionary.
        api_url: URL of the scoring endpoint.
        http_client: Reusable HTTP client.
        producer: Optional KafkaProducer for output topic.
        topic_out: Kafka topic for scored results (e.g. 'transactions-scored').
        sink_path: Optional path to JSONL audit sink.

    Returns:
        Merged record with transaction features, risk score, and flags.
    """
    start_time = time.perf_counter()

    # Call scoring API
    score_result = score_via_api(api_url, message, client=http_client)

    e2e_latency_ms = (time.perf_counter() - start_time) * 1000.0

    scored_record = {
        "transaction_id": message.get("transaction_id", score_result.get("transaction_id")),
        "amount": message["amount"],
        "features": message["features"],
        "actual_class": message.get("actual_class", 0),
        "risk_score": score_result["risk_score"],
        "is_suspicious": score_result["is_suspicious"],
        "threshold": score_result["threshold"],
        "scored_at": score_result.get("scored_at", datetime.now(timezone.utc).isoformat()),
        "api_latency_ms": score_result.get("latency_ms", 0.0),
        "e2e_latency_ms": round(e2e_latency_ms, 3),
    }

    # 1. Publish to transactions-scored Kafka topic (if configured)
    if producer is not None and topic_out:
        key = scored_record["transaction_id"]
        producer.send(topic_out, key=key, value=scored_record)

    # 2. Append to audit sink file (if configured)
    if sink_path:
        append_to_sink(scored_record, sink_path)

    return scored_record


def create_kafka_consumer(
    bootstrap_servers: str,
    topic_in: str,
    group_id: str = "fraud-detection-consumers",
    max_retries: int = 12,
    retry_delay: float = 3.0,
):
    """Instantiate a KafkaConsumer client with automatic retry backoff."""
    from kafka import KafkaConsumer

    logger.info("Connecting to Kafka at %s (waiting for broker to be ready)...", bootstrap_servers)
    for attempt in range(1, max_retries + 1):
        try:
            consumer = KafkaConsumer(
                topic_in,
                bootstrap_servers=bootstrap_servers.split(","),
                group_id=group_id,
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                key_deserializer=lambda k: k.decode("utf-8") if k else None,
            )
            logger.info("Connected to Kafka brokers at %s, subscribed to '%s'", bootstrap_servers, topic_in)
            return consumer
        except Exception as exc:
            if attempt < max_retries:
                logger.warning(
                    "Kafka not ready yet at %s (%s). Retrying in %.0fs (attempt %d/%d)...",
                    bootstrap_servers,
                    exc,
                    retry_delay,
                    attempt,
                    max_retries,
                )
                time.sleep(retry_delay)
            else:
                logger.error("Failed to connect Kafka consumer after %d attempts: %s", max_retries, exc)
                raise


def run_consumer_loop(
    consumer: Any,
    api_url: str,
    producer: Any = None,
    topic_out: Optional[str] = "transactions-scored",
    sink_path: Optional[Path | str] = DEFAULT_SINK_PATH,
    max_events: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute consumer loop polling incoming messages and scoring them."""
    stats = {
        "processed": 0,
        "suspicious": 0,
        "total_latency_ms": 0.0,
    }

    http_client = httpx.Client(timeout=3.0)

    logger.info("Consumer loop started. Awaiting transactions...")

    try:
        for raw_msg in consumer:
            message = raw_msg.value if hasattr(raw_msg, "value") else raw_msg

            try:
                record = process_transaction(
                    message=message,
                    api_url=api_url,
                    http_client=http_client,
                    producer=producer,
                    topic_out=topic_out,
                    sink_path=sink_path,
                )

                stats["processed"] += 1
                if record["is_suspicious"]:
                    stats["suspicious"] += 1
                stats["total_latency_ms"] += record["e2e_latency_ms"]

                flag_symbol = "[SUSPICIOUS]" if record["is_suspicious"] else "[NORMAL]"
                logger.info(
                    "%s tx=%s amount=$%.2f risk=%.4f (threshold=%.2f) latency=%.2fms",
                    flag_symbol,
                    record["transaction_id"][:8],
                    record["amount"],
                    record["risk_score"],
                    record["threshold"],
                    record["e2e_latency_ms"],
                )

            except Exception as exc:
                logger.error("Error processing transaction: %s", exc)

            if max_events and stats["processed"] >= max_events:
                logger.info("Reached target limit of %d events.", max_events)
                break

    except KeyboardInterrupt:
        logger.info("Consumer stopped by user.")
    finally:
        http_client.close()

    avg_latency = (stats["total_latency_ms"] / stats["processed"]) if stats["processed"] > 0 else 0.0
    flag_rate = (stats["suspicious"] / stats["processed"] * 100.0) if stats["processed"] > 0 else 0.0

    logger.info("=" * 60)
    logger.info("CONSUMER SESSION SUMMARY:")
    logger.info("  - Processed:   %d transactions", stats["processed"])
    logger.info("  - Suspicious:  %d flagged (%.2f%%)", stats["suspicious"], flag_rate)
    logger.info("  - Avg Latency: %.2f ms", avg_latency)
    logger.info("=" * 60)

    return {
        "processed": stats["processed"],
        "suspicious": stats["suspicious"],
        "flag_rate_pct": round(flag_rate, 2),
        "avg_latency_ms": round(avg_latency, 2),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for transaction consumer."""
    parser = argparse.ArgumentParser(
        description="Consume transactions from Kafka, score via FastAPI, and write to sink."
    )
    parser.add_argument(
        "--bootstrap-servers",
        type=str,
        default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        help="Kafka bootstrap servers (default: localhost:9092).",
    )
    parser.add_argument(
        "--topic-in",
        type=str,
        default=os.getenv("KAFKA_TOPIC_IN", "transactions-in"),
        help="Input topic to consume from (default: transactions-in).",
    )
    parser.add_argument(
        "--topic-out",
        type=str,
        default=os.getenv("KAFKA_TOPIC_OUT", "transactions-scored"),
        help="Output topic for scored transactions (default: transactions-scored).",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=os.getenv("SCORING_API_URL", "http://localhost:8000/score"),
        help="Scoring API endpoint (default: http://localhost:8000/score).",
    )
    parser.add_argument(
        "--group-id",
        type=str,
        default=os.getenv("KAFKA_GROUP_ID", "fraud-detection-consumers"),
        help="Consumer group ID (default: fraud-detection-consumers).",
    )
    parser.add_argument(
        "--sink-path",
        type=str,
        default=os.getenv("SINK_PATH", str(DEFAULT_SINK_PATH)),
        help=f"Filepath to write scored transactions log (default: {DEFAULT_SINK_PATH}).",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Stop after consuming N events (default: run continuously).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate consumption from sample dataset without requiring live Kafka broker.",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function for consumer."""
    args = parse_args()

    # Dry-run simulation mode (runs without live Kafka cluster)
    if args.dry_run:
        logger.info("Running in --dry-run simulation mode (no Kafka cluster required).")
        if not SAMPLE_DATA_PATH.exists():
            generate_synthetic_sample(output_path=SAMPLE_DATA_PATH, n_samples=100)
        df = load_data(SAMPLE_DATA_PATH)
        limit = args.max_events or 10

        simulated_messages = []
        for i in range(min(limit, len(df))):
            is_fraud = (i == 2)  # inject sample fraud for demonstration
            simulated_messages.append(format_transaction(df.iloc[i], is_injected_fraud=is_fraud))

        run_consumer_loop(
            consumer=simulated_messages,
            api_url=args.api_url,
            producer=None,
            topic_out=None,
            sink_path=args.sink_path,
            max_events=limit,
        )
        return

    # Live Kafka connection mode
    consumer = None
    producer = None
    try:
        from producer import create_producer

        consumer = create_kafka_consumer(
            bootstrap_servers=args.bootstrap_servers,
            topic_in=args.topic_in,
            group_id=args.group_id,
        )
        producer = create_producer(args.bootstrap_servers)
    except Exception:
        logger.warning(
            "Could not connect to Kafka at '%s'. Falling back to --dry-run simulation.",
            args.bootstrap_servers,
        )
        args.dry_run = True
        main()
        return

    run_consumer_loop(
        consumer=consumer,
        api_url=args.api_url,
        producer=producer,
        topic_out=args.topic_out,
        sink_path=args.sink_path,
        max_events=args.max_events,
    )


if __name__ == "__main__":
    main()