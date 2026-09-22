# Real-Time Fraud Detection API

A fraud-detection service (Stripe Radar–style) that scores transactions for risk in real time. Trains an anomaly-detection model on the Kaggle Credit Card Fraud dataset, serves it via FastAPI, and streams transactions through Kafka to simulate live traffic.

## Problem

Fintech platforms need to flag fraudulent transactions in real time, not in batch, to block or hold bad transactions before settlement.

## Goal

Ingest a stream of transactions, score each for fraud risk, and return/log that score within milliseconds.

## Users

- Backend systems calling the API synchronously at transaction time
- Ops/analysts reviewing flagged transactions

## Architecture

```
[Kafka Producer] --> transactions-in (topic) --> [Kafka Consumer]
                                                        |
                                                        v
                                              [FastAPI /score]
                                                        |
                                                        v
                                          transactions-scored (topic) --> [Sink: DB / log]
```

## Scope (MVP)

- Train an anomaly-detection model (Isolation Forest, later autoencoder) on the Kaggle Credit Card Fraud dataset
- Serve the model via a FastAPI `/score` endpoint (single transaction in → risk score + flag out)
- Kafka producer to simulate a live transaction stream (replay dataset as if real-time)
- Kafka consumer that pulls transactions, calls the model, writes results to a sink (DB/log/second topic)
- Basic threshold-based flagging (e.g., score > X = "suspicious")

## Out of Scope (MVP)

- Real payment processor integration
- User-facing dashboard
- Model retraining pipeline / drift monitoring
- Multi-model ensemble

## Functional Requirements

1. **Data prep** — clean/normalize Kaggle dataset, handle class imbalance
2. **Model** — train Isolation Forest baseline; benchmark against autoencoder reconstruction-error approach
3. **API** — `POST /score`: input transaction features, output `{risk_score, is_suspicious}`
4. **Streaming** — Kafka topic `transactions-in`; producer replays dataset at configurable rate; consumer scores and publishes to `transactions-scored`
5. **Persistence** — store scored transactions (Postgres or flat file/log for MVP)
6. **Latency target** — sub-100ms per scoring call

## Non-Functional Requirements

- Dockerized services (API, consumer, Kafka via docker-compose)
- Config-driven thresholds (no hardcoding)
- Basic logging/metrics (scores over time, flag rate)

## Tech Stack

- **Language/API:** Python, FastAPI
- **Modeling:** scikit-learn (Isolation Forest), PyTorch/TensorFlow (autoencoder)
- **Streaming:** Kafka
- **Infra:** Docker, docker-compose
- **Storage:** Postgres (optional for MVP)
- **Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Getting Started

```bash
# clone repo
git clone <repo-url>
cd fraud-detection-api

# start Kafka + services
docker-compose up -d

# train model
python train.py

# run API
uvicorn app.main:app --reload

# start producer (simulated live transactions)
python producer.py

# start consumer (scores stream)
python consumer.py
```

## Success Metrics

- **Model:** precision/recall/AUC on held-out test set (fraud is rare — recall matters most)
- **System:** end-to-end latency from stream ingestion to score written
- **Demo:** visibly flags injected "obviously fraudulent" synthetic transactions in the live replay

## Milestones

1. EDA + baseline Isolation Forest model, offline eval
2. FastAPI wrapper, tested with static requests
3. Kafka producer/consumer wired in, dataset replay working
4. End-to-end demo + architecture diagram
5. *(Stretch)* Autoencoder comparison, simple metrics dashboard

## Known Limitations / Risks

- Kaggle dataset is static/labeled — real fraud patterns drift; this is a known MVP limitation, not solved here
- Class imbalance can make naive accuracy misleading — precision/recall are reported instead of accuracy

## License

MIT
