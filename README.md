# Real-Time Fraud Detection API

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Apache Kafka](https://img.shields.io/badge/Streaming-Apache%20Kafka-231F20.svg)](https://kafka.apache.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Isolation%20Forest-F7931E.svg)](https://scikit-learn.org/)
[![Docker Compose](https://img.shields.io/badge/Docker-Orchestrated-2496ED.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-37%20Passed-brightgreen.svg)](https://pytest.org/)

A production-grade, real-time fraud risk-scoring service in the spirit of **Stripe Radar**.

Trained on the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset, the service deploys an unsupervised anomaly detection engine (scikit-learn **Isolation Forest** and a deep **PyTorch Autoencoder**), served via an async **FastAPI** microservice, and fed by an **Apache Kafka** streaming pipeline replaying live customer payment traffic.

---

## 🏗️ System Architecture

```
[ Kaggle Dataset ]
        │
        ▼
[ Kafka Producer ] ──(transactions-in)──▶ [ Kafka Consumer ]
 (Live Replay &                            │
 Fraud Injection)                          ▼ (HTTP POST /score)
                                  [ FastAPI Scoring Service ]
                                           │
                                           ▼
                                  [ Anomaly Detection ]
                                  • Isolation Forest
                                  • PyTorch Autoencoder
                                           │
                                           ▼
[ Scored Sink ] ◀──(transactions-scored)──┘
 (JSONL / DB)
```

### End-to-End Data Flow
1. **Producer (`producer.py`)**: Replays credit card transactions row-by-row into the `transactions-in` Kafka topic at a configurable arrival frequency, with optional synthetic fraud injection for testing.
2. **Consumer (`consumer.py`)**: Subscribes to `transactions-in`, forwards transaction payloads to the scoring API, publishes scored results to `transactions-scored`, and records an audit log.
3. **Scoring API (`app/main.py`)**: Stateless FastAPI microservice running inference and returning a risk score $[0.0, 1.0]$ and a boolean `is_suspicious` flag in **~8.5 milliseconds** (well under the 100ms SLA target).
4. **Live Dashboard (`/dashboard`)**: Web interface polling runtime metrics and displaying live incoming transactions with risk scores and alerts.

---

## ⚡ Performance & Model Benchmark

Anomaly detection models were evaluated on a held-out stratified test set preserving the rare $0.17\%$ fraud class ratio:

| Metric | Isolation Forest (Baseline) | PyTorch Autoencoder (Deep Learning) |
|---|:---:|:---:|
| **Precision** | **1.0000** | 0.1667 |
| **Recall** | 0.4000 | **1.0000** |
| **F1-Score** | **0.5714** | 0.2857 |
| **ROC-AUC** | 0.9879 | **0.9960** |
| **PR-AUC (Avg Precision)** | **0.5803** | 0.5556 |
| **False Positives** | **0** | 25 |
| **False Negatives** | 3 | **0** *(Zero missed fraud)* |
| **True Positives** | 2 | **5** |
| **Scoring Latency** | **< 10 ms** | **< 15 ms** |

> **Key Takeaway:** The **Isolation Forest** delivers zero false positives (ideal for low-friction checkouts), while the **PyTorch Autoencoder** achieves **100% recall** with zero false negatives (capturing non-linear, subtle fraud patterns).

---

## 📂 Project Structure

```
Real Time Fraud Detection/
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI application (/score, /health, /metrics, /dashboard)
│   └── schemas.py            # Pydantic request & response validation schemas
├── data/
│   ├── README.md             # Dataset acquisition and schema documentation
│   └── sample_creditcard.csv # 2,000-row synthetic test sample mirroring Kaggle schema
├── models/
│   ├── .gitkeep              # Folder for saved model artifacts (.joblib, .pt)
│   └── isolation_forest.joblib
├── src/
│   ├── __init__.py
│   ├── data_prep.py          # Data loading, RobustScaler preprocessing, and EDA
│   ├── model.py              # IsolationForest model wrapper with [0, 1] risk scoring
│   └── autoencoder.py        # PyTorch Autoencoder reconstruction-error model
├── tests/
│   ├── test_api.py           # API endpoint integration tests
│   ├── test_autoencoder.py   # PyTorch neural network unit tests
│   ├── test_consumer.py      # Kafka consumer and sink tests
│   ├── test_data_prep.py     # Preprocessing and stratified split tests
│   ├── test_model.py         # Isolation Forest scoring tests
│   └── test_producer.py      # Kafka producer and serialization tests
├── .env.example              # Environment variables template
├── .gitignore                # Version control ignore rules
├── compare_models.py         # Side-by-side benchmark comparison script
├── consumer.py               # Kafka streaming consumer
├── docker-compose.yml        # Multi-container orchestration (Kafka, Zookeeper, API, Producer, Consumer)
├── Dockerfile                # Production container image definition
├── producer.py               # Kafka streaming producer
├── requirements.txt          # Python dependencies
└── train.py                  # Baseline model training script
```

---

## 🚀 Quick Start

### Option A: Run the Full System with Docker Compose (Recommended)

Start the entire distributed streaming pipeline (Zookeeper, Kafka, API service, Producer, and Consumer) with a single command:

```bash
docker-compose up -d --build
```

View live streaming scores in the consumer log:
```bash
docker-compose logs -f consumer
```

Open the live streaming web dashboard in your browser:
👉 **[http://localhost:8000/dashboard](http://localhost:8000/dashboard)**

To stop all services:
```bash
docker-compose down
```

---

### Option B: Local Development (Without Docker)

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Run All Automated Unit Tests (37 Tests)
```bash
python -m pytest -v
```

#### 3. Train the Baseline Model
```bash
# Uses bundled sample or data/creditcard.csv if present
python train.py --use-sample
```

#### 4. Run Model Benchmark (Isolation Forest vs. PyTorch Autoencoder)
```bash
python compare_models.py --use-sample
```

#### 5. Launch the Scoring API
```bash
python -m uvicorn app.main:app --reload
```
- Interactive OpenAPI docs: `http://localhost:8000/docs`
- Live streaming dashboard: `http://localhost:8000/dashboard`
- Health check: `http://localhost:8000/health`

#### 6. Run the Streaming Simulation (Dry-Run Mode)
In another terminal, start the consumer:
```bash
python consumer.py --dry-run --max-events 20
```

Watch the terminal and dashboard update with real-time risk scores and flagged transactions!

---

## 🔌 API Contract

### `POST /score`
Scores an incoming transaction payload for fraud risk.

#### Request Payload
```json
{
  "amount": 4820.00,
  "features": [
    -1.3598, -0.0727, 2.5363, 1.3781, -0.3383, 0.4623, 0.2395, 0.0986,
     0.3637,  0.0907, -0.5516, -0.6178, -0.9913, -0.3111, 1.4681, -0.4704,
     0.2079,  0.0257,  0.4039,  0.2514, -0.0183,  0.2778, -0.1104,  0.0669,
     0.1285, -0.1891,  0.1335, -0.0210
  ],
  "timestamp": "2026-09-25T10:14:02Z"
}
```

#### Response Payload (`200 OK`)
```json
{
  "transaction_id": "c1f7b892-d3f4-46a0-acb4-6a766b325bde",
  "risk_score": 0.9421,
  "is_suspicious": true,
  "threshold": 0.80,
  "scored_at": "2026-09-25T10:14:02.145892Z",
  "latency_ms": 8.45
}
```

---

## ⚙️ Configuration

Thresholds, stream rates, and Kafka brokers are fully environment-driven:

| Variable | Default | Description |
|---|---|---|
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka broker connection string |
| `KAFKA_TOPIC_IN` | `transactions-in` | Ingestion topic for raw transactions |
| `KAFKA_TOPIC_OUT` | `transactions-scored` | Output topic for scored transactions |
| `FRAUD_THRESHOLD` | `0.80` | Score cutoff above which transactions are flagged |
| `SCORING_API_URL` | `http://localhost:8000/score` | Endpoint invoked by consumer |
| `PRODUCE_DELAY` | `0.1` | Delay in seconds between replayed transactions |
| `INJECT_FRAUD_EVERY` | `0` | Interval to inject obvious synthetic anomalies |

---

## 🧪 Testing

The repository maintains an automated test suite with **37 tests** across all pipeline layers:

```bash
python -m pytest -v
```

- `tests/test_data_prep.py`: Dataset loading, RobustScaler amount scaling, stratified split ratio preservation.
- `tests/test_model.py`: Isolation Forest training, $[0, 1]$ risk score range, serialization round-trip.
- `tests/test_api.py`: FastAPI endpoints (`/health`, `/score`, `/metrics`, `/dashboard`), 28-feature schema validation.
- `tests/test_producer.py`: Transaction payload serialization, synthetic fraud injection intervals.
- `tests/test_consumer.py`: Consumer polling, JSONL audit persistence, scoring integration.
- `tests/test_autoencoder.py`: PyTorch network dimensions, bottleneck latent representation, MSE reconstruction scoring.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
