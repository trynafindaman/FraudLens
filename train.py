"""Training script for Baseline Isolation Forest Fraud Detection Model.

Loads transaction dataset, trains the model, runs offline evaluation,
and serializes the trained artifact to models/isolation_forest.joblib.

Usage:
    python train.py
    python train.py --use-sample
    python train.py --data-path data/creditcard.csv --threshold 0.85
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.data_prep import (
    DEFAULT_DATA_PATH,
    SAMPLE_DATA_PATH,
    generate_synthetic_sample,
    get_stratified_split,
    load_data,
    preprocess_data,
)
from src.model import DEFAULT_MODEL_PATH, FraudDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train baseline Isolation Forest model for Real-Time Fraud Detection."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help=f"Path to transactions CSV (default: {DEFAULT_DATA_PATH}).",
    )
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use data/sample_creditcard.csv instead of full creditcard.csv.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in Isolation Forest (default: 100).",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.0017,
        help="Contamination parameter: expected fraud fraction (default: 0.0017).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Risk score threshold to flag suspicious transactions (default: 0.80).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction for stratified evaluation (default: 0.2).",
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help=f"Target path to save trained model (default: {DEFAULT_MODEL_PATH}).",
    )
    return parser.parse_args()


def main() -> None:
    """Execute training pipeline."""
    args = parse_args()

    data_path = Path(SAMPLE_DATA_PATH if args.use_sample else args.data_path)

    # If data does not exist, provide instructions and fallback to generated sample
    if not data_path.exists():
        if data_path == DEFAULT_DATA_PATH:
            logger.info("Full Kaggle 'creditcard.csv' not found.")
            logger.info("Auto-generating a sample dataset for immediate execution...")
            generate_synthetic_sample(
                output_path=SAMPLE_DATA_PATH, n_samples=3000, fraud_ratio=0.01
            )
            data_path = SAMPLE_DATA_PATH
        else:
            raise FileNotFoundError(f"Specified dataset does not exist: {data_path}")

    # 1. Load Data
    logger.info("Step 1: Loading transaction data from %s ...", data_path)
    df = load_data(data_path)

    # 2. Preprocess & Scale
    logger.info("Step 2: Preprocessing features and scaling Amount...")
    X, y, scaler = preprocess_data(df, fit_scaler=True)

    # Adjust contamination if using small sample with different fraud ratio
    empirical_fraud_rate = float(y.mean())
    if args.use_sample or data_path == SAMPLE_DATA_PATH:
        contamination = min(max(empirical_fraud_rate, 0.001), 0.05)
    else:
        contamination = args.contamination

    # 3. Stratified Train/Test Split
    logger.info("Step 3: Creating stratified train/test split (test_size=%.2f)...", args.test_size)
    X_train, X_test, y_train, y_test = get_stratified_split(X, y, test_size=args.test_size)

    # 4. Train Isolation Forest
    logger.info("Step 4: Training Isolation Forest model...")
    detector = FraudDetector(
        contamination=contamination,
        n_estimators=args.n_estimators,
        random_state=42,
        threshold=args.threshold,
    )
    detector.fit(X_train, scaler=scaler)

    # 5. Offline Evaluation on Held-Out Test Set
    logger.info("Step 5: Evaluating model on held-out test split...")
    metrics = detector.evaluate(X_test, y_test, threshold=args.threshold, verbose=True)

    # 6. Save Model Artifact
    logger.info("Step 6: Saving model bundle...")
    saved_path = detector.save(args.model_output)
    print(f"Model saved successfully to: {saved_path}")

    # 7. Verification: Test Single Transaction Scoring (matching API contract)
    print("-" * 60)
    print("TESTING SINGLE TRANSACTION INFERENCE (API CONTRACT VERIFICATION):")
    sample_normal_features = [0.1] * 28
    sample_fraud_features = [3.5 if i % 4 == 0 else -3.0 for i in range(28)]

    normal_result = detector.predict_single(amount=25.50, features=sample_normal_features)
    fraud_result = detector.predict_single(amount=4820.00, features=sample_fraud_features)

    norm_score = normal_result['risk_score']
    norm_flag = normal_result['is_suspicious']
    fraud_score = fraud_result['risk_score']
    fraud_flag = fraud_result['is_suspicious']
    print(f"Normal transaction ($25.50):   score={norm_score:.4f}, suspicious={norm_flag}")
    print(f"Suspicious transaction ($4820): score={fraud_score:.4f}, suspicious={fraud_flag}")
    print("=" * 60)


if __name__ == "__main__":
    main()
