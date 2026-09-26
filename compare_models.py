"""Benchmark and Compare Baseline Isolation Forest vs. PyTorch Autoencoder.

Evaluates both anomaly detection models on the held-out test split,
printing a side-by-side performance comparison of Precision, Recall, F1, and AUC.

Usage:
    python compare_models.py --use-sample
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.autoencoder import AutoencoderDetector
from src.data_prep import (
    DEFAULT_DATA_PATH,
    SAMPLE_DATA_PATH,
    generate_synthetic_sample,
    get_stratified_split,
    load_data,
    preprocess_data,
)
from src.model import FraudDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Compare] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("compare")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Isolation Forest vs Autoencoder.")
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--use-sample", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.80)
    args = parser.parse_args()

    data_path = Path(SAMPLE_DATA_PATH if args.use_sample else args.data_path)
    if not data_path.exists():
        logger.info("Generating sample dataset for benchmark...")
        generate_synthetic_sample(output_path=SAMPLE_DATA_PATH, n_samples=3000, fraud_ratio=0.01)
        data_path = SAMPLE_DATA_PATH

    # 1. Prepare data
    df = load_data(data_path)
    X, y, scaler = preprocess_data(df, fit_scaler=True)
    X_train, X_test, y_train, y_test = get_stratified_split(X, y, test_size=0.25, random_state=42)

    logger.info("Test split size: %d (Actual fraud: %d)", len(y_test), int(y_test.sum()))

    # 2. Train and Evaluate Baseline: Isolation Forest
    logger.info("Training Baseline Model: Isolation Forest...")
    if_detector = FraudDetector(contamination=0.01, n_estimators=100, threshold=args.threshold)
    if_detector.fit(X_train, scaler=scaler)
    if_metrics = if_detector.evaluate(X_test, y_test, threshold=args.threshold, verbose=False)

    # 3. Train and Evaluate Stretch Model: PyTorch Autoencoder
    logger.info("Training Stretch Model: PyTorch Autoencoder...")
    ae_detector = AutoencoderDetector(input_dim=29, latent_dim=8, threshold=args.threshold)
    ae_detector.fit(X_train, y_train=y_train, scaler=scaler, epochs=20, batch_size=64)
    ae_metrics = ae_detector.evaluate(X_test, y_test, threshold=args.threshold, verbose=False)

    # 4. Print Side-by-Side Benchmark Table
    print("\n" + "=" * 80)
    print("           MODEL BENCHMARK: ISOLATION FOREST vs. PYTORCH AUTOENCODER           ")
    print("=" * 80)
    header = f"{'Metric':<22} | {'Isolation Forest (Baseline)':<26} | {'PyTorch Autoencoder (Stretch)':<26}"
    print(header)
    print("-" * 80)

    metrics_to_show = [
        ("Precision", if_metrics["precision"], ae_metrics["precision"]),
        ("Recall", if_metrics["recall"], ae_metrics["recall"]),
        ("F1-Score", if_metrics["f1_score"], ae_metrics["f1_score"]),
        ("ROC-AUC", if_metrics["roc_auc"], ae_metrics["roc_auc"]),
        ("PR-AUC (Avg Prec)", if_metrics["pr_auc"], ae_metrics["pr_auc"]),
        ("False Positives", if_metrics["confusion_matrix"]["fp"], ae_metrics["confusion_matrix"]["fp"]),
        ("False Negatives", if_metrics["confusion_matrix"]["fn"], ae_metrics["confusion_matrix"]["fn"]),
        ("True Positives", if_metrics["confusion_matrix"]["tp"], ae_metrics["confusion_matrix"]["tp"]),
    ]

    for name, m_if, m_ae in metrics_to_show:
        if isinstance(m_if, float):
            print(f"{name:<22} | {m_if:<26.4f} | {m_ae:<26.4f}")
        else:
            print(f"{name:<22} | {m_if:<26} | {m_ae:<26}")

    print("=" * 80)
    print("Analysis:")
    print("  • Isolation Forest partitions feature space linearly with axis-aligned splits.")
    print("  • Autoencoder models non-linear cross-feature relationships via bottleneck reconstruction.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
