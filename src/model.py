"""Isolation Forest model for real-time fraud anomaly detection.

Implements model training, risk score normalization (0.0 to 1.0),
offline evaluation (Precision, Recall, F1, ROC-AUC, PR-AUC), and model persistence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler

from src.data_prep import FEATURE_COLS, V_COLS

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "isolation_forest.joblib"


class FraudDetector:
    """Wrapper around scikit-learn Isolation Forest for fraud detection scoring."""

    def __init__(
        self,
        contamination: float = 0.0017,
        n_estimators: int = 100,
        random_state: int = 42,
        threshold: float = 0.80,
    ) -> None:
        """Initialize FraudDetector.

        Args:
            contamination: Expected proportion of outliers in the data (~0.17% for Kaggle).
            n_estimators: Number of trees in the isolation forest.
            random_state: Random seed for reproducibility.
            threshold: Decision boundary on risk score (0.0 - 1.0) above which a transaction
                       is flagged as suspicious.
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.threshold = threshold

        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.scaler: RobustScaler | None = None
        self.min_score: float = 0.0
        self.max_score: float = 1.0
        self.feature_names: List[str] = list(FEATURE_COLS)
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame | np.ndarray, scaler: RobustScaler) -> "FraudDetector":
        """Fit Isolation Forest and calibrate score normalization range.

        Args:
            X: Feature matrix with 29 features (V1-V28 + scaled_amount).
            scaler: Pre-fitted RobustScaler used for scaling transaction amount.

        Returns:
            self
        """
        self.scaler = scaler
        X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

        logger.info(
            "Fitting IsolationForest (n_estimators=%d, contamination=%.4f) on %d samples...",
            self.n_estimators,
            self.contamination,
            X_mat.shape[0],
        )
        self.model.fit(X_mat)
        self.is_fitted = True

        # Compute raw anomaly scores for training set
        # score_samples returns negative anomaly score: lower = more anomalous
        # We invert so higher = more anomalous
        raw_scores = -self.model.score_samples(X_mat)
        self.min_score = float(np.min(raw_scores))
        self.max_score = float(np.max(raw_scores))

        logger.info(
            "Model fitted successfully. Raw score range: [%.4f, %.4f]",
            self.min_score,
            self.max_score,
        )
        return self

    def predict_risk(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Compute normalized fraud risk scores in range [0.0, 1.0].

        Higher risk score signifies a higher likelihood of fraud.

        Args:
            X: Feature matrix with 29 features.

        Returns:
            1D numpy array of risk scores in [0.0, 1.0].
        """
        if not self.is_fitted:
            raise RuntimeError("FraudDetector model is not fitted yet. Call fit() or load().")

        X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        raw_scores = -self.model.score_samples(X_mat)

        # Min-max normalization scaled to [0.0, 1.0] and clipped
        denom = (self.max_score - self.min_score) if (self.max_score > self.min_score) else 1.0
        risk_scores = np.clip((raw_scores - self.min_score) / denom, 0.0, 1.0)
        return np.round(risk_scores, 4)

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        threshold: float | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict risk scores and boolean suspicious flags.

        Args:
            X: Feature matrix.
            threshold: Optional custom threshold to override instance default.

        Returns:
            Tuple of (risk_scores: np.ndarray, is_suspicious: np.ndarray)
        """
        effective_threshold = self.threshold if threshold is None else threshold
        risk_scores = self.predict_risk(X)
        is_suspicious = risk_scores >= effective_threshold
        return risk_scores, is_suspicious

    def predict_single(
        self,
        amount: float,
        features: List[float] | np.ndarray,
        threshold: float | None = None,
    ) -> Dict[str, Any]:
        """Score a single incoming transaction matching API contract in design-doc.

        Args:
            amount: Transaction monetary amount.
            features: 28 PCA-anonymized features (V1 to V28).
            threshold: Optional threshold override.

        Returns:
            Dictionary with keys 'risk_score', 'is_suspicious', 'threshold'.
        """
        if not self.is_fitted or self.scaler is None:
            raise RuntimeError("FraudDetector is not fitted or missing scaler.")

        if len(features) != 28:
            raise ValueError(f"Expected 28 PCA features (V1..V28), received {len(features)}.")

        # Scale amount with fitted RobustScaler
        scaled_amount = float(self.scaler.transform([[amount]])[0, 0])

        # Combine 28 PCA features + scaled_amount = 29 features
        row = np.array(list(features) + [scaled_amount], dtype=float).reshape(1, -1)

        effective_threshold = self.threshold if threshold is None else threshold
        risk_score = float(self.predict_risk(row)[0])
        is_suspicious = bool(risk_score >= effective_threshold)

        return {
            "risk_score": risk_score,
            "is_suspicious": is_suspicious,
            "threshold": effective_threshold,
        }

    def evaluate(
        self,
        X_test: pd.DataFrame | np.ndarray,
        y_test: pd.Series | np.ndarray,
        threshold: float | None = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run offline evaluation on held-out test data.

        Calculates Precision, Recall, F1, ROC-AUC, PR-AUC, and Confusion Matrix.

        Args:
            X_test: Test features.
            y_test: True test binary labels (0 = normal, 1 = fraud).
            threshold: Threshold to use for binary decision (defaults to self.threshold).
            verbose: If True, prints formatted evaluation metrics to stdout.

        Returns:
            Dictionary containing evaluation metrics.
        """
        y_true = np.asarray(y_test)
        effective_threshold = self.threshold if threshold is None else threshold

        risk_scores = self.predict_risk(X_test)
        y_pred = (risk_scores >= effective_threshold).astype(int)

        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        # Check if both classes are present in y_true for AUC computation
        has_both_classes = len(np.unique(y_true)) > 1
        roc_auc = float(roc_auc_score(y_true, risk_scores)) if has_both_classes else 0.0
        pr_auc = float(average_precision_score(y_true, risk_scores)) if has_both_classes else 0.0

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        metrics = {
            "threshold": effective_threshold,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "total_test_samples": len(y_true),
            "actual_fraud_samples": int(np.sum(y_true)),
            "flagged_fraud_samples": int(np.sum(y_pred)),
        }

        if verbose:
            print("\n" + "=" * 60)
            print("         OFFLINE MODEL EVALUATION (ISOLATION FOREST)         ")
            print("=" * 60)
            print(f"Evaluation Threshold: {effective_threshold:.2f}")
            print(f"Test Set Size:        {len(y_true):,} (Actual Fraud: {int(np.sum(y_true)):,})")
            print(f"Flagged Suspicious:   {int(np.sum(y_pred)):,}")
            print("-" * 60)
            print("Metrics (Imbalance-focused):")
            print(f"  - Precision:        {precision:.4f}")
            print(f"  - Recall:           {recall:.4f}")
            print(f"  - F1-Score:         {f1:.4f}")
            print(f"  - ROC-AUC:          {roc_auc:.4f}")
            print(f"  - PR-AUC (Avg Prec):{pr_auc:.4f}")
            print("-" * 60)
            print("Confusion Matrix:")
            print(f"  - True Negatives  (Legit -> Legit):   {tn:,}")
            print(f"  - False Positives (Legit -> Flagged): {fp:,}")
            print(f"  - False Negatives (Fraud -> Missed):  {fn:,}")
            print(f"  - True Positives  (Fraud -> Flagged): {tp:,}")
            print("=" * 60 + "\n")

        return metrics

    def save(self, filepath: Path | str = DEFAULT_MODEL_PATH) -> Path:
        """Serialize and save the model, scaler, and metadata.

        Args:
            filepath: Target file path (.joblib).

        Returns:
            Path where the model bundle was saved.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        bundle = {
            "model": self.model,
            "scaler": self.scaler,
            "threshold": self.threshold,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "feature_names": self.feature_names,
        }
        joblib.dump(bundle, path)
        logger.info("Saved model artifact to %s", path)
        return path

    @classmethod
    def load(cls, filepath: Path | str = DEFAULT_MODEL_PATH) -> "FraudDetector":
        """Load a serialized FraudDetector artifact from disk.

        Args:
            filepath: Path to saved .joblib file.

        Returns:
            Loaded and ready-to-infer FraudDetector instance.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at '{path}'.")

        logger.info("Loading model artifact from %s ...", path)
        bundle = joblib.load(path)

        instance = cls(
            contamination=bundle.get("contamination", 0.0017),
            n_estimators=bundle.get("n_estimators", 100),
            random_state=bundle.get("random_state", 42),
            threshold=bundle.get("threshold", 0.80),
        )
        instance.model = bundle["model"]
        instance.scaler = bundle["scaler"]
        instance.min_score = bundle["min_score"]
        instance.max_score = bundle["max_score"]
        instance.feature_names = bundle.get("feature_names", list(FEATURE_COLS))
        instance.is_fitted = True

        logger.info("Model loaded successfully. Ready for inference.")
        return instance
