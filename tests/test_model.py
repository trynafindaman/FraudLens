"""Tests for Isolation Forest model, risk scoring, evaluation, and serialization."""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.data_prep import generate_synthetic_sample, preprocess_data, get_stratified_split
from src.model import FraudDetector


@pytest.fixture
def trained_detector(tmp_path: Path):
    """Fixture providing a fitted FraudDetector on synthetic data."""
    csv_file = tmp_path / "train_sample.csv"
    df = generate_synthetic_sample(
        output_path=csv_file, n_samples=400, fraud_ratio=0.05, random_state=42
    )
    X, y, scaler = preprocess_data(df, fit_scaler=True)
    X_train, X_test, y_train, y_test = get_stratified_split(X, y, test_size=0.25, random_state=42)

    detector = FraudDetector(contamination=0.05, n_estimators=50, random_state=42, threshold=0.75)
    detector.fit(X_train, scaler=scaler)
    return detector, X_test, y_test


def test_unfitted_model_raises():
    """Verify predicting before fitting raises RuntimeError."""
    detector = FraudDetector()
    with pytest.raises(RuntimeError, match="not fitted"):
        detector.predict_risk(np.zeros((5, 29)))


def test_model_fit(trained_detector):
    """Verify model fits and sets appropriate attributes."""
    detector, _, _ = trained_detector
    assert detector.is_fitted
    assert detector.scaler is not None
    assert detector.min_score < detector.max_score


def test_model_predict_risk_range(trained_detector):
    """Verify output risk scores are within valid [0.0, 1.0] range."""
    detector, X_test, _ = trained_detector
    risk_scores = detector.predict_risk(X_test)

    assert len(risk_scores) == len(X_test)
    assert np.all(risk_scores >= 0.0)
    assert np.all(risk_scores <= 1.0)


def test_model_predict_flags(trained_detector):
    """Verify boolean suspicious flags correspond to threshold."""
    detector, X_test, _ = trained_detector
    risk_scores, is_suspicious = detector.predict(X_test, threshold=0.70)

    assert len(risk_scores) == len(X_test)
    assert len(is_suspicious) == len(X_test)
    assert np.array_equal(is_suspicious, risk_scores >= 0.70)


def test_predict_single_contract(trained_detector):
    """Verify single transaction prediction matches the design-doc API contract."""
    detector, _, _ = trained_detector
    features = [0.0] * 28

    result = detector.predict_single(amount=125.50, features=features, threshold=0.80)

    assert "risk_score" in result
    assert "is_suspicious" in result
    assert "threshold" in result
    assert isinstance(result["risk_score"], float)
    assert 0.0 <= result["risk_score"] <= 1.0
    assert isinstance(result["is_suspicious"], bool)
    assert result["threshold"] == 0.80


def test_predict_single_invalid_features(trained_detector):
    """Verify passing invalid number of features raises ValueError."""
    detector, _, _ = trained_detector
    with pytest.raises(ValueError, match="Expected 28 PCA features"):
        detector.predict_single(amount=50.0, features=[1.0, 2.0])


def test_model_evaluate(trained_detector):
    """Verify offline evaluation calculates precision, recall, F1, and AUC metrics."""
    detector, X_test, y_test = trained_detector
    metrics = detector.evaluate(X_test, y_test, threshold=0.70, verbose=False)

    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "confusion_matrix" in metrics

    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0

    cm = metrics["confusion_matrix"]
    assert cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"] == len(y_test)


def test_model_save_and_load(trained_detector, tmp_path: Path):
    """Verify serializing and reloading produces identical inference predictions."""
    detector, X_test, _ = trained_detector
    model_file = tmp_path / "test_model.joblib"

    # Save
    detector.save(model_file)
    assert model_file.exists()

    # Load
    loaded_detector = FraudDetector.load(model_file)
    assert loaded_detector.is_fitted

    # Predictions before and after reload must match
    orig_scores = detector.predict_risk(X_test)
    loaded_scores = loaded_detector.predict_risk(X_test)
    assert np.allclose(orig_scores, loaded_scores)

    # Test single transaction prediction matches
    features = [0.2] * 28
    orig_single = detector.predict_single(amount=99.0, features=features)
    loaded_single = loaded_detector.predict_single(amount=99.0, features=features)
    assert orig_single == loaded_single
