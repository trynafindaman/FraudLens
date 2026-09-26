"""Tests for PyTorch Autoencoder anomaly detector."""

from pathlib import Path
import pytest
import numpy as np
import torch

from src.autoencoder import AutoencoderDetector, AutoencoderNet
from src.data_prep import generate_synthetic_sample, preprocess_data, get_stratified_split


@pytest.fixture
def sample_dataset(tmp_path: Path):
    """Fixture providing train and test features."""
    csv_file = tmp_path / "test_ae.csv"
    df = generate_synthetic_sample(output_path=csv_file, n_samples=300, fraud_ratio=0.05, random_state=42)
    X, y, scaler = preprocess_data(df, fit_scaler=True)
    X_train, X_test, y_train, y_test = get_stratified_split(X, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test, scaler


def test_autoencoder_network_dimensions():
    """Verify PyTorch Autoencoder layer shapes."""
    net = AutoencoderNet(input_dim=29, latent_dim=8)
    dummy_input = torch.randn(10, 29)
    output = net(dummy_input)

    assert output.shape == (10, 29)


def test_autoencoder_fit_and_predict_risk(sample_dataset):
    """Verify Autoencoder training and risk scoring in range [0, 1]."""
    X_train, X_test, y_train, _, scaler = sample_dataset

    detector = AutoencoderDetector(input_dim=29, latent_dim=8, threshold=0.75)
    detector.fit(X_train, y_train=y_train, scaler=scaler, epochs=5, batch_size=32)

    assert detector.is_fitted
    risk_scores = detector.predict_risk(X_test)

    assert len(risk_scores) == len(X_test)
    assert np.all(risk_scores >= 0.0)
    assert np.all(risk_scores <= 1.0)


def test_autoencoder_unfitted_raises():
    """Verify predicting on unfitted detector raises RuntimeError."""
    detector = AutoencoderDetector(input_dim=29)
    with pytest.raises(RuntimeError, match="not fitted"):
        detector.predict_risk(np.zeros((5, 29)))


def test_autoencoder_evaluate(sample_dataset):
    """Verify evaluation generates classification metrics."""
    X_train, X_test, y_train, y_test, scaler = sample_dataset

    detector = AutoencoderDetector(input_dim=29, latent_dim=8, threshold=0.70)
    detector.fit(X_train, y_train=y_train, scaler=scaler, epochs=5, batch_size=32)

    metrics = detector.evaluate(X_test, y_test, threshold=0.70, verbose=False)

    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_autoencoder_save_and_load(sample_dataset, tmp_path: Path):
    """Verify Autoencoder weights checkpoint save and reload."""
    X_train, X_test, y_train, _, scaler = sample_dataset
    model_file = tmp_path / "test_ae.pt"

    detector = AutoencoderDetector(input_dim=29, latent_dim=8)
    detector.fit(X_train, y_train=y_train, scaler=scaler, epochs=5, batch_size=32)
    detector.save(model_file)

    assert model_file.exists()

    loaded = AutoencoderDetector.load(model_file)
    assert loaded.is_fitted

    preds_original = detector.predict_risk(X_test)
    preds_loaded = loaded.predict_risk(X_test)

    assert np.allclose(preds_original, preds_loaded, atol=1e-4)
