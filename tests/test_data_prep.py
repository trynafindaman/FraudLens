"""Tests for the data preparation and EDA module."""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

from src.data_prep import (
    generate_synthetic_sample,
    load_data,
    run_eda,
    preprocess_data,
    get_stratified_split,
    EXPECTED_COLUMNS,
    FEATURE_COLS,
    V_COLS,
    AMOUNT_COL,
    CLASS_COL,
)


@pytest.fixture
def sample_csv_path(tmp_path: Path) -> Path:
    """Fixture providing a temporary synthetic CSV file path."""
    csv_file = tmp_path / "test_creditcard.csv"
    generate_synthetic_sample(
        output_path=csv_file, n_samples=300, fraud_ratio=0.05, random_state=42
    )
    return csv_file


def test_generate_synthetic_sample(tmp_path: Path):
    """Test generating a synthetic sample CSV."""
    target_path = tmp_path / "sample.csv"
    df = generate_synthetic_sample(output_path=target_path, n_samples=200, fraud_ratio=0.05)

    assert target_path.exists()
    assert len(df) == 200
    assert list(df.columns) == EXPECTED_COLUMNS
    assert (df[CLASS_COL] == 1).sum() > 0
    assert (df[CLASS_COL] == 0).sum() > 0


def test_load_data_valid(sample_csv_path: Path):
    """Test loading a valid dataset CSV."""
    df = load_data(sample_csv_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 300
    for col in EXPECTED_COLUMNS:
        assert col in df.columns


def test_load_data_missing_file(tmp_path: Path):
    """Test loading a non-existent file raises FileNotFoundError."""
    non_existent = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_data(non_existent)


def test_load_data_missing_columns(tmp_path: Path):
    """Test loading a CSV with missing required columns raises ValueError."""
    invalid_csv = tmp_path / "invalid.csv"
    df_invalid = pd.DataFrame({"Amount": [10.0], "Class": [0]})
    df_invalid.to_csv(invalid_csv, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_data(invalid_csv)


def test_run_eda(sample_csv_path: Path):
    """Test running EDA generates complete summary statistics."""
    df = load_data(sample_csv_path)
    eda = run_eda(df, verbose=False)

    assert eda["total_rows"] == 300
    assert eda["total_columns"] == len(EXPECTED_COLUMNS)
    assert eda["total_nulls"] == 0
    assert eda["normal_count"] + eda["fraud_count"] == 300
    assert eda["fraud_count"] > 0
    assert 0.0 < eda["fraud_percentage"] < 100.0
    assert "mean" in eda["amount_normal"]
    assert "mean" in eda["amount_fraud"]


def test_preprocess_data(sample_csv_path: Path):
    """Test preprocessing creates 29 features and scales Amount."""
    df = load_data(sample_csv_path)
    X, y, scaler = preprocess_data(df, fit_scaler=True)

    assert isinstance(scaler, RobustScaler)
    assert X.shape == (300, 29)
    assert list(X.columns) == FEATURE_COLS
    assert len(y) == 300
    assert set(y.unique()).issubset({0, 1})

    # Verify scaling applied without NaN
    assert not X.isnull().any().any()


def test_preprocess_data_with_existing_scaler(sample_csv_path: Path):
    """Test preprocessing using a pre-fitted scaler in inference mode."""
    df = load_data(sample_csv_path)
    X_train, _, scaler = preprocess_data(df, fit_scaler=True)

    # Use on new data without re-fitting
    df_new = df.head(10).copy()
    X_new, y_new, _ = preprocess_data(df_new, scaler=scaler, fit_scaler=False)

    assert X_new.shape == (10, 29)
    assert len(y_new) == 10


def test_get_stratified_split(sample_csv_path: Path):
    """Test stratified split preserves fraud class ratio."""
    df = load_data(sample_csv_path)
    X, y, _ = preprocess_data(df, fit_scaler=True)

    X_train, X_test, y_train, y_test = get_stratified_split(X, y, test_size=0.2, random_state=42)

    assert len(X_train) == 240
    assert len(X_test) == 60
    assert len(y_train) == 240
    assert len(y_test) == 60

    train_fraud_rate = y_train.mean()
    test_fraud_rate = y_test.mean()

    # Both train and test should have fraud examples and closely matched rates
    assert (y_train == 1).sum() > 0
    assert (y_test == 1).sum() > 0
    assert abs(train_fraud_rate - test_fraud_rate) < 0.02
