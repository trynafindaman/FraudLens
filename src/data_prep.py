"""Data preparation and Exploratory Data Analysis (EDA) module.

Handles loading, validating, analyzing, scaling, and partitioning the Kaggle
Credit Card Fraud Detection dataset for anomaly detection modeling.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants matching Kaggle Credit Card Fraud dataset schema
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DATA_PATH = DEFAULT_DATA_DIR / "creditcard.csv"
SAMPLE_DATA_PATH = DEFAULT_DATA_DIR / "sample_creditcard.csv"

TIME_COL = "Time"
AMOUNT_COL = "Amount"
CLASS_COL = "Class"
V_COLS = [f"V{i}" for i in range(1, 29)]
EXPECTED_COLUMNS = [TIME_COL] + V_COLS + [AMOUNT_COL, CLASS_COL]
FEATURE_COLS = V_COLS + ["scaled_amount"]


def generate_synthetic_sample(
    output_path: Path | str = SAMPLE_DATA_PATH,
    n_samples: int = 2000,
    fraud_ratio: float = 0.005,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic dataset mirroring the Kaggle Credit Card schema.

    Enables testing and verifying data preparation and EDA pipelines without
    requiring the full external Kaggle download immediately.

    Args:
        output_path: File path to save the generated CSV.
        n_samples: Total number of rows to generate.
        fraud_ratio: Fraction of rows that represent fraudulent transactions.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame containing synthetic transactions matching the Kaggle schema.
    """
    output_path = Path(output_path)
    rng = np.random.default_rng(random_state)

    n_fraud = max(1, int(n_samples * fraud_ratio))
    n_normal = n_samples - n_fraud

    # Normal transactions: V1-V28 standard normal, amounts lognormal
    v_normal = rng.normal(loc=0.0, scale=1.0, size=(n_normal, 28))
    amounts_normal = np.round(rng.lognormal(mean=3.0, sigma=1.2, size=n_normal), 2)
    time_normal = np.sort(rng.uniform(0, 172800, size=n_normal))
    class_normal = np.zeros(n_normal, dtype=int)

    # Fraud transactions: shifted distributions for certain PCA components
    v_fraud = rng.normal(loc=0.0, scale=1.0, size=(n_fraud, 28))
    v_fraud[:, [1, 3, 9, 11, 13]] += rng.choice([-3.0, 3.0], size=(n_fraud, 5))
    amounts_fraud = np.round(rng.lognormal(mean=4.2, sigma=1.5, size=n_fraud), 2)
    time_fraud = rng.uniform(0, 172800, size=n_fraud)
    class_fraud = np.ones(n_fraud, dtype=int)

    # Combine normal and fraud
    times = np.concatenate([time_normal, time_fraud])
    vs = np.vstack([v_normal, v_fraud])
    amounts = np.concatenate([amounts_normal, amounts_fraud])
    classes = np.concatenate([class_normal, class_fraud])

    # Shuffle combined dataset
    shuffle_idx = rng.permutation(n_samples)
    data = {TIME_COL: times[shuffle_idx]}
    for idx, col_name in enumerate(V_COLS):
        data[col_name] = vs[shuffle_idx, idx]
    data[AMOUNT_COL] = amounts[shuffle_idx]
    data[CLASS_COL] = classes[shuffle_idx]

    df = pd.DataFrame(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(
        "Saved synthetic sample with %d rows (%d fraud) to %s",
        n_samples,
        n_fraud,
        output_path,
    )
    return df


def load_data(filepath: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and validate the transaction dataset from a CSV file.

    Args:
        filepath: Path to the credit card transactions CSV.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If expected columns are missing.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Please download 'creditcard.csv' from Kaggle:\n"
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            f"and place it at '{DEFAULT_DATA_PATH}', or generate a sample using:\n"
            "  python -m src.data_prep --generate-sample"
        )

    logger.info("Loading transaction dataset from %s ...", path)
    df = pd.read_csv(path)

    # Validate schema
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_cols)}")

    logger.info("Loaded %d transactions with %d columns successfully.", len(df), len(df.columns))
    return df


def run_eda(df: pd.DataFrame, verbose: bool = True) -> Dict[str, Any]:
    """Perform Exploratory Data Analysis (EDA) on the transactions dataset.

    Calculates shapes, missing values, class distributions, and amount statistics.

    Args:
        df: Loaded transaction DataFrame.
        verbose: If True, prints a formatted summary to stdout.

    Returns:
        Dictionary containing EDA summary metrics.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())

    class_counts = df[CLASS_COL].value_counts().to_dict()
    normal_count = int(class_counts.get(0, 0))
    fraud_count = int(class_counts.get(1, 0))
    fraud_pct = (fraud_count / total_rows * 100.0) if total_rows > 0 else 0.0
    normal_pct = (normal_count / total_rows * 100.0) if total_rows > 0 else 0.0
    imbalance_ratio = (normal_count / fraud_count) if fraud_count > 0 else float("inf")

    amount_overall = df[AMOUNT_COL].describe().to_dict()
    normal_amounts = df[df[CLASS_COL] == 0][AMOUNT_COL]
    fraud_amounts = df[df[CLASS_COL] == 1][AMOUNT_COL]

    amount_normal_stats = {
        "mean": float(normal_amounts.mean()) if not normal_amounts.empty else 0.0,
        "std": float(normal_amounts.std()) if not normal_amounts.empty else 0.0,
        "median": float(normal_amounts.median()) if not normal_amounts.empty else 0.0,
        "min": float(normal_amounts.min()) if not normal_amounts.empty else 0.0,
        "max": float(normal_amounts.max()) if not normal_amounts.empty else 0.0,
    }
    amount_fraud_stats = {
        "mean": float(fraud_amounts.mean()) if not fraud_amounts.empty else 0.0,
        "std": float(fraud_amounts.std()) if not fraud_amounts.empty else 0.0,
        "median": float(fraud_amounts.median()) if not fraud_amounts.empty else 0.0,
        "min": float(fraud_amounts.min()) if not fraud_amounts.empty else 0.0,
        "max": float(fraud_amounts.max()) if not fraud_amounts.empty else 0.0,
    }

    eda_summary = {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "total_nulls": total_nulls,
        "normal_count": normal_count,
        "fraud_count": fraud_count,
        "fraud_percentage": fraud_pct,
        "normal_percentage": normal_pct,
        "imbalance_ratio": imbalance_ratio,
        "amount_overall": amount_overall,
        "amount_normal": amount_normal_stats,
        "amount_fraud": amount_fraud_stats,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("          EXPLORATORY DATA ANALYSIS (EDA) SUMMARY          ")
        print("=" * 60)
        print(f"Shape: {total_rows:,} rows x {total_cols} columns")
        print(f"Missing Values: {total_nulls} nulls across all columns")
        print("-" * 60)
        print("Class Distribution:")
        print(f"  - Normal (0): {normal_count:,} ({normal_pct:.3f}%)")
        print(f"  - Fraud  (1): {fraud_count:,} ({fraud_pct:.3f}%)")
        print(f"  - Imbalance: 1 fraud per {imbalance_ratio:,.1f} normal transactions")
        print("-" * 60)
        print("Transaction Amount ($) Comparison:")
        print(
            f"  - Overall: mean = ${amount_overall.get('mean', 0.0):.2f}, "
            f"median = ${amount_overall.get('50%', 0.0):.2f}, "
            f"max = ${amount_overall.get('max', 0.0):.2f}"
        )
        print(
            f"  - Normal:  mean = ${amount_normal_stats['mean']:.2f}, "
            f"median = ${amount_normal_stats['median']:.2f}, "
            f"max = ${amount_normal_stats['max']:.2f}"
        )
        print(
            f"  - Fraud:   mean = ${amount_fraud_stats['mean']:.2f}, "
            f"median = ${amount_fraud_stats['median']:.2f}, "
            f"max = ${amount_fraud_stats['max']:.2f}"
        )
        print("=" * 60 + "\n")

    return eda_summary


def preprocess_data(
    df: pd.DataFrame,
    scaler: RobustScaler | None = None,
    fit_scaler: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, RobustScaler]:
    """Clean, scale Amount with RobustScaler, and prepare feature matrix X and target y.

    RobustScaler is used because transaction amounts exhibit heavy right-skewed
    tails with extreme outliers that would distort standard mean/variance scaling.

    Args:
        df: Raw transactions DataFrame.
        scaler: Existing fitted RobustScaler, or None to create a new one.
        fit_scaler: Whether to fit the scaler on df (set False during test/inference).

    Returns:
        Tuple of (X: DataFrame of 29 features [V1-V28, scaled_amount],
                  y: Series of target Class [0, 1],
                  scaler: fitted RobustScaler instance)
    """
    df_clean = df.copy()

    # Drop nulls if any exist
    if df_clean.isnull().any().any():
        logger.warning("Found nulls in dataset, dropping rows with null values.")
        df_clean = df_clean.dropna()

    if scaler is None:
        scaler = RobustScaler()

    # Scale Amount
    amount_values = df_clean[[AMOUNT_COL]].values
    if fit_scaler:
        scaled_amount = scaler.fit_transform(amount_values)
    else:
        scaled_amount = scaler.transform(amount_values)

    # Assemble feature matrix X
    X = df_clean[V_COLS].copy()
    X["scaled_amount"] = scaled_amount

    y = df_clean[CLASS_COL].copy()

    return X, y, scaler


def get_stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Partition features and labels into stratified train and test sets.

    Stratification ensures that the rare fraud class ratio (~0.17%) is
    faithfully preserved in both training and test partitions.

    Args:
        X: Feature matrix.
        y: Binary target series.
        test_size: Proportion of dataset allocated to the test split.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    logger.info(
        "Stratified split complete: Train=%d (fraud=%d), Test=%d (fraud=%d)",
        len(y_train),
        int(y_train.sum()),
        len(y_test),
        int(y_test.sum()),
    )
    return X_train, X_test, y_train, y_test


def main() -> None:
    """CLI entry point for running data preparation and EDA."""
    parser = argparse.ArgumentParser(
        description="Data preparation and EDA for Real-Time Fraud Detection."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help=f"Path to creditcard.csv (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate a synthetic sample CSV for immediate testing without full Kaggle download.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=2000,
        help="Number of rows for synthetic sample (default: 2000).",
    )
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use data/sample_creditcard.csv instead of the full creditcard.csv.",
    )
    args = parser.parse_args()

    # Handle synthetic sample generation
    if args.generate_sample:
        generate_synthetic_sample(output_path=SAMPLE_DATA_PATH, n_samples=args.sample_rows)
        print(f"Sample generated successfully at: {SAMPLE_DATA_PATH}")
        return

    # Determine file to load
    target_path = Path(SAMPLE_DATA_PATH if args.use_sample else args.data_path)

    # Fallback to generating sample if data doesn't exist yet
    if not target_path.exists():
        if target_path == DEFAULT_DATA_PATH:
            logger.info(
                "Kaggle 'creditcard.csv' not found. "
                "Generating a sample dataset for demonstration..."
            )
            generate_synthetic_sample(
                output_path=SAMPLE_DATA_PATH, n_samples=args.sample_rows
            )
            target_path = SAMPLE_DATA_PATH
        else:
            raise FileNotFoundError(f"Requested dataset file not found: {target_path}")

    # Load data
    df = load_data(target_path)

    # Run EDA
    run_eda(df, verbose=True)

    # Run Preprocessing & Stratified Splitting
    logger.info("Executing preprocessing and stratified train/test split...")
    X, y, scaler = preprocess_data(df, fit_scaler=True)
    X_train, X_test, y_train, y_test = get_stratified_split(X, y, test_size=0.2)

    train_fraud_cnt = int(y_train.sum())
    train_fraud_pct = y_train.mean() * 100.0
    test_fraud_cnt = int(y_test.sum())
    test_fraud_pct = y_test.mean() * 100.0

    print("-" * 60)
    print("PREPROCESSING & SPLIT VERIFICATION:")
    print(f"  - Feature count: {X.shape[1]} features (V1-V28 + scaled_amount)")
    print(
        f"  - Training set:  {len(X_train):,} samples "
        f"(Fraud: {train_fraud_cnt:,}, {train_fraud_pct:.3f}%)"
    )
    print(
        f"  - Testing set:   {len(X_test):,} samples "
        f"(Fraud: {test_fraud_cnt:,}, {test_fraud_pct:.3f}%)"
    )
    print(f"  - Amount scaler: {type(scaler).__name__} fitted successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
