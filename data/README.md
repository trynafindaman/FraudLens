# Dataset Directory

This directory is intended to store the Kaggle Credit Card Fraud Detection dataset.

## Download Instructions

1. Download `creditcard.csv` from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Place the unzipped `creditcard.csv` directly into this directory:
   `data/creditcard.csv`

## Dataset Schema

- `Time`: Number of seconds elapsed between this transaction and the first transaction in the dataset.
- `V1` – `V28`: PCA-transformed numeric features (anonymized to protect user identity and sensitive features).
- `Amount`: Transaction amount.
- `Class`: Response variable (1 for fraudulent transaction, 0 for legitimate transaction).

## Generating a Sample for Quick Testing

You can generate a synthetic sample mirroring this schema for quick testing without downloading the full dataset by running:
```bash
python -m src.data_prep --generate-sample
```
This will create `data/sample_creditcard.csv`.
