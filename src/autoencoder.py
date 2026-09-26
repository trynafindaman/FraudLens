"""PyTorch Autoencoder for Fraud Anomaly Detection (Milestone 5 Stretch Goal).

Reconstruction-error anomaly detection:
The model is trained predominantly on normal transactions (y == 0) to learn their
latent manifold. High reconstruction error on unseen transactions signals an anomaly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data_prep import FEATURE_COLS

logger = logging.getLogger(__name__)

DEFAULT_AE_PATH = Path(__file__).resolve().parent.parent / "models" / "autoencoder.pt"


class AutoencoderNet(nn.Module):
    """Deep autoencoder network with symmetrical encoder-decoder bottleneck."""

    def __init__(self, input_dim: int = 29, latent_dim: int = 8) -> None:
        """Initialize symmetrical Autoencoder neural network architecture."""
        super().__init__()
        # Encoder: 29 -> 16 -> 8
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.ReLU(),
        )
        # Decoder: 8 -> 16 -> 29
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and decoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderDetector:
    """Wrapper for training, scoring, evaluating, and persisting the Autoencoder."""

    def __init__(
        self,
        input_dim: int = 29,
        latent_dim: int = 8,
        lr: float = 1e-3,
        threshold: float = 0.80,
        random_state: int = 42,
    ) -> None:
        """Initialize Autoencoder detector hyper-parameters and network."""
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.threshold = threshold
        self.network = AutoencoderNet(input_dim=input_dim, latent_dim=latent_dim)
        self.scaler: Optional[RobustScaler] = None
        self.min_mse: float = 0.0
        self.max_mse: float = 1.0
        self.is_fitted: bool = False

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: Optional[pd.Series | np.ndarray] = None,
        scaler: Optional[RobustScaler] = None,
        epochs: int = 15,
        batch_size: int = 64,
    ) -> "AutoencoderDetector":
        """Train Autoencoder on normal transactions (y == 0) to minimize reconstruction loss.

        Args:
            X_train: Training features.
            y_train: Labels (if provided, filters to train strictly on legitimate transactions).
            scaler: Pre-fitted RobustScaler for transaction amounts.
            epochs: Training epochs.
            batch_size: Batch size for mini-batch SGD.
        """
        self.scaler = scaler
        X_mat = X_train.values if isinstance(X_train, pd.DataFrame) else np.asarray(X_train)

        # Train exclusively on normal transactions if labels are provided
        if y_train is not None:
            y_arr = np.asarray(y_train)
            normal_mask = (y_arr == 0)
            X_train_normal = X_mat[normal_mask]
            logger.info(
                "Filtered %d normal transactions for Autoencoder training.",
                len(X_train_normal),
            )
        else:
            X_train_normal = X_mat

        tensor_x = torch.tensor(X_train_normal, dtype=torch.float32)
        dataset = TensorDataset(tensor_x)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        self.network.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for (batch_x,) in loader:
                optimizer.zero_grad()
                reconstructed = self.network(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)

            epoch_loss /= len(X_train_normal)
            if epoch % 5 == 0 or epoch == epochs:
                logger.info(
                    "Autoencoder Epoch [%d/%d] - Reconstruction Loss: %.5f",
                    epoch,
                    epochs,
                    epoch_loss,
                )

        self.network.eval()
        self.is_fitted = True

        # Calibrate min/max reconstruction error bounds on training set
        with torch.no_grad():
            reconstructed_all = self.network(tensor_x)
            mse = torch.mean((tensor_x - reconstructed_all) ** 2, dim=1).numpy()
            self.min_mse = float(np.percentile(mse, 1))
            self.max_mse = float(np.percentile(mse, 99))

        return self

    def predict_risk(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Compute normalized fraud risk score based on MSE reconstruction error.

        Higher reconstruction error = higher likelihood of fraud.
        """
        if not self.is_fitted:
            raise RuntimeError("AutoencoderDetector is not fitted yet.")

        X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        tensor_x = torch.tensor(X_mat, dtype=torch.float32)

        self.network.eval()
        with torch.no_grad():
            reconstructed = self.network(tensor_x)
            mse = torch.mean((tensor_x - reconstructed) ** 2, dim=1).numpy()

        denom = (self.max_mse - self.min_mse) if self.max_mse > self.min_mse else 1.0
        risk_scores = np.clip((mse - self.min_mse) / denom, 0.0, 1.0)
        return np.round(risk_scores, 4)

    def evaluate(
        self,
        X_test: pd.DataFrame | np.ndarray,
        y_test: pd.Series | np.ndarray,
        threshold: Optional[float] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run offline evaluation on test set."""
        y_true = np.asarray(y_test)
        effective_threshold = self.threshold if threshold is None else threshold

        risk_scores = self.predict_risk(X_test)
        y_pred = (risk_scores >= effective_threshold).astype(int)

        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        has_both_classes = len(np.unique(y_true)) > 1
        roc_auc = float(roc_auc_score(y_true, risk_scores)) if has_both_classes else 0.0
        pr_auc = float(average_precision_score(y_true, risk_scores)) if has_both_classes else 0.0

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        metrics = {
            "model_name": "PyTorch Autoencoder",
            "threshold": effective_threshold,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        }

        if verbose:
            print("\n" + "=" * 60)
            print("         OFFLINE EVALUATION (PYTORCH AUTOENCODER)           ")
            print("=" * 60)
            print(f"Threshold:           {effective_threshold:.2f}")
            print(f"Precision:           {precision:.4f}")
            print(f"Recall:              {recall:.4f}")
            print(f"F1-Score:            {f1:.4f}")
            print(f"ROC-AUC:             {roc_auc:.4f}")
            print(f"PR-AUC (Avg Prec):   {pr_auc:.4f}")
            print("=" * 60 + "\n")

        return metrics

    def save(self, filepath: Path | str = DEFAULT_AE_PATH) -> Path:
        """Save network weights and metadata securely."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        scaler_center = (
            torch.tensor(self.scaler.center_, dtype=torch.float32)
            if self.scaler is not None and hasattr(self.scaler, "center_")
            else None
        )
        scaler_scale = (
            torch.tensor(self.scaler.scale_, dtype=torch.float32)
            if self.scaler is not None and hasattr(self.scaler, "scale_")
            else None
        )
        checkpoint = {
            "state_dict": self.network.state_dict(),
            "scaler_center": scaler_center,
            "scaler_scale": scaler_scale,
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "threshold": self.threshold,
            "min_mse": self.min_mse,
            "max_mse": self.max_mse,
        }
        torch.save(checkpoint, path)
        logger.info("Saved Autoencoder checkpoint to %s", path)
        return path

    @classmethod
    def load(cls, filepath: Path | str = DEFAULT_AE_PATH) -> "AutoencoderDetector":
        """Load network weights and metadata safely using weights_only=True."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Autoencoder checkpoint not found at '{path}'.")

        logger.info("Loading Autoencoder checkpoint from %s ...", path)
        checkpoint = torch.load(path, map_location=torch.device("cpu"), weights_only=True)

        instance = cls(
            input_dim=checkpoint["input_dim"],
            latent_dim=checkpoint["latent_dim"],
            threshold=checkpoint["threshold"],
        )
        instance.network.load_state_dict(checkpoint["state_dict"])

        has_center = checkpoint.get("scaler_center") is not None
        has_scale = checkpoint.get("scaler_scale") is not None
        if has_center and has_scale:
            rebuilt_scaler = RobustScaler()
            rebuilt_scaler.center_ = checkpoint["scaler_center"].numpy()
            rebuilt_scaler.scale_ = checkpoint["scaler_scale"].numpy()
            instance.scaler = rebuilt_scaler
        elif "scaler" in checkpoint:
            instance.scaler = checkpoint["scaler"]
        else:
            instance.scaler = None

        instance.min_mse = checkpoint["min_mse"]
        instance.max_mse = checkpoint["max_mse"]
        instance.is_fitted = True
        return instance
