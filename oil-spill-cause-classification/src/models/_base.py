"""
Abstract Base Classifier
========================
Common interface that all three model tracks must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseClassifier(ABC):
    """Unified interface for all cause classifiers (TF-IDF, DistilBERT, MLX LLM)."""

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray | list[str],
        y_train: np.ndarray,
        X_val: np.ndarray | list[str] | None = None,
        y_val: np.ndarray | None = None,
    ) -> "BaseClassifier":
        """Train the classifier.

        Args:
            X_train: Training features or text documents.
            y_train: Multi-label binary matrix of shape (n_samples, n_labels).
            X_val: Optional validation data for threshold tuning / early stopping.
            y_val: Optional validation labels.

        Returns:
            self (for chaining).
        """
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray | list[str]) -> np.ndarray:
        """Return probability matrix of shape (n_samples, n_labels)."""
        ...

    @abstractmethod
    def predict(
        self,
        X: np.ndarray | list[str],
        thresholds: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return binary predictions using per-label thresholds."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model artifacts to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseClassifier":
        """Restore model artifacts from disk."""
        ...

    # ---- Optional helpers that subclasses may override ----

    def get_feature_importance(self, label_idx: int = 0) -> list[tuple[str, float]]:
        """Return top features for a given label. Only meaningful for linear models."""
        return []

    def get_params(self) -> dict:
        """Return a dict of model hyperparameters for logging/reproducibility."""
        return {}
