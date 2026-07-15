"""
Ensemble Classifier
===================
Weighted voting ensemble that combines predictions from all three modeling
tracks (TF-IDF, DistilBERT, MLX LLM) for improved accuracy.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from . import config
from .models._base import BaseClassifier

logger = logging.getLogger(__name__)


class EnsembleClassifier(BaseClassifier):
    """Weighted average ensemble of multiple cause classifiers.

    Combines probability estimates from each model using learned weights
    optimized on the validation set.
    """

    def __init__(
        self,
        classifiers: dict[str, BaseClassifier],
        weights: Optional[dict[str, float]] = None,
    ):
        """
        Args:
            classifiers: Dict mapping model name → fitted BaseClassifier.
            weights: Optional per-model weights. If None, uniform weights are used.
        """
        self.classifiers = classifiers
        self.weights = weights or {name: 1.0 for name in classifiers}
        self.thresholds_: Optional[np.ndarray] = None
        self.label_names_: list[str] = []
        self._is_fitted = True  # Ensemble is ready once classifiers are provided

    # ------------------------------------------------------------------
    # Training (weight learning)
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: list[str],
        y_train: np.ndarray,
        X_val: list[str] | None = None,
        y_val: np.ndarray | None = None,
    ) -> "EnsembleClassifier":
        """Learn optimal ensemble weights from validation data.

        Optimizes per-model weights to maximize macro-F1 on the validation set
        using a simple grid search over weight ratios.
        """
        if not self.classifiers:
            raise RuntimeError("No classifiers provided to ensemble.")

        self.label_names_ = config.LABEL_NAMES[: y_train.shape[1]]

        if X_val is not None and y_val is not None:
            self.weights = self._learn_weights(X_val, y_val)
            self.thresholds_ = self._tune_thresholds(X_val, y_val)

        return self

    def _learn_weights(
        self, X_val: list[str], y_val: np.ndarray
    ) -> dict[str, float]:
        """Learn per-model weights via grid search on validation macro-F1.

        For 2 models: grid search over w ∈ [0, 1] with w₁ = 1 - w₀.
        For 3 models: grid search over the 2-simplex with steps of 0.2.
        For N  > 3 models: sample random weight vectors and pick best.
        """
        from sklearn.metrics import f1_score

        model_names = list(self.classifiers.keys())
        n_models = len(model_names)
        if n_models <= 1:
            return {name: 1.0 for name in model_names}

        # Get probabilities from each model
        all_probs = {}
        for name, clf in self.classifiers.items():
            all_probs[name] = clf.predict_proba(X_val)

        best_weights = {name: 1.0 / n_models for name in model_names}
        best_f1 = 0.0

        # Generate candidate weight vectors
        candidates: list[list[float]] = []

        if n_models == 2:
            # Grid: w₀ ∈ [0, 1], w₁ = 1 - w₀
            for w0 in np.linspace(0.0, 1.0, 11):
                candidates.append([w0, 1.0 - w0])
        elif n_models == 3:
            # Sample the 2-simplex: w₀ + w₁ + w₂ = 1.0, all ≥ 0
            for w0 in np.linspace(0.0, 1.0, 7):
                for w1 in np.linspace(0.0, 1.0 - w0, 7):
                    w2 = 1.0 - w0 - w1
                    if w2 >= 0:
                        candidates.append([w0, w1, w2])
        else:
            # Random sampling for N > 3
            rng = np.random.RandomState(config.RANDOM_STATE)
            for _ in range(500):
                raw = rng.dirichlet(np.ones(n_models))
                candidates.append(raw.tolist())

        for weights in candidates:
            ensemble_probs = np.zeros_like(all_probs[model_names[0]])
            for i, name in enumerate(model_names):
                ensemble_probs += weights[i] * all_probs[name]

            preds = (ensemble_probs >= 0.5).astype(int)
            f1 = f1_score(y_val, preds, average="macro", zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_weights = {name: weights[i] for i, name in enumerate(model_names)}

        logger.info("Learned ensemble weights: %s (val macro-F1=%.3f)",
                     {k: f"{v:.2f}" for k, v in best_weights.items()}, best_f1)
        return best_weights

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: list[str]) -> np.ndarray:
        """Weighted average of probability estimates."""
        if not self.classifiers:
            raise RuntimeError("Ensemble has no classifiers.")

        all_probs = []
        weight_sum = 0.0
        for name, clf in self.classifiers.items():
            w = self.weights.get(name, 1.0)
            all_probs.append(w * clf.predict_proba(X))
            weight_sum += w

        return np.sum(all_probs, axis=0) / weight_sum

    def predict(
        self,
        X: list[str],
        thresholds: np.ndarray | None = None,
    ) -> np.ndarray:
        """Binary predictions using per-label thresholds."""
        probs = self.predict_proba(X)
        t = thresholds if thresholds is not None else self.thresholds_
        if t is None:
            t = np.full(probs.shape[1], 0.5)
        return (probs >= t).astype(int)

    # ------------------------------------------------------------------
    # Threshold tuning
    # ------------------------------------------------------------------

    def _tune_thresholds(
        self, X_val: list[str], y_val: np.ndarray
    ) -> np.ndarray:
        from sklearn.metrics import f1_score

        probs = self.predict_proba(X_val)
        n_labels = y_val.shape[1]
        best_thresholds = np.full(n_labels, 0.5)
        grid = np.linspace(config.THRESHOLD_MIN, config.THRESHOLD_MAX, config.THRESHOLD_STEPS)

        for j in range(n_labels):
            best_t, best_f1 = 0.5, 0.0
            for t in grid:
                f1 = f1_score(y_val[:, j], (probs[:, j] >= t).astype(int), zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
            best_thresholds[j] = best_t

        return best_thresholds

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save ensemble weights and thresholds (individual models saved separately)."""
        import json
        from pathlib import Path

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        metadata = {
            "weights": self.weights,
            "thresholds": self.thresholds_.tolist() if self.thresholds_ is not None else None,
            "label_names": self.label_names_,
        }
        with open(out / "ensemble_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved ensemble metadata to %s", path)

    @classmethod
    def load(cls, path: str) -> "EnsembleClassifier":
        """Load ensemble metadata. Individual models must be loaded separately."""
        import json
        from pathlib import Path

        p = Path(path)
        with open(p / "ensemble_metadata.json") as f:
            metadata = json.load(f)

        inst = cls(classifiers={}, weights=metadata["weights"])
        inst.thresholds_ = (
            np.array(metadata["thresholds"])
            if metadata["thresholds"] is not None
            else None
        )
        inst.label_names_ = metadata.get("label_names", [])
        return inst

    def get_params(self) -> dict:
        return {
            "model": "Ensemble (Weighted Voting)",
            "n_classifiers": len(self.classifiers),
            "weights": self.weights,
        }
