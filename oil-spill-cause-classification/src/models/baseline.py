"""
TF-IDF + Logistic Regression Baseline
=====================================
Fast, interpretable multi-label baseline using TF-IDF vectorization and
One-vs-Rest Logistic Regression with per-label threshold tuning.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from .. import config
from ._base import BaseClassifier

logger = logging.getLogger(__name__)


class BaselineClassifier(BaseClassifier):
    """TF-IDF + OneVsRest(LogisticRegression) for multi-label cause classification.

    Simple, fast, and interpretable — serves as the performance floor for
    more complex models.
    """

    def __init__(
        self,
        max_features: int = config.TFIDF_MAX_FEATURES,
        ngram_range: tuple[int, int] = config.TFIDF_NGRAM_RANGE,
        min_df: int = config.TFIDF_MIN_DF,
        C: float = config.BASELINE_C,
        random_state: int = config.RANDOM_STATE,
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.C = C
        self.random_state = random_state

        # Lazy-initialized
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier: Optional[OneVsRestClassifier] = None
        self.thresholds_: Optional[np.ndarray] = None
        self.label_names_: list[str] = []
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: list[str],
        y_train: np.ndarray,
        X_val: list[str] | None = None,
        y_val: np.ndarray | None = None,
    ) -> "BaselineClassifier":
        """Fit TF-IDF vectorizer and OneVsRest classifier.

        Args:
            X_train: List of document strings.
            y_train: Binary label matrix of shape (n_samples, n_labels).
            X_val: Optional validation documents for threshold tuning.
            y_val: Optional validation labels.
        """
        logger.info("Fitting TF-IDF vectorizer (max_features=%d, ngram=%s) ...",
                     self.max_features, self.ngram_range)
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=True,
            stop_words="english",
        )
        X_train_vec = self.vectorizer.fit_transform(X_train)

        logger.info("Training OneVsRestClassifier(LogisticRegression(C=%.1f)) ...", self.C)
        self.classifier = OneVsRestClassifier(
            LogisticRegression(
                C=self.C,
                class_weight="balanced",
                max_iter=2000,
                solver="lbfgs",
                n_jobs=-1,
                random_state=self.random_state,
            ),
            n_jobs=-1,
        )
        self.classifier.fit(X_train_vec, y_train)

        self.label_names_ = config.LABEL_NAMES[: y_train.shape[1]]

        # Mark fitted before threshold tuning (vectorizer + classifier are ready)
        self._is_fitted = True

        # Tune thresholds
        if X_val is not None and y_val is not None:
            self.thresholds_ = self._tune_thresholds(X_val, y_val)
        else:
            self.thresholds_ = np.full(y_train.shape[1], 0.5)
        logger.info("Baseline fitted — %d labels, thresholds: %s",
                     len(self.thresholds_),
                     {name: f"{t:.3f}" for name, t in zip(self.label_names_, self.thresholds_)})
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: list[str]) -> np.ndarray:
        """Return probability matrix (n_samples, n_labels)."""
        self._check_fitted()
        X_vec = self.vectorizer.transform(X)
        n_samples = X_vec.shape[0]
        n_labels = len(self.classifier.estimators_)
        probs = np.zeros((n_samples, n_labels), dtype=np.float64)

        for j, est in enumerate(self.classifier.estimators_):
            try:
                # Standard: predict_proba returns (n, 2)
                p = est.predict_proba(X_vec)
                if p.ndim == 2 and p.shape[1] >= 2:
                    probs[:, j] = p[:, 1]
                elif p.ndim == 1:
                    # Single-class edge case — use decision_function fallback
                    probs[:, j] = p
                else:
                    probs[:, j] = p.ravel()
            except (AttributeError, ValueError):
                # Fall back to decision_function
                try:
                    d = est.decision_function(X_vec)
                    probs[:, j] = 1 / (1 + np.exp(-d))
                except Exception:
                    # Last resort — all samples get constant prob
                    probs[:, j] = 0.5

        return probs

    def predict(
        self,
        X: list[str],
        thresholds: np.ndarray | None = None,
    ) -> np.ndarray:
        """Binary predictions using per-label thresholds."""
        probs = self.predict_proba(X)
        t = thresholds if thresholds is not None else self.thresholds_
        return (probs >= t).astype(int)

    # ------------------------------------------------------------------
    # Threshold tuning
    # ------------------------------------------------------------------

    def _tune_thresholds(
        self,
        X_val: list[str],
        y_val: np.ndarray,
        metric: str = config.THRESHOLD_METRIC,
        n_steps: int = config.THRESHOLD_STEPS,
    ) -> np.ndarray:
        """Grid search per-label thresholds to maximize F1 on validation set."""
        from sklearn.metrics import f1_score

        probs = self.predict_proba(X_val)
        n_labels = y_val.shape[1]
        best_thresholds = np.full(n_labels, 0.5)

        grid = np.linspace(config.THRESHOLD_MIN, config.THRESHOLD_MAX, n_steps)

        for j in range(n_labels):
            best_t, best_f1 = 0.5, 0.0
            y_true_j = y_val[:, j]
            for t in grid:
                y_pred_j = (probs[:, j] >= t).astype(int)
                f1 = f1_score(y_true_j, y_pred_j, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
            best_thresholds[j] = best_t

        logger.info("Tuned thresholds (F1 per label): val_macro=%.3f",
                     f1_score(y_val, (probs >= best_thresholds).astype(int),
                              average="macro", zero_division=0))
        return best_thresholds

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self, label_idx: int = 0) -> list[tuple[str, float]]:
        """Return top-20 TF-IDF features for a given label."""
        self._check_fitted()
        feature_names = self.vectorizer.get_feature_names_out()
        coef = self.classifier.estimators_[label_idx].coef_.ravel()

        top_indices = np.argsort(np.abs(coef))[-20:][::-1]
        return [(feature_names[i], float(coef[i])) for i in top_indices]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Pickle vectorizer, classifier, and save thresholds as JSON."""
        self._check_fitted()
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(out / "classifier.pkl", "wb") as f:
            pickle.dump(self.classifier, f)

        metadata = {
            "thresholds": self.thresholds_.tolist(),
            "label_names": self.label_names_,
            "params": {
                "max_features": self.max_features,
                "ngram_range": list(self.ngram_range),
                "min_df": self.min_df,
                "C": self.C,
            },
        }
        with open(out / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved baseline model to %s", path)

    @classmethod
    def load(cls, path: str) -> "BaselineClassifier":
        """Restore from disk."""
        p = Path(path)
        with open(p / "vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open(p / "classifier.pkl", "rb") as f:
            classifier = pickle.load(f)
        with open(p / "metadata.json") as f:
            metadata = json.load(f)

        inst = cls(**metadata["params"])
        inst.vectorizer = vectorizer
        inst.classifier = classifier
        inst.thresholds_ = np.array(metadata["thresholds"])
        inst.label_names_ = metadata["label_names"]
        inst._is_fitted = True
        logger.info("Loaded baseline model from %s", path)
        return inst

    def get_params(self) -> dict:
        return {
            "model": "TF-IDF + LogisticRegression",
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "C": self.C,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("BaselineClassifier is not fitted. Call .fit() first.")
