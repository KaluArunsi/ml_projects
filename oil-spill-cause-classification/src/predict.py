"""
Prediction Module
=================
Unified inference interface for loading trained models and running
predictions on new oil spill incident data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import config
from .models._base import BaseClassifier
from .models.baseline import BaselineClassifier
from .models.distilbert_model import DistilBertClassifier

logger = logging.getLogger(__name__)


class Predictor:
    """Loads trained models and runs inference on user-provided incident data.

    Supports:
    - Single incident prediction from structured dict or raw text
    - Batch prediction from CSV/DataFrame
    - Multiple model tracks (TF-IDF, DistilBERT, MLX LLM)

    Usage::

        predictor = Predictor()
        predictor.load_model("baseline", "models/tfidf")
        result = predictor.predict_single(
            description="Vessel ran aground near coast...",
            posts=["Coast Guard responding to grounded vessel."],
        )
        print(result["predicted_causes"])
    """

    def __init__(self):
        self.models: dict[str, BaseClassifier] = {}
        self.label_names: list[str] = config.LABEL_NAMES
        self._commodity_map = config.COMMODITY_CATEGORIES

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, model_key: str, path: str) -> None:
        """Load a trained model from disk.

        Args:
            model_key: One of 'baseline', 'distilbert', 'mlx_llm'.
            path: Path to saved model directory.
        """
        if model_key == "baseline":
            self.models[model_key] = BaselineClassifier.load(path)
        elif model_key == "distilbert":
            self.models[model_key] = DistilBertClassifier.load(path)
        elif model_key == "mlx_llm":
            try:
                from .models.llm_mlx import MLXLLMClassifier
                self.models[model_key] = MLXLLMClassifier.load(path)
            except ImportError as e:
                logger.warning("Cannot load MLX model: %s", e)
        else:
            raise ValueError(f"Unknown model key: {model_key}. Use 'baseline', 'distilbert', or 'mlx_llm'.")

    def load_all_models(self, models_dir: str = config.MODELS_DIR) -> int:
        """Load all available models from the models directory.

        Returns:
            Number of models successfully loaded.
        """
        model_dirs = {
            "baseline": Path(models_dir) / "tfidf",
            "distilbert": Path(models_dir) / "distilbert",
            "mlx_llm": Path(models_dir) / "phi",
        }

        loaded = 0
        for key, path in model_dirs.items():
            if path.exists() and (path / "metadata.json").exists():
                try:
                    self.load_model(key, str(path))
                    loaded += 1
                except Exception as e:
                    logger.warning("Failed to load %s from %s: %s", key, path, e)

        logger.info("Loaded %d models from %s", loaded, models_dir)
        return loaded

    # ------------------------------------------------------------------
    # Single prediction
    # ------------------------------------------------------------------

    def predict_single(
        self,
        description: str,
        posts: list[str] | None = None,
        commodity: str | None = None,
        model_key: str = "all",
    ) -> dict:
        """Predict causes for a single oil spill incident.

        Args:
            description: Incident description text.
            posts: Optional list of associated social media post texts.
            commodity: Optional commodity type for additional classification.
            model_key: Which model to use ('baseline', 'distilbert', 'mlx_llm',
                       or 'all' for ensemble of loaded models).

        Returns:
            Dict with keys:
            - ``predicted_causes``: list of predicted cause labels
            - ``probabilities``: dict mapping cause → probability
            - ``model_used``: which model(s) produced the prediction
            - ``commodity_category``: predicted commodity (if commodity provided)
        """
        # Assemble document (same as training)
        document = self._assemble_document(description, posts)

        # Run model(s)
        if model_key == "all":
            return self._predict_all_models(document, commodity)

        if model_key not in self.models:
            raise ValueError(
                f"Model '{model_key}' not loaded. Available: {list(self.models.keys())}"
            )

        clf = self.models[model_key]
        probs = clf.predict_proba([document])[0]

        # Get thresholds and predict
        thresholds = getattr(clf, "thresholds_", np.full(len(probs), 0.5))
        preds = (probs >= thresholds).astype(int)

        predicted_causes = [
            self.label_names[i]
            for i, p in enumerate(preds)
            if p == 1 and i < len(self.label_names)
        ]

        return {
            "predicted_causes": predicted_causes,
            "probabilities": {
                self.label_names[i]: float(probs[i])
                for i in range(len(probs))
                if i < len(self.label_names)
            },
            "model_used": model_key,
            "commodity_category": self._predict_commodity(commodity) if commodity else None,
        }

    def predict_from_text(
        self, text: str, model_key: str = "all"
    ) -> dict:
        """Predict from raw text (no structured fields needed)."""
        return self.predict_single(description=text, model_key=model_key)

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        df: pd.DataFrame,
        description_col: str = "description",
        posts_col: str | None = None,
        model_key: str = "all",
    ) -> pd.DataFrame:
        """Run inference on a DataFrame of incidents.

        Args:
            df: DataFrame with incident data.
            description_col: Column name for incident descriptions.
            posts_col: Optional column with pipe-separated post texts.
            model_key: Model to use.

        Returns:
            DataFrame with added prediction columns.
        """
        result_df = df.copy()

        # Assemble documents
        documents = []
        for _, row in df.iterrows():
            desc = row.get(description_col, "")
            posts = None
            if posts_col and posts_col in df.columns:
                raw = row.get(posts_col, "")
                posts = str(raw).split("|") if pd.notna(raw) else None
            documents.append(self._assemble_document(desc, posts))

        # Predict with best available model
        if model_key == "all":
            clf = self._get_best_available_model()
        else:
            clf = self.models.get(model_key)
            if clf is None:
                raise ValueError(f"Model '{model_key}' not loaded.")

        probs = clf.predict_proba(documents)
        thresholds = getattr(clf, "thresholds_", np.full(probs.shape[1], 0.5))
        preds = (probs >= thresholds).astype(int)

        # Add prediction columns
        result_df["predicted_causes"] = [
            ", ".join(
                self.label_names[i]
                for i in range(len(row))
                if row[i] == 1 and i < len(self.label_names)
            )
            for row in preds
        ]

        # Add per-label probabilities
        for i, label in enumerate(self.label_names):
            if i < probs.shape[1]:
                result_df[f"prob_{label.lower().replace(' ', '_').replace('+', 'plus')}"] = probs[:, i]

        result_df["n_predicted_causes"] = preds.sum(axis=1)
        result_df["model_used"] = model_key

        return result_df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _assemble_document(
        self, description: str, posts: list[str] | None = None
    ) -> str:
        """Assemble document text (same format as training)."""
        parts = [str(description).strip()] if description else []
        if posts:
            parts.extend([p.strip() for p in posts[: config.MAX_POSTS_PER_INCIDENT] if p and p.strip()])
        return " [SEP] ".join(parts)

    def _predict_all_models(self, document: str, commodity: str | None) -> dict:
        """Ensemble prediction across all loaded models."""
        if not self.models:
            return {
                "predicted_causes": [],
                "probabilities": {},
                "model_used": "none",
                "error": "No models loaded. Train or load a model first.",
            }

        all_probs = []
        for name, clf in self.models.items():
            try:
                probs = clf.predict_proba([document])[0]
                all_probs.append(probs)
            except Exception as e:
                logger.warning("Model '%s' failed: %s", name, e)

        if not all_probs:
            return {"predicted_causes": [], "probabilities": {}, "model_used": "all_failed"}

        # Average ensemble
        avg_probs = np.mean(all_probs, axis=0)
        preds = (avg_probs >= 0.5).astype(int)

        predicted_causes = [
            self.label_names[i]
            for i, p in enumerate(preds)
            if p == 1 and i < len(self.label_names)
        ]

        return {
            "predicted_causes": predicted_causes,
            "probabilities": {
                self.label_names[i]: float(avg_probs[i])
                for i in range(len(avg_probs))
                if i < len(self.label_names)
            },
            "model_used": f"ensemble({','.join(self.models.keys())})",
            "commodity_category": self._predict_commodity(commodity) if commodity else None,
        }

    def _get_best_available_model(self) -> BaseClassifier:
        """Return the best available model (prefer DistilBERT, then baseline)."""
        for key in ["distilbert", "mlx_llm", "baseline"]:
            if key in self.models:
                return self.models[key]
        raise RuntimeError("No models loaded.")

    def _predict_commodity(self, raw: str) -> str:
        """Simple commodity prediction via synonym lookup."""
        import re

        if not raw or not isinstance(raw, str):
            return config.UNKNOWN_COMMODITY_LABEL

        cleaned = raw.lower().strip()
        cleaned = re.sub(r"\(.*?\)", "", cleaned)
        cleaned = re.sub(r"[^\w\s/#-]", "", cleaned)
        cleaned = cleaned.strip()

        for canonical, variants in config.COMMODITY_CATEGORIES.items():
            for variant in variants:
                if variant.lower().strip() == cleaned:
                    return canonical
        return config.UNKNOWN_COMMODITY_LABEL
