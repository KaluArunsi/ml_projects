"""
DistilBERT Multi-Label Classifier
=================================
Fine-tunes ``distilbert-base-uncased`` with a multi-label classification head
and Focal Loss for handling extreme class imbalance. Uses MPS (Apple Silicon GPU)
for training acceleration.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    DistilBertConfig,
    DistilBertModel,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .. import config
from ._base import BaseClassifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Multi-label Focal Loss.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter. Higher = more focus on hard examples.
        alpha: Per-class weight tensor of shape (n_labels,). Auto-computed if None.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else torch.tensor([]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: (B, L) raw logits.  targets: (B, L) binary {0,1}."""
        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none",
        )
        # p_t = p if y=1 else (1-p)
        p_t = probs * targets.float() + (1 - probs) * (1 - targets.float())
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * bce_loss

        if self.alpha.numel() > 0:
            alpha_t = self.alpha.to(logits.device)
            loss = loss * alpha_t.unsqueeze(0)

        return loss.mean()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DistilBertMultiLabel(nn.Module):
    """DistilBERT encoder → pooled output → Dropout → Linear → Sigmoid."""

    def __init__(
        self,
        model_name: str = config.HF_MODEL_NAME,
        n_labels: int = config.NUM_LABELS,
        dropout: float = config.BERT_DROPOUT,
    ):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 768
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, n_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fn = FocalLoss(
                gamma=config.FOCAL_LOSS_GAMMA,
                alpha=getattr(self, "_alpha", None),
            )
            loss = loss_fn(logits, labels.float())

        return {"loss": loss, "logits": logits}

    def set_alpha(self, alpha: torch.Tensor) -> None:
        """Set per-class alpha weights for focal loss."""
        self._alpha = alpha


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class IncidentDataset(Dataset):
    """Tokenized incident documents for Hugging Face Trainer.

    Supports class-weighted sampling via ``get_sampler()`` to handle
    extreme label imbalance (e.g., Grounding 287 vs Tsunami 10).
    """

    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        tokenizer: AutoTokenizer,
        max_length: int = config.BERT_MAX_SEQ_LENGTH,
    ):
        self.labels = torch.from_numpy(labels).float()
        self.texts = texts
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def get_sampler(self) -> "torch.utils.data.WeightedRandomSampler":
        """Return a WeightedRandomSampler that oversamples rare-label examples.

        Each sample's weight is proportional to the inverse frequency of
        its rarest label — ensuring minority classes appear more often.
        """
        import torch

        n_samples = len(self.labels)
        n_labels = self.labels.shape[1]

        # Per-label inverse frequency
        label_counts = self.labels.sum(dim=0).numpy()
        label_weights = 1.0 / (label_counts + 1)

        # Per-sample weight = max label weight among its positive labels
        sample_weights = np.zeros(n_samples)
        for i in range(n_samples):
            pos_mask = self.labels[i].numpy() > 0
            if pos_mask.any():
                sample_weights[i] = label_weights[pos_mask].max()
            else:
                sample_weights[i] = 0.0  # no labels (shouldn't happen)

        # Normalize
        sample_weights = sample_weights / sample_weights.sum()

        return torch.utils.data.WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=n_samples * 2,  # 2x oversampling per epoch
            replacement=True,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class DistilBertClassifier(BaseClassifier):
    """DistilBERT fine-tuned for multi-label cause classification."""

    def __init__(
        self,
        model_name: str = config.HF_MODEL_NAME,
        n_labels: int = config.NUM_LABELS,
        max_seq_length: int = config.BERT_MAX_SEQ_LENGTH,
        epochs: int = config.BERT_EPOCHS,
        batch_size: int = config.BERT_BATCH_SIZE,
        lr: float = config.BERT_LR,
        weight_decay: float = config.BERT_WEIGHT_DECAY,
        warmup_ratio: float = config.BERT_WARMUP_RATIO,
        dropout: float = config.BERT_DROPOUT,
        grad_accum: int = config.BERT_GRADIENT_ACCUM_STEPS,
        random_state: int = config.RANDOM_STATE,
    ):
        self.model_name = model_name
        self.n_labels = n_labels
        self.max_seq_length = max_seq_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.dropout = dropout
        self.grad_accum = grad_accum
        self.random_state = random_state

        # Lazy
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[DistilBertMultiLabel] = None
        self.thresholds_: Optional[np.ndarray] = None
        self.label_names_: list[str] = []
        self._is_fitted = False
        self._device = self._resolve_device()

    @staticmethod
    def _resolve_device() -> str:
        if torch.backends.mps.is_available():
            return "mps"
        return "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: list[str],
        y_train: np.ndarray,
        X_val: list[str] | None = None,
        y_val: np.ndarray | None = None,
    ) -> "DistilBertClassifier":
        """Fine-tune DistilBERT with Focal Loss."""
        device = self._device
        logger.info("Using device: %s", device)
        logger.info("Loading tokenizer + model: %s", self.model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = DistilBertMultiLabel(
            model_name=self.model_name,
            n_labels=y_train.shape[1],
            dropout=self.dropout,
        )

        # Set alpha weights for focal loss
        pos_counts = y_train.sum(axis=0)
        pos_rate = (pos_counts + 1) / (y_train.shape[0] + 1)
        alpha = torch.tensor(1.0 - pos_rate, dtype=torch.float32)
        self.model.set_alpha(alpha)
        logger.info("Focal loss alpha weights: min=%.2f max=%.2f", alpha.min(), alpha.max())

        self.model.to(device)

        # Create datasets
        train_ds = IncidentDataset(X_train, y_train, self.tokenizer, self.max_seq_length)
        val_ds = None
        if X_val is not None and y_val is not None:
            val_ds = IncidentDataset(X_val, y_val, self.tokenizer, self.max_seq_length)

        # Training args
        output_dir = os.path.join(config.MODELS_DIR, "distilbert_checkpoints")
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 2,
            gradient_accumulation_steps=self.grad_accum,
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            logging_steps=20,
            eval_strategy="epoch" if val_ds else "no",
            save_strategy="epoch" if val_ds else "no",
            load_best_model_at_end=bool(val_ds),
            metric_for_best_model="eval_macro_f1" if val_ds else None,
            greater_is_better=True,
            save_total_limit=2,
            remove_unused_columns=True,
            seed=self.random_state,
            dataloader_drop_last=False,
            fp16=False,                     # MPS doesn't support fp16 autocast reliably
            bf16=False,
            report_to="none",
        )

        # Clear MPS cache before training
        if self._device == "mps":
            import torch
            torch.mps.empty_cache()
            import gc
            gc.collect()

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=self._make_compute_metrics(),
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=config.BERT_EARLY_STOPPING_PATIENCE,
            )],
        )

        logger.info("Starting training — %d epochs, batch=%d, device=%s",
                     self.epochs, self.batch_size, device)
        trainer.train()

        # Restore best model if loaded
        self.model = self.model.to(device)
        self.label_names_ = config.LABEL_NAMES[: y_train.shape[1]]

        # Tune thresholds
        if X_val is not None and y_val is not None:
            self.thresholds_ = self._tune_thresholds(X_val, y_val)
        else:
            self.thresholds_ = np.full(y_train.shape[1], 0.5)

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_proba(self, X: list[str]) -> np.ndarray:
        """Run inference and return sigmoid probabilities."""
        self._check_fitted()
        self.model.eval()

        all_probs = []
        for i in range(0, len(X), self.batch_size * 2):
            batch_texts = X[i : i + self.batch_size * 2]
            enc = self.tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self._device)
            attention_mask = enc["attention_mask"].to(self._device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs["logits"]).cpu().numpy()
            all_probs.append(probs)

        return np.concatenate(all_probs, axis=0)

    def predict(
        self,
        X: list[str],
        thresholds: np.ndarray | None = None,
    ) -> np.ndarray:
        probs = self.predict_proba(X)
        t = thresholds if thresholds is not None else self.thresholds_
        return (probs >= t).astype(int)

    # ------------------------------------------------------------------
    # Threshold tuning
    # ------------------------------------------------------------------

    def _tune_thresholds(self, X_val: list[str], y_val: np.ndarray) -> np.ndarray:
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

        logger.info("DistilBERT tuned thresholds: macro-F1 on val = %.3f",
                     f1_score(y_val, (probs >= best_thresholds).astype(int),
                              average="macro", zero_division=0))
        return best_thresholds

    # ------------------------------------------------------------------
    # Metrics callback
    # ------------------------------------------------------------------

    def _make_compute_metrics(self):
        """Build a compute_metrics function for HF Trainer."""
        from sklearn.metrics import f1_score

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            probs = 1.0 / (1.0 + np.exp(-logits))
            preds = (probs >= 0.5).astype(int)
            return {
                "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
                "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            }

        return compute_metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        self._check_fitted()
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(out))
        self.tokenizer.save_pretrained(str(out))

        metadata = {
            "thresholds": self.thresholds_.tolist(),
            "label_names": self.label_names_,
            "params": {
                "model_name": self.model_name,
                "n_labels": self.n_labels,
                "max_seq_length": self.max_seq_length,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "dropout": self.dropout,
            },
        }
        with open(out / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved DistilBERT model to %s", path)

    @classmethod
    def load(cls, path: str) -> "DistilBertClassifier":
        p = Path(path)
        with open(p / "training_metadata.json") as f:
            metadata = json.load(f)

        inst = cls(**metadata["params"])
        inst.tokenizer = AutoTokenizer.from_pretrained(str(p))

        # Load encoder from saved directory (safetensors or pytorch_model.bin)
        encoder = DistilBertModel.from_pretrained(str(p))
        inst.model = DistilBertMultiLabel(
            model_name=metadata["params"]["model_name"],
            n_labels=metadata["params"]["n_labels"],
        )
        inst.model.encoder = encoder
        # Load classifier head weights if present
        model_file = p / "model.safetensors"
        if model_file.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(model_file))
            inst.model.load_state_dict(state_dict, strict=False)
        elif (p / "pytorch_model.bin").exists():
            state_dict = torch.load(p / "pytorch_model.bin", map_location="cpu")
            inst.model.load_state_dict(state_dict, strict=False)

        inst.thresholds_ = np.array(metadata["thresholds"])
        inst.label_names_ = metadata["label_names"]
        inst._is_fitted = True
        inst.model.to(inst._device)
        logger.info("Loaded DistilBERT model from %s", path)
        return inst

    def get_params(self) -> dict:
        return {
            "model": "DistilBERT (Multi-Label)",
            "model_name": self.model_name,
            "max_seq_length": self.max_seq_length,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "device": self._device,
        }

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("DistilBertClassifier not fitted. Call .fit() first.")
