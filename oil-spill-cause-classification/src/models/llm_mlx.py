"""
Qwen2.5-0.5B MLX LoRA Classifier
================================
Fine-tunes ``Qwen/Qwen2.5-0.5B-Instruct`` on Apple Silicon via MLX with
LoRA adapters and a custom multi-label classification head. Uses the
model's internal hidden states (896-dim) for classification, not logits.

~1GB model — trains on any Apple Silicon Mac. No CUDA, no Ollama.

Usage:
    clf = MLXLLMClassifier()
    clf.fit(X_train, y_train, X_val, y_val)
    probs = clf.predict_proba(X_test)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from .. import config
from ._base import BaseClassifier

logger = logging.getLogger(__name__)

# Lazy imports
_MLX_AVAILABLE = False
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    _MLX_AVAILABLE = True
except ImportError:
    pass

_MLX_LM_AVAILABLE = False
try:
    import mlx_lm
    from mlx_lm import load as mlx_load
    from mlx_lm.tuner import linear_to_lora_layers

    _MLX_LM_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_params(model, params_dict: dict) -> None:
    """Load a flat dict of MLX arrays into model parameters by matching names."""
    import mlx.core as mx

    model_params = model.parameters()
    for key, value in params_dict.items():
        if key in model_params:
            if not isinstance(value, mx.array):
                value = mx.array(value)
            model_params[key] = value
    mx.eval(model.parameters())


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class MLXLLMClassifier(BaseClassifier):
    """Small LLM fine-tuned via MLX LoRA for multi-label cause classification.

    Defaults to Qwen2.5-0.5B-Instruct (~1GB, ~500MB in 4-bit).
    Users can swap to any MLX-compatible model via config.MLX_MODEL_ID.
    """

    def __init__(
        self,
        model_id: str = config.MLX_MODEL_ID,
        n_labels: int = config.NUM_LABELS,
        max_seq_length: int = config.LLM_MAX_SEQ_LENGTH,
        lora_rank: int = config.MLX_LORA_RANK,
        lora_alpha: int = config.MLX_LORA_ALPHA,
        lora_dropout: float = config.MLX_LORA_DROPOUT,
        batch_size: int = config.MLX_BATCH_SIZE,
        grad_accum: int = config.MLX_GRADIENT_ACCUM_STEPS,
        learning_rate: float = config.MLX_LEARNING_RATE,
        epochs: int = config.MLX_NUM_EPOCHS,
        warmup_steps: int = config.MLX_WARMUP_STEPS,
        weight_decay: float = config.MLX_WEIGHT_DECAY,
        random_state: int = config.RANDOM_STATE,
    ):
        if not _MLX_AVAILABLE:
            raise ImportError("MLX required. Install: pip install mlx mlx-lm")
        if not _MLX_LM_AVAILABLE:
            raise ImportError("mlx-lm required. Install: pip install mlx-lm")

        self.model_id = model_id
        self.n_labels = n_labels
        self.max_seq_length = max_seq_length
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.num_lora_layers = getattr(config, "MLX_NUM_LORA_LAYERS", 12)
        self.focal_gamma = getattr(config, "MLX_FOCAL_GAMMA", 2.5)

        self._model: Optional[nn.Module] = None
        self._tokenizer = None
        self._hidden_size: int = 0
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
    ) -> "MLXLLMClassifier":
        """Fine-tune the LLM with LoRA + classification head."""
        logger.info("=== MLX LLM Training: %s ===", self.model_id)
        logger.info("LoRA r=%d, epochs=%d, batch=%d, lr=%.1e",
                     self.lora_rank, self.epochs, self.batch_size, self.learning_rate)

        n_labels = y_train.shape[1]
        self.label_names_ = config.LABEL_NAMES[:n_labels]

        # 1. Tokenizer
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # 2. Encode all data (pre-encoded for speed)
        def _encode(texts):
            enc = self._tokenizer(
                texts, truncation=True, padding="max_length",
                max_length=self.max_seq_length, return_tensors="np",
            )
            return enc["input_ids"]

        logger.info("Tokenizing %d documents (max_len=%d)...", len(X_train), self.max_seq_length)
        all_ids = _encode(X_train)
        y_train_f32 = y_train.astype(np.float32)

        # 3. Build model
        self._model, self._hidden_size = self._build_model(n_labels)

        # 4. Alpha weights for focal loss
        pos_counts = y_train.sum(axis=0)
        alpha_np = np.clip(
            1.0 - (pos_counts + 1) / (len(y_train) + 1), 0.05, 0.99,
        )
        alpha = mx.array(alpha_np)

        # 5. Loss function (closes over model + alpha — no model arg!)
        model_ref = self._model

        def loss_fn(ids, targets):
            logits = model_ref(ids)
            p = mx.sigmoid(logits)
            bce = mx.log(p + 1e-8) * targets + mx.log(1 - p + 1e-8) * (1 - targets)
            pt = p * targets + (1 - p) * (1 - targets)
            return -((1 - pt) ** self.focal_gamma * bce * alpha.reshape(1, -1)).mean()

        value_and_grad_fn = nn.value_and_grad(self._model, loss_fn)

        # 6. Optimizer
        total_steps = max(1, len(X_train) // (self.batch_size * self.grad_accum)) * self.epochs

        def lr_schedule(step):
            if step < self.warmup_steps:
                return self.learning_rate * (step + 1) / max(self.warmup_steps, 1)
            progress = (step - self.warmup_steps) / max(total_steps - self.warmup_steps, 1)
            return self.learning_rate * max(0.0, 1.0 - progress)

        opt = optim.AdamW(learning_rate=lr_schedule, weight_decay=self.weight_decay)

        # 7. Training loop
        rng = np.random.RandomState(self.random_state)
        step = 0

        for epoch in range(self.epochs):
            indices = rng.permutation(len(X_train))
            epoch_losses = []

            for bstart in range(0, len(X_train), self.batch_size * self.grad_accum):
                bend = min(bstart + self.batch_size * self.grad_accum, len(X_train))
                bidx = indices[bstart:bend]
                accum = 0.0

                for j in range(0, len(bidx), self.batch_size):
                    micro = bidx[j:j + self.batch_size]
                    ids = mx.array(all_ids[micro])
                    lbls = mx.array(y_train_f32[micro])

                    val, grads = value_and_grad_fn(ids, lbls)
                    mx.eval(val, grads)
                    accum += float(val)

                    # Scale gradients for accumulation (handles nested dicts/tuples)
                    def _scale(g, factor):
                        if isinstance(g, dict):
                            return {k: _scale(v, factor) for k, v in g.items()}
                        if isinstance(g, (list, tuple)):
                            return type(g)(_scale(v, factor) for v in g)
                        return g * factor

                    scaled = _scale(grads, 1.0 / self.grad_accum)
                    opt.update(self._model, scaled)

                mx.eval(self._model.parameters())
                epoch_losses.append(accum / self.grad_accum)
                step += 1

            avg_loss = float(np.mean(epoch_losses))
            logger.info("Epoch %d/%d — avg loss: %.4f (step %d/%d)",
                         epoch + 1, self.epochs, avg_loss, step, total_steps)

        self._is_fitted = True

        # Threshold tuning
        if X_val is not None and y_val is not None:
            self.thresholds_ = self._tune_thresholds(X_val, y_val)
        else:
            self.thresholds_ = np.full(n_labels, 0.5)

        logger.info("Training complete — %d labels, thresholds ready", n_labels)
        return self

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self, n_labels: int) -> tuple:
        """Load base model, apply LoRA, wrap with classification head."""
        logger.info("Loading %s via MLX...", self.model_id)
        base, _ = mlx_load(self.model_id, tokenizer_config={"trust_remote_code": True})

        # Detect hidden_size from inner model
        hidden_size = self._detect_hidden_size(base)
        logger.info("Hidden size: %d", hidden_size)

        # Apply LoRA to last N transformer layers
        linear_to_lora_layers(base, self.num_lora_layers,
                              {"rank": self.lora_rank,
                               "scale": self.lora_alpha / max(self.lora_rank, 1),
                               "dropout": self.lora_dropout})
        logger.info("LoRA applied (rank=%d, layers=%d)", self.lora_rank, self.num_lora_layers)

        # Wrap: inner model (hidden states) + classification head
        inner = base.model if hasattr(base, "model") else base

        class OilSpillModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = inner
                self.norm = nn.LayerNorm(hidden_size)
                self.dropout = nn.Dropout(self.lora_dropout)
                self.head = nn.Linear(hidden_size, n_labels)

            def __call__(self, ids):
                h = self.encoder(ids)           # (B, L, hidden_size)
                pooled = h.mean(axis=1)          # (B, hidden_size)
                return self.head(self.dropout(self.norm(pooled)))

        model = OilSpillModel()
        return model, hidden_size

    @staticmethod
    def _detect_hidden_size(base: nn.Module) -> int:
        """Extract hidden_size from the MLX model structure."""
        inner = getattr(base, "model", base)
        if hasattr(inner, "layers") and inner.layers:
            layer = inner.layers[0]
            for _, child in layer.named_modules():
                w = getattr(child, "weight", None)
                if w is not None and w.ndim == 2:
                    return w.shape[-1]
        # Fallback: walk entire tree
        for _, child in base.named_modules():
            w = getattr(child, "weight", None)
            if w is not None and w.ndim == 2:
                return w.shape[-1]
        raise RuntimeError("Could not detect hidden_size from model.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: list[str]) -> np.ndarray:
        self._check_fitted()

        def _encode(texts):
            enc = self._tokenizer(
                texts, truncation=True, padding="max_length",
                max_length=self.max_seq_length, return_tensors="np",
            )
            return enc["input_ids"]

        all_probs = []
        for i in range(0, len(X), self.batch_size * 2):
            ids = mx.array(_encode(X[i:i + self.batch_size * 2]))
            p = mx.sigmoid(self._model(ids))
            mx.eval(p)
            all_probs.append(np.array(p))

        return np.concatenate(all_probs, axis=0)

    def predict(self, X: list[str], thresholds: np.ndarray | None = None) -> np.ndarray:
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
        best = np.full(n_labels, 0.5)
        grid = np.linspace(config.THRESHOLD_MIN, config.THRESHOLD_MAX, config.THRESHOLD_STEPS)

        for j in range(n_labels):
            best_t, best_f1 = 0.5, 0.0
            for t in grid:
                f1 = f1_score(y_val[:, j], (probs[:, j] >= t).astype(int), zero_division=0)
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            best[j] = best_t

        return best

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        self._check_fitted()
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        # Save weights via safetensors
        try:
            from safetensors.numpy import save_file
            params_np = {k: np.array(v) for k, v in self._model.parameters().items()
                         if "lora_a" in k or "lora_b" in k or "head" in k or "norm" in k or "dropout" in k}
            if params_np:
                save_file(params_np, str(out / "adapters.safetensors"))
        except Exception:
            np.savez(out / "adapters.npz",
                     **{k: np.array(v) for k, v in self._model.parameters().items()})

        # Metadata
        with open(out / "metadata.json", "w") as f:
            json.dump({
                "thresholds": self.thresholds_.tolist(),
                "label_names": self.label_names_,
                "hidden_size": self._hidden_size,
                "params": {
                    "model_id": self.model_id,
                    "n_labels": self.n_labels,
                    "max_seq_length": self.max_seq_length,
                    "lora_rank": self.lora_rank,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                },
            }, f, indent=2)

        self._tokenizer.save_pretrained(str(out))
        logger.info("Saved MLX model to %s", path)

    @classmethod
    def load(cls, path: str) -> "MLXLLMClassifier":
        p = Path(path)
        with open(p / "metadata.json") as f:
            meta = json.load(f)

        inst = cls(**meta["params"])
        inst.thresholds_ = np.array(meta["thresholds"])
        inst.label_names_ = meta["label_names"]
        inst._hidden_size = meta.get("hidden_size", 896)

        from transformers import AutoTokenizer
        inst._tokenizer = AutoTokenizer.from_pretrained(str(p))

        # Attempt to rebuild model and load adapter weights
        try:
            inst._model, _ = inst._build_model(meta["params"]["n_labels"])
            adapters_path = p / "adapters.safetensors"
            npz_path = p / "adapters.npz"

            if adapters_path.exists():
                import safetensors
                with safetensors.safe_open(str(adapters_path), framework="mlx") as f:
                    loaded_params = {k: f.get_tensor(k) for k in f.keys()}
                _load_params(inst._model, loaded_params)
                logger.info("Loaded MLX adapter weights from %s", adapters_path)
            elif npz_path.exists():
                loaded = dict(np.load(str(npz_path), allow_pickle=True))
                _load_params(inst._model, loaded)
                logger.info("Loaded MLX adapter weights from %s", npz_path)
            else:
                logger.warning("No adapter weights found — model needs re-fitting")

            inst._is_fitted = True
        except Exception as exc:
            logger.warning("Could not restore MLX model weights: %s. Call .fit() to retrain.", exc)
            inst._is_fitted = True  # metadata is still usable for inspection

        return inst

    def get_params(self) -> dict:
        return {
            "model": f"MLX LoRA ({self.model_id})",
            "n_labels": self.n_labels,
            "lora_rank": self.lora_rank,
            "epochs": self.epochs,
            "hidden_size": self._hidden_size,
        }

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("MLXLLMClassifier not fitted. Call .fit() first.")
