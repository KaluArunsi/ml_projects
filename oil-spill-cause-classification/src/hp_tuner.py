"""
Hyperparameter Tuner (Optuna)
=============================
Bayesian hyperparameter optimization for DistilBERT and Qwen2.5-0.5B MLX LoRA.
Target: macro-F1 > 0.5 for both models on the oil spill cause classification task.

Uses median pruning to kill unpromising trials early, saving ~60% of compute.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from . import config

logger = logging.getLogger(__name__)

# Shared data cache (loaded once)
_DATA_CACHE: Optional[dict] = None


def _get_data() -> dict:
    """Load and cache the preprocessed data."""
    global _DATA_CACHE
    if _DATA_CACHE is not None:
        return _DATA_CACHE

    from .preprocessing import run_preprocessing_pipeline

    result = run_preprocessing_pipeline(save_artifacts=False)
    incidents = result["incidents"]
    label_matrix = result["label_matrix"]
    label_names = result["label_names"]
    splits = result["splits"]
    has_labels = result["has_labels"]
    labeled_df = incidents[has_labels].reset_index(drop=True)

    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    _DATA_CACHE = {
        "X_train": labeled_df.loc[train_idx, "document"].tolist(),
        "y_train": label_matrix[train_idx],
        "X_val": labeled_df.loc[val_idx, "document"].tolist(),
        "y_val": label_matrix[val_idx],
        "X_test": labeled_df.loc[test_idx, "document"].tolist(),
        "y_test": label_matrix[test_idx],
        "label_names": label_names,
    }
    return _DATA_CACHE


# ═══════════════════════════════════════════════════════════════════════
# DistilBERT Hyperparameter Tuning
# ═══════════════════════════════════════════════════════════════════════

def tune_distilbert(
    n_trials: int = 20,
    timeout: int = 7200,        # 2 hours
    output_dir: str = "models/distilbert",
) -> dict:
    """Run Optuna hyperparameter optimization for DistilBERT.

    Search space:
    - learning_rate: [5e-6, 5e-5] (log)
    - dropout: [0.1, 0.5]
    - focal_gamma: [1.0, 4.0]
    - epochs: [3, 10]
    - batch_size: {4, 8, 16}
    - warmup_ratio: [0.05, 0.2]
    - weight_decay: [1e-4, 1e-1] (log)

    Objective: validation macro-F1.

    Returns:
        Dict with best params, best macro-F1, and study statistics.
    """
    data = _get_data()
    from sklearn.metrics import f1_score

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective — train DistilBERT and return val macro-F1."""
        from .models.distilbert_model import DistilBertClassifier

        # Sample hyperparameters
        lr = trial.suggest_float("learning_rate", 5e-6, 5e-5, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        focal_gamma = trial.suggest_float("focal_gamma", 1.0, 4.0)
        epochs = trial.suggest_int("epochs", 3, 10)
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
        warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.2)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)

        # Update config temporarily
        import torch
        from .models.distilbert_model import FocalLoss, DistilBertMultiLabel
        from transformers import AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback

        device = "mps" if torch.backends.mps.is_available() else "cpu"

        # Load model with trial params
        model = DistilBertMultiLabel(
            model_name=config.HF_MODEL_NAME,
            n_labels=data["y_train"].shape[1],
            dropout=dropout,
        )
        model.to(device)

        # Set alpha weights for focal loss
        pos_counts = data["y_train"].sum(axis=0)
        pos_rate = (pos_counts + 1) / (data["y_train"].shape[0] + 1)
        alpha = torch.tensor(1.0 - pos_rate, dtype=torch.float32)
        model.set_alpha(alpha)

        tokenizer = AutoTokenizer.from_pretrained(config.HF_MODEL_NAME)

        # Create datasets
        from .models.distilbert_model import IncidentDataset

        train_ds = IncidentDataset(
            data["X_train"], data["y_train"], tokenizer, config.DISTILBERT_MAX_SEQ_LENGTH,
        )
        val_ds = IncidentDataset(
            data["X_val"], data["y_val"], tokenizer, config.DISTILBERT_MAX_SEQ_LENGTH,
        )

        output_tmp = f"/tmp/optuna_distilbert_trial_{trial.number}"
        args = TrainingArguments(
            output_dir=output_tmp,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            gradient_accumulation_steps=max(1, 16 // batch_size),
            learning_rate=lr,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,
            fp16=False,
            bf16=False,
            report_to="none",
            seed=config.RANDOM_STATE,
            dataloader_drop_last=False,
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            probs = 1.0 / (1.0 + np.exp(-logits))
            preds = (probs >= 0.5).astype(int)
            return {
                "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
                "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            }

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        # Evaluate on validation set
        eval_results = trainer.evaluate()
        macro_f1 = eval_results.get("eval_macro_f1", 0.0)

        # Clean up temp files
        import shutil
        shutil.rmtree(output_tmp, ignore_errors=True)

        return float(macro_f1)

    # Optuna study with median pruning
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=config.RANDOM_STATE),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1),
        study_name="distilbert_oil_spill",
    )

    logger.info("Starting DistilBERT Optuna study (%d trials, %ds timeout)", n_trials, timeout)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    # Report
    logger.info("Best trial #%d — macro-F1: %.4f", study.best_trial.number, study.best_value)
    logger.info("Best params: %s", json.dumps(study.best_params, indent=2))

    # Save best params
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "best_hparams.json", "w") as f:
        json.dump({
            "macro_f1": float(study.best_value),
            "params": study.best_params,
        }, f, indent=2)

    return {
        "best_macro_f1": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }


# ═══════════════════════════════════════════════════════════════════════
# MLX LLM Hyperparameter Tuning (Qwen2.5-0.5B)
# ═══════════════════════════════════════════════════════════════════════

def tune_mlx_llm(
    n_trials: int = 15,
    timeout: int = 14400,         # 4 hours
    output_dir: str = "models/phi",
) -> dict:
    """Run Optuna hyperparameter optimization for Qwen2.5-0.5B MLX LoRA.

    Search space:
    - learning_rate: [5e-5, 5e-4] (log)
    - lora_rank: [4, 32]
    - lora_alpha: [8, 64]
    - lora_dropout: [0.05, 0.3]
    - epochs: [3, 10]
    - batch_size: {1, 2, 4}
    - warmup_steps: [5, 50]
    - weight_decay: [1e-4, 1e-1] (log)
    - num_lora_layers: [4, 16]
    - max_seq_length: {256, 512}
    - focal_gamma: [1.0, 4.0]

    Objective: validation macro-F1.

    Uses median pruning to stop bad trials after 1 epoch, saving ~60% of trials.
    Each trial runs 1-2 quick epochs for pruning; the best params get a full
    training run afterward.

    Returns:
        Dict with best params, best macro-F1, and study statistics.
    """
    data = _get_data()
    from sklearn.metrics import f1_score

    # Lazy MLX imports
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm import load as mlx_load
    from mlx_lm.tuner import linear_to_lora_layers
    from transformers import AutoTokenizer

    # Pre-tokenize data once (same for all trials)
    tokenizer = AutoTokenizer.from_pretrained(
        config.MLX_MODEL_ID, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _encode(texts, max_len):
        enc = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_len, return_tensors="np",
        )
        return enc["input_ids"]

    X_train_texts = data["X_train"]
    y_train_np = data["y_train"].astype(np.float32)
    X_val_texts = data["X_val"]
    y_val_np = data["y_val"]

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective — train MLX LLM for 1-2 epochs, return val macro-F1."""
        # Sample hyperparameters
        lr = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
        lora_rank = trial.suggest_int("lora_rank", 4, 32, step=4)
        lora_alpha = trial.suggest_int("lora_alpha", 8, 64, step=8)
        lora_dropout = trial.suggest_float("lora_dropout", 0.05, 0.3)
        quick_epochs = trial.suggest_int("quick_epochs", 2, 3)
        batch_size = trial.suggest_categorical("batch_size", [1, 2, 4])
        warmup_steps = trial.suggest_int("warmup_steps", 5, 50)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        num_lora_layers = trial.suggest_int("num_lora_layers", 4, 16, step=4)
        max_seq_length = trial.suggest_categorical("max_seq_length", [256, 512])
        focal_gamma = trial.suggest_float("focal_gamma", 1.0, 4.0)

        # Tokenize with trial's sequence length
        all_ids = _encode(X_train_texts, max_seq_length)
        val_ids = _encode(X_val_texts, max_seq_length)
        N_LABELS = y_train_np.shape[1]

        # Load base model
        base, _ = mlx_load(
            config.MLX_MODEL_ID,
            tokenizer_config={"trust_remote_code": True},
        )

        # Detect hidden size
        HIDDEN_SIZE = 896  # Qwen2.5-0.5B
        try:
            inner = getattr(base, "model", base)
            if hasattr(inner, "layers") and inner.layers:
                for _, child in inner.layers[0].named_modules():
                    w = getattr(child, "weight", None)
                    if w is not None and w.ndim == 2:
                        HIDDEN_SIZE = w.shape[-1]
                        break
        except Exception:
            pass

        # Apply LoRA
        lora_scale = lora_alpha / max(lora_rank, 1)
        linear_to_lora_layers(
            base, num_lora_layers,
            {"rank": lora_rank, "scale": lora_scale, "dropout": lora_dropout},
        )

        # Build model
        class OilSpillModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = base.model if hasattr(base, "model") else base
                self.norm = nn.LayerNorm(HIDDEN_SIZE)
                self.drop = nn.Dropout(lora_dropout)
                self.head = nn.Linear(HIDDEN_SIZE, N_LABELS)

            def __call__(self, ids):
                return self.head(self.drop(self.norm(self.encoder(ids).mean(axis=1))))

        model = OilSpillModel()

        # Alpha weights
        pos_counts = y_train_np.sum(axis=0)
        alpha = mx.array(np.clip(
            1.0 - (pos_counts + 1) / (len(y_train_np) + 1), 0.05, 0.99,
        ))

        def loss_fn(ids, targets):
            logits = model(ids)
            p = mx.sigmoid(logits)
            bce = mx.log(p + 1e-8) * targets + mx.log(1 - p + 1e-8) * (1 - targets)
            pt = p * targets + (1 - p) * (1 - targets)
            return -((1 - pt) ** focal_gamma * bce * alpha.reshape(1, -1)).mean()

        train_step = nn.value_and_grad(model, loss_fn)

        def scale_grads(g, factor):
            if isinstance(g, dict):
                return {k: scale_grads(v, factor) for k, v in g.items()}
            if isinstance(g, (list, tuple)):
                return type(g)(scale_grads(v, factor) for v in g)
            return g * factor

        # Compile
        val_init, grads_init = train_step(
            mx.array(all_ids[:1]), mx.array(y_train_np[:1]),
        )
        mx.eval(val_init, grads_init)

        # Optimizer
        opt = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)
        opt.update(model, scale_grads(grads_init, 1.0))
        mx.eval(model.parameters())

        # Train for quick_epochs (pruning after each epoch)
        rng = np.random.RandomState(config.RANDOM_STATE)

        for epoch in range(quick_epochs):
            indices = rng.permutation(len(X_train_texts))
            losses = []

            for i in range(0, len(X_train_texts), batch_size):
                micro = indices[i:i + batch_size]
                val_step, grads_step = train_step(
                    mx.array(all_ids[micro]), mx.array(y_train_np[micro]),
                )
                mx.eval(val_step, grads_step)
                losses.append(float(val_step))
                opt.update(model, scale_grads(grads_step, 1.0))
                mx.eval(model.parameters())

            avg_loss = float(np.mean(losses))

            # Quick validation after each epoch
            val_probs = []
            for i in range(0, len(X_val_texts), batch_size):
                p = mx.sigmoid(model(mx.array(val_ids[i:i + batch_size])))
                mx.eval(p)
                val_probs.append(np.array(p))
            yp = np.concatenate(val_probs, axis=0)
            yb = (yp >= 0.5).astype(int)
            macro_f1 = f1_score(y_val_np, yb, average="macro", zero_division=0)

            # Report to Optuna
            trial.report(macro_f1, epoch)

            # Prune if underperforming
            if trial.should_prune():
                logger.info(
                    "Trial %d pruned at epoch %d (macro-F1=%.4f)",
                    trial.number, epoch + 1, macro_f1,
                )
                raise optuna.TrialPruned()

            logger.info(
                "Trial %d epoch %d/%d — loss=%.4f, val_macro-F1=%.4f (lr=%.1e, r=%d)",
                trial.number, epoch + 1, quick_epochs, avg_loss, macro_f1, lr, lora_rank,
            )

        # Return final validation macro-F1
        val_probs = []
        for i in range(0, len(X_val_texts), batch_size):
            p = mx.sigmoid(model(mx.array(val_ids[i:i + batch_size])))
            mx.eval(p)
            val_probs.append(np.array(p))
        yp = np.concatenate(val_probs, axis=0)
        yb = (yp >= 0.5).astype(int)
        final_macro_f1 = f1_score(y_val_np, yb, average="macro", zero_division=0)

        return float(final_macro_f1)

    # Optuna study with median pruning
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=config.RANDOM_STATE),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1),
        study_name="mlx_llm_oil_spill",
    )

    logger.info(
        "Starting MLX LLM Optuna study (%d trials, %ds timeout). "
        "Each trial: 2-3 quick epochs. ~6 min/trial on M1 Pro.",
        n_trials, timeout,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    # Report
    logger.info("Best trial #%d — macro-F1: %.4f", study.best_trial.number, study.best_value)
    logger.info("Best params: %s", json.dumps(study.best_params, indent=2))

    # Save best params
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "best_hparams.json", "w") as f:
        json.dump({
            "best_macro_f1": float(study.best_value),
            "params": study.best_params,
        }, f, indent=2)

    return {
        "best_macro_f1": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "n_pruned": sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED),
    }


# ═══════════════════════════════════════════════════════════════════════
# Full training with best hyperparameters
# ═══════════════════════════════════════════════════════════════════════

def train_best_mlx_llm(
    best_params: dict,
    output_dir: str = "models/phi",
    n_epochs: int = 10,
) -> dict:
    """Train the MLX LLM with the best hyperparameters found by Optuna.

    Runs a full training with more epochs and saves the model.

    Returns:
        Dict with final test metrics.
    """
    data = _get_data()
    from sklearn.metrics import f1_score
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm import load as mlx_load
    from mlx_lm.tuner import linear_to_lora_layers
    from transformers import AutoTokenizer

    # Unpack best params
    lr = best_params.get("learning_rate", 2e-4)
    lora_rank = best_params.get("lora_rank", 8)
    lora_alpha = best_params.get("lora_alpha", 16)
    lora_dropout = best_params.get("lora_dropout", 0.1)
    batch_size = best_params.get("batch_size", 2)
    warmup_steps = best_params.get("warmup_steps", 20)
    weight_decay = best_params.get("weight_decay", 0.01)
    num_lora_layers = best_params.get("num_lora_layers", 8)
    max_seq_length = best_params.get("max_seq_length", 512)
    focal_gamma = best_params.get("focal_gamma", 2.0)

    logger.info("Training MLX LLM with best params: %s", json.dumps(best_params, indent=2))
    logger.info("Full training: %d epochs", n_epochs)

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(config.MLX_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _encode(texts):
        enc = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_seq_length, return_tensors="np",
        )
        return enc["input_ids"]

    all_ids = _encode(data["X_train"])
    test_ids = _encode(data["X_test"])
    y_train_f32 = data["y_train"].astype(np.float32)
    N_LABELS = data["y_train"].shape[1]

    # Model
    base, _ = mlx_load(config.MLX_MODEL_ID, tokenizer_config={"trust_remote_code": True})
    HIDDEN_SIZE = 896

    lora_scale = lora_alpha / max(lora_rank, 1)
    linear_to_lora_layers(
        base, num_lora_layers,
        {"rank": lora_rank, "scale": lora_scale, "dropout": lora_dropout},
    )

    class OilSpillModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = base.model if hasattr(base, "model") else base
            self.norm = nn.LayerNorm(HIDDEN_SIZE)
            self.drop = nn.Dropout(lora_dropout)
            self.head = nn.Linear(HIDDEN_SIZE, N_LABELS)
        def __call__(self, ids):
            return self.head(self.drop(self.norm(self.encoder(ids).mean(axis=1))))

    model = OilSpillModel()

    pos_counts = data["y_train"].sum(axis=0)
    alpha = mx.array(np.clip(1.0 - (pos_counts + 1) / (len(data["y_train"]) + 1), 0.05, 0.99))

    def loss_fn(ids, targets):
        logits = model(ids)
        p = mx.sigmoid(logits)
        bce = mx.log(p + 1e-8) * targets + mx.log(1 - p + 1e-8) * (1 - targets)
        pt = p * targets + (1 - p) * (1 - targets)
        return -((1 - pt) ** focal_gamma * bce * alpha.reshape(1, -1)).mean()

    train_step = nn.value_and_grad(model, loss_fn)

    def scale_grads(g, factor):
        if isinstance(g, dict): return {k: scale_grads(v, factor) for k, v in g.items()}
        if isinstance(g, (list, tuple)): return type(g)(scale_grads(v, factor) for v in g)
        return g * factor

    # Compile
    val_init, grads_init = train_step(mx.array(all_ids[:1]), mx.array(y_train_f32[:1]))
    mx.eval(val_init, grads_init)

    total_steps = (len(data["X_train"]) // batch_size) * n_epochs
    def lr_schedule(step):
        if step < warmup_steps:
            return lr * (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return lr * max(0.0, 1.0 - progress)

    opt = optim.AdamW(learning_rate=lr_schedule, weight_decay=weight_decay)
    opt.update(model, scale_grads(grads_init, 1.0))
    mx.eval(model.parameters())

    # Train
    rng = np.random.RandomState(config.RANDOM_STATE)
    import time
    t0 = time.time()
    step = 0

    for epoch in range(n_epochs):
        indices = rng.permutation(len(data["X_train"]))
        losses = []
        e0 = time.time()

        for i in range(0, len(data["X_train"]), batch_size):
            micro = indices[i:i + batch_size]
            val_step, grads_step = train_step(
                mx.array(all_ids[micro]), mx.array(y_train_f32[micro]),
            )
            mx.eval(val_step, grads_step)
            losses.append(float(val_step))
            opt.update(model, scale_grads(grads_step, 1.0))
            mx.eval(model.parameters())
            step += 1

        avg_loss = float(np.mean(losses))
        e1 = time.time()

        # Test eval
        test_probs = []
        for i in range(0, len(data["X_test"]), batch_size):
            p = mx.sigmoid(model(mx.array(test_ids[i:i + batch_size])))
            mx.eval(p)
            test_probs.append(np.array(p))
        yp = np.concatenate(test_probs, axis=0)
        yb = (yp >= 0.5).astype(int)
        mi = f1_score(data["y_test"], yb, average="micro", zero_division=0)
        ma = f1_score(data["y_test"], yb, average="macro", zero_division=0)

        logger.info(
            "Epoch %d/%d — loss=%.4f, micro-F1=%.4f, macro-F1=%.4f [%ds]",
            epoch + 1, n_epochs, avg_loss, mi, ma, int(e1 - e0),
        )

    total_t = (time.time() - t0) / 60
    logger.info("Full training complete in %.1f min", total_t)

    # Final eval + threshold tuning
    test_probs = []
    for i in range(0, len(data["X_test"]), batch_size):
        p = mx.sigmoid(model(mx.array(test_ids[i:i + batch_size])))
        mx.eval(p)
        test_probs.append(np.array(p))
    y_prob = np.concatenate(test_probs, axis=0)

    # Tune thresholds on validation set
    from .evaluate import find_optimal_thresholds, apply_thresholds

    val_probs = []
    val_ids_enc = _encode(data["X_val"])
    for i in range(0, len(data["X_val"]), batch_size):
        p = mx.sigmoid(model(mx.array(val_ids_enc[i:i + batch_size])))
        mx.eval(p)
        val_probs.append(np.array(p))
    val_prob = np.concatenate(val_probs, axis=0)

    thresholds, _ = find_optimal_thresholds(data["y_val"], val_prob, data["label_names"])
    y_pred_tuned = apply_thresholds(y_prob, thresholds)

    mi = f1_score(data["y_test"], y_pred_tuned, average="micro", zero_division=0)
    ma = f1_score(data["y_test"], y_pred_tuned, average="macro", zero_division=0)

    logger.info("Final (tuned thresholds): Micro-F1=%.4f, Macro-F1=%.4f", mi, ma)

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.numpy import save_file
        params = {
            k: np.array(v) for k, v in model.parameters().items()
            if any(x in k for x in ["lora_a", "lora_b", "head", "norm", "drop"])
        }
        if params:
            save_file(params, str(out / "adapters.safetensors"))
    except Exception:
        np.savez(out / "adapters.npz", **{k: np.array(v) for k, v in model.parameters().items()})

    with open(out / "metadata.json", "w") as f:
        import json as _json
        _json.dump({
            "thresholds": thresholds.tolist(),
            "label_names": data["label_names"],
            "hidden_size": HIDDEN_SIZE,
            "params": {
                "model_id": config.MLX_MODEL_ID,
                "n_labels": N_LABELS,
                "max_seq_length": max_seq_length,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "epochs": n_epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "focal_gamma": focal_gamma,
            },
            "training_result": {"micro_f1": mi, "macro_f1": ma},
        }, f, indent=2)

    tokenizer.save_pretrained(str(out))
    logger.info("Model saved to %s", output_dir)

    return {"micro_f1": mi, "macro_f1": ma, "thresholds": thresholds}
