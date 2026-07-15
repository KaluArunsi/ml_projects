"""
Evaluation Module
=================
Shared multi-label metrics, per-label threshold tuning, and visualization.
Used across all three modeling tracks.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    label_names: list[str],
) -> dict:
    """Compute comprehensive per-label and aggregate metrics.

    Args:
        y_true: Ground-truth binary matrix (n_samples, n_labels).
        y_pred: Binary predictions (n_samples, n_labels).
        y_prob: Predicted probabilities (n_samples, n_labels).
        label_names: Ordered label names.

    Returns:
        Nested dict with keys:
        - ``per_label``: DataFrame with per-label precision, recall, f1, support, roc_auc.
        - ``micro``: dict of micro-averaged metrics.
        - ``macro``: dict of macro-averaged metrics.
        - ``hamming_loss``: float.
        - ``subset_accuracy``: float (exact match ratio).
    """
    n_labels = y_true.shape[1]
    per_label_rows = []

    for j in range(n_labels):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        yprob = y_prob[:, j] if y_prob.shape[1] > j else None

        p, r, f1, s = precision_recall_fscore_support(
            yt, yp, average="binary", zero_division=0,
        )
        support = int(yt.sum())

        roc = None
        if yprob is not None:
            try:
                roc = roc_auc_score(yt, yprob)
            except ValueError:
                roc = None  # only one class present

        per_label_rows.append({
            "label": label_names[j] if j < len(label_names) else f"label_{j}",
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f1), 4),
            "support": support,
            "roc_auc": round(float(roc), 4) if roc is not None else None,
        })

    per_label_df = pd.DataFrame(per_label_rows).set_index("label")

    # Aggregate
    micro_prec, micro_rec, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0,
    )
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0,
    )
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    ham_loss = hamming_loss(y_true, y_pred)
    subset_acc = accuracy_score(y_true, y_pred)  # exact match for multi-label

    return {
        "per_label": per_label_df,
        "micro": {"precision": micro_prec, "recall": micro_rec, "f1": micro_f1},
        "macro": {"precision": macro_prec, "recall": macro_rec, "f1": macro_f1},
        "weighted_f1": weighted_f1,
        "hamming_loss": ham_loss,
        "subset_accuracy": subset_acc,
    }


def format_metrics_summary(metrics: dict) -> str:
    """Return a human-readable summary string for a metrics dict."""
    lines = [
        "=" * 60,
        "EVALUATION RESULTS",
        "=" * 60,
        f"  Micro-F1:      {metrics['micro']['f1']:.4f}",
        f"  Macro-F1:      {metrics['macro']['f1']:.4f}",
        f"  Weighted-F1:   {metrics['weighted_f1']:.4f}",
        f"  Hamming Loss:  {metrics['hamming_loss']:.4f}",
        f"  Subset Acc:    {metrics['subset_accuracy']:.4f}",
        "",
        "Per-label:",
    ]
    for label, row in metrics["per_label"].iterrows():
        lines.append(
            f"  {label:<22s}  P={row['precision']:.3f}  R={row['recall']:.3f}  "
            f"F1={row['f1']:.3f}  (n={int(row['support'])})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def find_optimal_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: list[str] | None = None,
    metric: str = config.THRESHOLD_METRIC,
    n_steps: int = config.THRESHOLD_STEPS,
) -> tuple[np.ndarray, dict[str, float]]:
    """Per-label grid search for optimal decision thresholds.

    Args:
        y_true: Ground-truth binary matrix.
        y_prob: Predicted probabilities.
        label_names: Label names for logging.
        metric: "f1" or "f2" to optimize.
        n_steps: Number of grid points in [THRESHOLD_MIN, THRESHOLD_MAX].

    Returns:
        (thresholds_array, per_label_best_f1_dict).
    """
    n_labels = y_true.shape[1]
    grid = np.linspace(config.THRESHOLD_MIN, config.THRESHOLD_MAX, n_steps)
    best_thresholds = np.full(n_labels, 0.5)
    best_scores: dict[str, float] = {}

    for j in range(n_labels):
        yt = y_true[:, j]
        best_t, best_score = 0.5, 0.0

        for t in grid:
            yp = (y_prob[:, j] >= t).astype(int)
            if metric == "f2":
                score = f1_score(yt, yp, average="binary", zero_division=0, beta=2.0)
            else:
                score = f1_score(yt, yp, average="binary", zero_division=0)

            if score > best_score:
                best_score = score
                best_t = t

        best_thresholds[j] = best_t
        name = label_names[j] if label_names else f"label_{j}"
        best_scores[name] = float(best_score)

    if label_names:
        logger.info("Optimal thresholds: %s",
                     {n: f"{t:.3f}" for n, t in zip(label_names, best_thresholds)})

    return best_thresholds, best_scores


def apply_thresholds(y_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Apply per-label thresholds to probability matrix."""
    return (y_prob >= thresholds).astype(int)


# ---------------------------------------------------------------------------
# Visualization (matplotlib)
# ---------------------------------------------------------------------------

def _ensure_output_dir(output_dir: str) -> Path:
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_per_label_f1(
    metrics: dict,
    output_dir: str = "output/plots",
    model_name: str = "model",
) -> str:
    """Bar chart of per-label F1 scores."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style(config.PLOT_STYLE)

    df = metrics["per_label"].copy()
    df = df.sort_values("f1")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df.index, df["f1"], color="steelblue", edgecolor="white")
    ax.axvline(metrics["micro"]["f1"], color="darkred", linestyle="--",
               label=f"Micro-F1: {metrics['micro']['f1']:.3f}")
    ax.axvline(metrics["macro"]["f1"], color="darkorange", linestyle="--",
               label=f"Macro-F1: {metrics['macro']['f1']:.3f}")
    ax.set_xlabel("F1 Score")
    ax.set_title(f"Per-Label F1 Scores — {model_name}")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)

    # Annotate bars
    for bar, val in zip(bars, df["f1"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    out = _ensure_output_dir(output_dir)
    path = out / f"{model_name}_per_label_f1.png"
    fig.tight_layout()
    fig.savefig(path, dpi=config.PLOT_DPI)
    plt.close(fig)
    return str(path)


def plot_model_comparison(
    all_metrics: dict[str, dict],
    output_dir: str = "output/plots",
) -> str:
    """Side-by-side bar chart comparing macro-F1 and micro-F1 across models."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style(config.PLOT_STYLE)

    models = list(all_metrics.keys())
    micro_f1s = [all_metrics[m]["micro"]["f1"] for m in models]
    macro_f1s = [all_metrics[m]["macro"]["f1"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, micro_f1s, width, label="Micro-F1", color="steelblue")
    ax.bar(x + width / 2, macro_f1s, width, label="Macro-F1", color="darkorange")

    ax.set_ylabel("F1 Score")
    ax.set_title("Model Comparison — Oil Spill Cause Classification")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1)

    for i, (mic, mac) in enumerate(zip(micro_f1s, macro_f1s)):
        ax.text(i - width / 2, mic + 0.01, f"{mic:.3f}", ha="center", fontsize=10)
        ax.text(i + width / 2, mac + 0.01, f"{mac:.3f}", ha="center", fontsize=10)

    out = _ensure_output_dir(output_dir)
    path = out / "model_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=config.PLOT_DPI)
    plt.close(fig)
    return str(path)
