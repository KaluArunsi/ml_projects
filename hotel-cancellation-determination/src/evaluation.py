import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import learning_curve
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .config import SEVERITY_BINS, SEVERITY_COLORS, SEVERITY_LABELS

logger = logging.getLogger(__name__)


def savefig(name: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("      saved -> %s", path)


def plot_class_distribution(y: pd.Series, output_dir: str) -> None:
    counts = y.value_counts().sort_index()
    labels = ["Not Cancelled", "Cancelled"]
    colors = ["steelblue", "tomato"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, counts.values, color=colors, edgecolor="white")
    axes[0].set_title("Booking Counts")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 200, f"{v:,}", ha="center", fontsize=10)
    axes[1].pie(counts.values, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Cancellation Split")
    fig.suptitle("Class Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("01_class_distribution", output_dir)


def plot_feature_distributions(df: pd.DataFrame, output_dir: str) -> None:
    num_cols = [
        "lead_time", "adr", "stays_in_weekend_nights", "stays_in_week_nights",
        "adults", "children", "days_in_waiting_list",
        "previous_cancellations", "booking_changes", "total_of_special_requests",
    ]
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for ax, col in zip(axes.flat, num_cols):
        ax.hist(df[col].dropna(), bins=40, color="steelblue", edgecolor="white")
        ax.set_title(col.replace("_", " ").title(), fontsize=9)
    fig.suptitle("Numerical Feature Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("02_feature_distributions", output_dir)


def plot_lr_evaluation(y_test, y_pred, y_prob, output_dir: str) -> None:
    from sklearn.metrics import classification_report, roc_auc_score

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec, rec, _ = precision_recall_curve(y_test, y_prob)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ConfusionMatrixDisplay(cm, display_labels=["Not Cancelled", "Cancelled"]).plot(
        ax=axes[0, 0], colorbar=False, cmap="Blues"
    )
    axes[0, 0].set_title("Confusion Matrix")

    auc = roc_auc_score(y_test, y_prob)
    axes[0, 1].plot(fpr, tpr, color="royalblue", lw=2, label=f"ROC-AUC = {auc:.3f}")
    axes[0, 1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0, 1].set_xlabel("False Positive Rate")
    axes[0, 1].set_ylabel("True Positive Rate")
    axes[0, 1].set_title("ROC Curve")
    axes[0, 1].legend()

    ap = average_precision_score(y_test, y_prob)
    axes[1, 0].plot(rec, prec, color="darkorange", lw=2, label=f"AP = {ap:.3f}")
    axes[1, 0].axhline(y_test.mean(), color="gray", linestyle="--", lw=1, label=f"Baseline = {y_test.mean():.2f}")
    axes[1, 0].set_xlabel("Recall")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_title("Precision-Recall Curve")
    axes[1, 0].legend()

    report = classification_report(y_test, y_pred, output_dict=True)
    x = np.arange(2)
    width = 0.25
    for i, m in enumerate(["precision", "recall", "f1-score"]):
        axes[1, 1].bar(x + i * width, [report[c][m] for c in ["0", "1"]], width, label=m.capitalize())
    axes[1, 1].set_xticks(x + width)
    axes[1, 1].set_xticklabels(["Not Cancelled", "Cancelled"])
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_title("Per-Class Precision / Recall / F1")
    axes[1, 1].legend()

    fig.suptitle("Logistic Regression Evaluation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    savefig("03_lr_evaluation", output_dir)


def plot_lr_learning_curve(X_train, y_train, output_dir: str) -> None:
    train_sizes, train_scores, val_scores = learning_curve(
        LogisticRegression(max_iter=1000, random_state=42),
        X_train, y_train,
        cv=5, scoring="f1", n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 8),
    )
    _plot_learning_curve(
        train_sizes, train_scores, val_scores,
        title="Learning Curve — Logistic Regression",
        ylabel="F1 Score",
        train_label="Train F1",
        val_label="CV F1 (5-fold)",
        filename="04_lr_learning_curve",
        output_dir=output_dir,
    )


def plot_severity_distribution(y_prob_full: np.ndarray, output_dir: str) -> None:
    tier_short = ["Very Unlikely", "Unlikely", "Likely", "Most Likely"]
    tiers = pd.cut(y_prob_full, bins=SEVERITY_BINS, labels=[0, 1, 2, 3]).astype(int)
    counts = pd.Series(tiers).value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(y_prob_full, bins=60, color="steelblue", edgecolor="white")
    for b in SEVERITY_BINS[1:-1]:
        axes[0].axvline(b, color="crimson", linestyle="--", lw=1.5)
    axes[0].set_xlabel("P(Cancel)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Cancellation Probability Distribution")

    bars = axes[1].bar(tier_short, counts.values, color=SEVERITY_COLORS, edgecolor="white")
    for bar, v in zip(bars, counts.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 30, f"{v:,}", ha="center", fontsize=9)
    axes[1].set_xlabel("Severity Tier")
    axes[1].set_ylabel("Bookings")
    axes[1].set_title("Bookings per Severity Tier")

    fig.suptitle("Severity Scoring", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("05_severity_distribution", output_dir)


def plot_xgb_evaluation(y_test_sev, y_pred_sev, feature_names, importances, output_dir: str) -> None:
    from sklearn.metrics import classification_report

    tier_short = ["Very\nUnlikely", "Unlikely", "Likely", "Most\nLikely"]
    cm = confusion_matrix(y_test_sev, y_pred_sev)
    report = classification_report(y_test_sev, y_pred_sev, output_dict=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ConfusionMatrixDisplay(cm, display_labels=["T0", "T1", "T2", "T3"]).plot(
        ax=axes[0], colorbar=False, cmap="Blues"
    )
    axes[0].set_title("Confusion Matrix (4 Tiers)")

    x = np.arange(4)
    width = 0.25
    for i, m in enumerate(["precision", "recall", "f1-score"]):
        axes[1].bar(x + i * width, [report[str(c)][m] for c in range(4)], width, label=m.capitalize())
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(tier_short)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Per-Tier Precision / Recall / F1")
    axes[1].legend()

    top_n = 15
    idx = np.argsort(importances)[-top_n:]
    axes[2].barh(np.array(feature_names)[idx], importances[idx], color="steelblue")
    axes[2].set_title("Top 15 Feature Importances")
    axes[2].set_xlabel("Gain")

    fig.suptitle("XGBoost Severity Classifier Evaluation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("06_xgb_evaluation", output_dir)


def plot_xgb_confidence(y_test_sev, y_prob_sev, output_dir: str) -> None:
    tier_short = ["Very Unlikely", "Unlikely", "Likely", "Most Likely"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for t in range(4):
        mask = np.asarray(y_test_sev, dtype=int) == t
        if mask.sum() > 0:
            axes[t].hist(y_prob_sev[mask, t], bins=30, color=SEVERITY_COLORS[t], edgecolor="white")
        axes[t].set_title(f"Tier {t}: {tier_short[t]}", fontsize=9)
        axes[t].set_xlabel("Confidence")
        axes[t].set_ylabel("Count")
    fig.suptitle("XGBoost Confidence Distributions", fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig("07_xgb_confidence", output_dir)


def plot_xgb_pr_curves(y_test_sev, y_prob_sev, output_dir: str) -> None:
    tier_short = ["Very Unlikely", "Unlikely", "Likely", "Most Likely"]
    y_bin = np.eye(4, dtype=int)[np.asarray(y_test_sev, dtype=int)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ap_scores = []
    for t in range(4):
        prec, rec, _ = precision_recall_curve(y_bin[:, t], y_prob_sev[:, t])
        ap = average_precision_score(y_bin[:, t], y_prob_sev[:, t])
        ap_scores.append(ap)
        axes[0].plot(rec, prec, color=SEVERITY_COLORS[t], lw=2, label=f"Tier {t} (AP={ap:.2f})")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("PR Curves — One-vs-Rest")
    axes[0].legend(fontsize=9)

    axes[1].bar(tier_short, ap_scores, color=SEVERITY_COLORS, edgecolor="white")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Average Precision")
    axes[1].set_title("Average Precision per Tier")
    for i, v in enumerate(ap_scores):
        axes[1].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)

    fig.suptitle("XGBoost Precision-Recall Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("08_xgb_pr_curves", output_dir)


def plot_xgb_learning_curve(X_train, y_train_sev, output_dir: str) -> None:
    xgb_lc = XGBClassifier(
        objective="multi:softprob", num_class=4,
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, eval_metric="mlogloss", verbosity=0,
    )
    fit_params = {"sample_weight": compute_sample_weight("balanced", y_train_sev)}
    train_sizes, train_scores, val_scores = learning_curve(
        xgb_lc, X_train, y_train_sev,
        cv=3, scoring="f1_macro", n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 6),
        fit_params=fit_params,
    )
    _plot_learning_curve(
        train_sizes, train_scores, val_scores,
        title="Learning Curve — XGBoost Severity Classifier",
        ylabel="F1 Macro",
        train_label="Train F1 macro",
        val_label="CV F1 macro (3-fold)",
        filename="09_xgb_learning_curve",
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _plot_learning_curve(train_sizes, train_scores, val_scores,
                          title, ylabel, train_label, val_label, filename, output_dir):
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    val_mean, val_std     = val_scores.mean(axis=1),   val_scores.std(axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, "o-", color="royalblue", label=train_label)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="royalblue")
    ax.plot(train_sizes, val_mean, "o-", color="darkorange", label=val_label)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="darkorange")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    savefig(filename, output_dir)
