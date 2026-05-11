import logging
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from .config import CLUSTER_FEATURES, OCC_PROB_BINS, OCC_PROB_COLORS, OCC_PROB_LABELS
from .models import compute_prg_curve

logger = logging.getLogger(__name__)

_PALETTE = {"Repaid": "steelblue", "Defaulted": "tomato"}
_MODEL_COLORS = ["royalblue", "darkorange", "seagreen", "mediumpurple"]


def _savefig(name: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("      saved -> %s", path)


def _plot_4panel_eval(y_test, y_pred, y_prob, title: str, filename: str, output_dir: str) -> None:
    y_test_arr = np.asarray(y_test)
    cm = confusion_matrix(y_test_arr, y_pred)
    fpr, tpr, _ = roc_curve(y_test_arr, y_prob)
    prec, rec, _ = precision_recall_curve(y_test_arr, y_prob)
    pg, rg, auprg = compute_prg_curve(y_test_arr, y_prob)
    auc = roc_auc_score(y_test_arr, y_prob)
    pi = y_test_arr.mean()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ConfusionMatrixDisplay(cm, display_labels=["Repaid", "Defaulted"]).plot(
        ax=axes[0, 0], colorbar=False, cmap="Blues"
    )
    axes[0, 0].set_title("Confusion Matrix")

    axes[0, 1].plot(fpr, tpr, color="royalblue", lw=2, label=f"ROC-AUC = {auc:.4f}")
    axes[0, 1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0, 1].set_xlabel("False Positive Rate")
    axes[0, 1].set_ylabel("True Positive Rate")
    axes[0, 1].set_title("ROC Curve")
    axes[0, 1].legend()

    axes[1, 0].plot(rec, prec, color="darkorange", lw=2, label="PR curve")
    axes[1, 0].axhline(pi, color="gray", linestyle="--", lw=1, label=f"Baseline = {pi:.2f}")
    axes[1, 0].set_xlabel("Recall")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_title("Precision-Recall Curve")
    axes[1, 0].legend()

    order = np.argsort(rg)
    axes[1, 1].plot(rg[order], pg[order], color="seagreen", lw=2, label=f"AUPRG = {auprg:.4f}")
    axes[1, 1].fill_between(rg[order], pg[order], alpha=0.1, color="seagreen")
    axes[1, 1].set_xlabel("Recall Gain")
    axes[1, 1].set_ylabel("Precision Gain")
    axes[1, 1].set_title("Precision-Recall Gain Curve")
    axes[1, 1].legend()

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    _savefig(filename, output_dir)


def _plot_importances(importances, feature_names, top_n: int, title: str,
                      filename: str, output_dir: str, color: str = "steelblue") -> None:
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.35)))
    ax.barh(np.array(feature_names)[idx], np.array(importances)[idx], color=color)
    ax.set_xlabel("Importance")
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    _savefig(filename, output_dir)


# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------

def plot_class_distribution(y: pd.Series, output_dir: str) -> None:
    counts = y.value_counts().sort_index()
    labels = ["Repaid (0)", "Defaulted (1)"]
    colors = ["steelblue", "tomato"]
    total = counts.sum()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bars = axes[0].bar(labels, counts.values, color=colors, edgecolor="white")
    for bar, v in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 30,
                     f"{v:,} ({v/total*100:.1f}%)", ha="center", fontsize=10)
    axes[0].set_title("Loan Counts")
    axes[0].set_ylabel("Count")

    axes[1].pie(counts.values, labels=labels, colors=colors,
                autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Default Split")

    fig.suptitle("Class Distribution — 9,578 Loans", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig("01_class_distribution", output_dir)


def plot_univariate_numeric(df: pd.DataFrame, output_dir: str) -> None:
    num_cols = [
        "int.rate", "installment", "dti", "fico", "days.with.cr.line",
        "revol.bal", "revol.util", "inq.last.6mths", "monthly_income",
        "payment_to_income_ratio", "dscr_proxy", "debt_burden_annual",
    ]
    cols_present = [c for c in num_cols if c in df.columns]
    n = len(cols_present)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.5))
    for ax, col in zip(axes.flat, cols_present):
        data = df[col].dropna()
        ax.hist(data, bins=50, color="steelblue", edgecolor="white", alpha=0.8, density=True)
        try:
            data.plot.kde(ax=ax, color="navy", lw=1.5)
        except Exception:
            pass
        ax.axvline(data.median(), color="crimson", linestyle="--", lw=1.2, label="Median")
        ax.axvline(data.mean(), color="darkorange", linestyle=":", lw=1.2, label="Mean")
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Univariate Numeric Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig("02_univariate_numeric", output_dir)


def plot_categorical_analysis(df: pd.DataFrame, output_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    purpose_counts = df["purpose"].value_counts()
    axes[0].bar(purpose_counts.index, purpose_counts.values, color="steelblue", edgecolor="white")
    axes[0].set_title("Loan Purpose Counts")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=35)

    purpose_dr = df.groupby("purpose")["cannot_pay_back"].mean().sort_values(ascending=False)
    overall_dr = df["cannot_pay_back"].mean()
    bar_colors = ["tomato" if v > overall_dr else "steelblue" for v in purpose_dr.values]
    axes[1].bar(purpose_dr.index, purpose_dr.values, color=bar_colors, edgecolor="white")
    axes[1].axhline(overall_dr, color="gray", linestyle="--", lw=1.5, label=f"Overall {overall_dr:.2%}")
    axes[1].set_title("Default Rate by Purpose")
    axes[1].set_ylabel("Default Rate")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].legend()

    if "occ_tier" in df.columns:
        tier_order = ["Doubtful/Loss", "Substandard", "Special Mention", "Pass"]
        tier_order = [t for t in tier_order if t in df["occ_tier"].astype(str).unique()]
        tier_counts = df["occ_tier"].astype(str).value_counts().reindex(tier_order, fill_value=0)
        tier_dr = df.groupby(df["occ_tier"].astype(str))["cannot_pay_back"].mean().reindex(tier_order, fill_value=0)
        ax2 = axes[2]
        bars = ax2.bar(tier_order, tier_counts.values, color=["#e74c3c", "#e67e22", "#f39c12", "#27ae60"],
                       edgecolor="white")
        ax2b = ax2.twinx()
        ax2b.plot(tier_order, tier_dr.values, "o--", color="navy", lw=2, label="Default rate")
        ax2b.set_ylabel("Default Rate")
        ax2b.legend(loc="upper right", fontsize=8)
        ax2.set_title("OCC FICO Tier Distribution")
        ax2.set_ylabel("Count")
        ax2.tick_params(axis="x", rotation=20)

    fig.suptitle("Categorical Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig("03_categorical_analysis", output_dir)


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    num_df = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in ["lof_flag", "mahal_flag"] if c in df.columns]
    )
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 6},
                linewidths=0.3, ax=ax)
    ax.set_title("Pearson Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig("04_correlation_heatmap", output_dir)


def plot_boxplots_by_target(df: pd.DataFrame, output_dir: str) -> None:
    features = ["fico", "int.rate", "monthly_income", "dti",
                "dscr_proxy", "payment_to_income_ratio",
                "revol.util", "inq.last.6mths", "days.with.cr.line"]
    features = [f for f in features if f in df.columns]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, feat in zip(axes.flat, features):
        for val, label, color in [(0, "Repaid", "steelblue"), (1, "Defaulted", "tomato")]:
            data = df.loc[df["cannot_pay_back"] == val, feat].dropna()
            ax.boxplot(data, positions=[val], widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor=color, alpha=0.6),
                       medianprops=dict(color="black", lw=2),
                       whiskerprops=dict(color=color),
                       capprops=dict(color=color),
                       flierprops=dict(marker=".", alpha=0.3, markersize=3))
            ax.scatter([val], [data.mean()], marker="D", color="gold",
                       edgecolor="black", zorder=5, s=40, label=f"{label} mean" if val == 0 else None)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Repaid", "Defaulted"])
        ax.set_title(feat, fontsize=9, fontweight="bold")
    for ax in axes.flat[len(features):]:
        ax.set_visible(False)
    fig.suptitle("Feature Distributions by Default Outcome", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig("05_boxplots_by_target", output_dir)


def plot_violin_capacity(df: pd.DataFrame, output_dir: str) -> None:
    features = ["dscr_proxy", "payment_to_income_ratio", "dti", "log.annual.inc"]
    features = [f for f in features if f in df.columns]
    labels = {"dscr_proxy": "DSCR Proxy", "payment_to_income_ratio": "Payment-to-Income",
               "dti": "DTI", "log.annual.inc": "Log Annual Income"}

    fig, axes = plt.subplots(1, len(features), figsize=(14, 5))
    for ax, feat in zip(axes, features):
        plot_df = df[[feat, "cannot_pay_back"]].copy()
        plot_df["Class"] = plot_df["cannot_pay_back"].map({0: "Repaid", 1: "Defaulted"})
        sns.violinplot(data=plot_df, x="Class", y=feat, palette=_PALETTE,
                       inner="box", ax=ax, order=["Repaid", "Defaulted"])
        ax.set_title(labels.get(feat, feat), fontsize=10, fontweight="bold")
        ax.set_xlabel("")
    fig.suptitle("Capacity Feature Distributions by Default Outcome", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig("06_violin_capacity", output_dir)


def plot_purpose_stacked(df: pd.DataFrame, output_dir: str) -> None:
    purpose_df = (
        df.groupby("purpose")["cannot_pay_back"]
        .value_counts(normalize=True)
        .unstack()
        .rename(columns={0: "Repaid", 1: "Defaulted"})
        .sort_values("Defaulted", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    purpose_df[["Repaid", "Defaulted"]].plot(
        kind="bar", stacked=True, color=["steelblue", "tomato"],
        edgecolor="white", ax=ax
    )
    overall_dr = df["cannot_pay_back"].mean()
    ax.axhline(1 - overall_dr, color="gray", linestyle="--", lw=1.2,
               label=f"Overall repaid rate {1-overall_dr:.1%}")
    ax.set_title("Default Rate by Loan Purpose", fontsize=13, fontweight="bold")
    ax.set_ylabel("Proportion")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    plt.tight_layout()
    _savefig("07_purpose_stacked", output_dir)


def plot_pairplot(df: pd.DataFrame, output_dir: str) -> None:
    features = ["fico", "int.rate", "dti", "inq.last.6mths", "log.annual.inc"]
    features = [f for f in features if f in df.columns]
    sample = df[features + ["cannot_pay_back"]].sample(n=min(1500, len(df)), random_state=42)
    sample["Class"] = sample["cannot_pay_back"].map({0: "Repaid", 1: "Defaulted"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = sns.pairplot(
            sample, vars=features, hue="Class",
            palette=_PALETTE, diag_kind="kde",
            plot_kws={"alpha": 0.3, "s": 10},
        )
    g.figure.suptitle("Pair Plot — Top 5 Discriminating Features", y=1.02, fontweight="bold")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "08_pairplot.png")
    g.figure.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(g.figure)
    logger.info("      saved -> %s", path)


# ---------------------------------------------------------------------------
# Outlier detection plots
# ---------------------------------------------------------------------------

def plot_lof_outliers(df: pd.DataFrame, lof_scores: np.ndarray, output_dir: str) -> None:
    cols = [c for c in CLUSTER_FEATURES if c in df.columns]
    X_sc = StandardScaler().fit_transform(df[cols].values.astype(float))
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_sc)

    flag = df["lof_flag"].values if "lof_flag" in df.columns else np.zeros(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    normal_mask = flag == 0
    axes[0].scatter(coords[normal_mask, 0], coords[normal_mask, 1],
                    c="steelblue", s=5, alpha=0.4, label="Normal")
    axes[0].scatter(coords[~normal_mask, 0], coords[~normal_mask, 1],
                    c="tomato", s=20, alpha=0.8, label=f"LOF outlier (n={int((~normal_mask).sum())})")
    axes[0].set_xlabel("PC 1")
    axes[0].set_ylabel("PC 2")
    axes[0].set_title("LOF Outliers in PCA Space")
    axes[0].legend(fontsize=9)

    axes[1].hist(lof_scores[normal_mask], bins=60, color="steelblue",
                 edgecolor="white", alpha=0.7, label="Normal", density=True)
    axes[1].hist(lof_scores[~normal_mask], bins=30, color="tomato",
                 edgecolor="white", alpha=0.8, label="LOF outlier", density=True)
    axes[1].set_xlabel("LOF Score")
    axes[1].set_ylabel("Density")
    axes[1].set_title("LOF Score Distribution")
    axes[1].legend()

    fig.suptitle("Local Outlier Factor Detection", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig("09_lof_outliers", output_dir)


def plot_mahalanobis(df: pd.DataFrame, D2: np.ndarray, cutoff: float, output_dir: str) -> None:
    flag = (D2 > cutoff).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(D2[flag == 0], bins=80, color="steelblue", edgecolor="white",
                 alpha=0.7, label="Normal", density=True)
    axes[0].hist(D2[flag == 1], bins=30, color="tomato", edgecolor="white",
                 alpha=0.8, label=f"Flagged (n={int(flag.sum())})", density=True)
    axes[0].axvline(cutoff, color="crimson", linestyle="--", lw=2,
                    label=f"χ² 99.9% threshold = {cutoff:.1f}")
    axes[0].set_xlabel("Squared Mahalanobis Distance")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Mahalanobis Distance Distribution")
    axes[0].legend(fontsize=9)

    axes[1].boxplot(
        [D2[flag == 0], D2[flag == 1]],
        labels=["Normal", "Flagged"],
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.6),
        medianprops=dict(color="black", lw=2),
    )
    axes[1].axhline(cutoff, color="crimson", linestyle="--", lw=1.5, label=f"Threshold = {cutoff:.1f}")
    axes[1].set_ylabel("Squared Mahalanobis Distance")
    axes[1].set_title("Distance by Flag Status")
    axes[1].legend(fontsize=9)

    fig.suptitle("Mahalanobis Distance Outlier Detection", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig("10_mahalanobis", output_dir)


# ---------------------------------------------------------------------------
# Clustering plots
# ---------------------------------------------------------------------------

def plot_cluster_selection(k_range: list, inertias: list, sil_scores: list, output_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(k_range, inertias, "o-", color="steelblue", lw=2)
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method")
    axes[0].set_xticks(k_range)

    axes[1].plot(k_range, sil_scores, "o-", color="darkorange", lw=2)
    best_k = k_range[int(np.argmax(sil_scores))]
    axes[1].axvline(best_k, color="crimson", linestyle="--", lw=1.5, label=f"Best k = {best_k}")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xticks(k_range)
    axes[1].legend()

    fig.suptitle("K-Means Cluster Selection", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig("11_cluster_selection", output_dir)


def plot_cluster_pca(df: pd.DataFrame, output_dir: str) -> None:
    cols = [c for c in CLUSTER_FEATURES if c in df.columns]
    X_sc = StandardScaler().fit_transform(df[cols].values.astype(float))
    coords = PCA(n_components=2, random_state=42).fit_transform(X_sc)
    clusters = df["cluster_label"].values
    targets = df["cannot_pay_back"].values

    n_clusters = len(np.unique(clusters))
    cluster_colors = plt.cm.tab10(np.linspace(0, 0.5, n_clusters))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for c in np.unique(clusters):
        mask = clusters == c
        axes[0].scatter(coords[mask, 0], coords[mask, 1],
                        color=cluster_colors[c], s=5, alpha=0.5, label=f"Cluster {c}")
    axes[0].set_xlabel("PC 1")
    axes[0].set_ylabel("PC 2")
    axes[0].set_title("K-Means Clusters")
    axes[0].legend(markerscale=3, fontsize=9)

    for val, label, color in [(0, "Repaid", "steelblue"), (1, "Defaulted", "tomato")]:
        mask = targets == val
        axes[1].scatter(coords[mask, 0], coords[mask, 1],
                        color=color, s=5, alpha=0.4, label=label)
    axes[1].set_xlabel("PC 1")
    axes[1].set_ylabel("PC 2")
    axes[1].set_title("Default Outcome Overlay")
    axes[1].legend(markerscale=3, fontsize=9)

    fig.suptitle("K-Means Clusters in PCA Space", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig("12_cluster_pca", output_dir)


def plot_cluster_profiles(df: pd.DataFrame, output_dir: str) -> None:
    profile_cols = ["fico", "int.rate", "dti", "monthly_income",
                    "payment_to_income_ratio", "revol.util"]
    profile_cols = [c for c in profile_cols if c in df.columns]

    cluster_dr = df.groupby("cluster_label")["cannot_pay_back"].mean()
    n_clusters = cluster_dr.shape[0]
    colors = plt.cm.tab10(np.linspace(0, 0.5, n_clusters))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bars = axes[0].bar(cluster_dr.index.astype(str), cluster_dr.values,
                       color=colors, edgecolor="white")
    axes[0].axhline(df["cannot_pay_back"].mean(), color="gray", linestyle="--",
                    lw=1.5, label=f"Overall {df['cannot_pay_back'].mean():.2%}")
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("Default Rate")
    axes[0].set_title("Default Rate per Cluster")
    axes[0].legend()
    for bar, v in zip(bars, cluster_dr.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                     f"{v:.2%}", ha="center", fontsize=10)

    profile_means = df.groupby("cluster_label")[profile_cols].mean()
    profile_norm = (profile_means - profile_means.min()) / (profile_means.max() - profile_means.min() + 1e-9)
    x = np.arange(len(profile_cols))
    width = 0.25
    for i, (cid, row) in enumerate(profile_norm.iterrows()):
        axes[1].bar(x + i * width, row.values, width, color=colors[i],
                    label=f"Cluster {cid}", edgecolor="white")
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels([c.replace("_", "\n") for c in profile_cols], fontsize=8)
    axes[1].set_ylabel("Normalised Mean")
    axes[1].set_title("Cluster Feature Profiles (normalised)")
    axes[1].legend(fontsize=9)

    fig.suptitle("Cluster Risk Profiles", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig("13_cluster_profiles", output_dir)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def plot_smoteenn(y_train: np.ndarray, y_res: np.ndarray, output_dir: str) -> None:
    labels = ["Repaid (0)", "Defaulted (1)"]
    before = pd.Series(y_train).value_counts().sort_index().values
    after = pd.Series(y_res).value_counts().sort_index().values

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, before, color=["steelblue", "tomato"], edgecolor="white")
    for i, v in enumerate(before):
        axes[0].text(i, v + 10, f"{v:,}", ha="center", fontsize=10)
    axes[0].set_title("Before SMOTE-ENN")
    axes[0].set_ylabel("Count")

    axes[1].bar(labels, after, color=["steelblue", "tomato"], edgecolor="white")
    for i, v in enumerate(after):
        axes[1].text(i, v + 10, f"{v:,}", ha="center", fontsize=10)
    axes[1].set_title("After SMOTE-ENN")
    axes[1].set_ylabel("Count")

    fig.suptitle("SMOTE-ENN Class Resampling", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig("14_smoteenn", output_dir)


# ---------------------------------------------------------------------------
# Model evaluation plots
# ---------------------------------------------------------------------------

def plot_lr_evaluation(y_test, y_pred, y_prob, output_dir: str) -> None:
    _plot_4panel_eval(y_test, y_pred, y_prob,
                      "Logistic Regression Evaluation", "15_lr_eval", output_dir)


def plot_lr_coefficients(lr, feature_names: list, output_dir: str) -> None:
    coefs = lr.coef_[0]
    top_n = min(20, len(coefs))
    idx = np.argsort(np.abs(coefs))[-top_n:]
    top_coefs = coefs[idx]
    top_names = np.array(feature_names)[idx]
    colors = ["tomato" if c > 0 else "seagreen" for c in top_coefs]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.35)))
    ax.barh(top_names, top_coefs, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Coefficient (log-odds scale)")
    ax.set_title("Logistic Regression — Top 20 Coefficients", fontweight="bold")
    plt.tight_layout()
    _savefig("16_lr_coefficients", output_dir)


def plot_rf_evaluation(y_test, y_pred, y_prob, output_dir: str) -> None:
    _plot_4panel_eval(y_test, y_pred, y_prob,
                      "Random Forest Evaluation", "17_rf_eval", output_dir)


def plot_rf_importances(rf, feature_names: list, output_dir: str) -> None:
    _plot_importances(rf.feature_importances_, feature_names, top_n=20,
                      title="Random Forest — Feature Importances (Gini)",
                      filename="18_rf_importance", output_dir=output_dir, color="steelblue")


def plot_xgb_evaluation(y_test, y_pred, y_prob, output_dir: str) -> None:
    _plot_4panel_eval(y_test, y_pred, y_prob,
                      "XGBoost Evaluation", "19_xgb_eval", output_dir)


def plot_xgb_importances(xgb, feature_names: list, output_dir: str) -> None:
    _plot_importances(xgb.feature_importances_, feature_names, top_n=20,
                      title="XGBoost — Feature Importances (Gain)",
                      filename="20_xgb_importance", output_dir=output_dir, color="darkorange")


def plot_lgbm_evaluation(y_test, y_pred, y_prob, output_dir: str) -> None:
    _plot_4panel_eval(y_test, y_pred, y_prob,
                      "LightGBM Evaluation", "21_lgbm_eval", output_dir)


def plot_lgbm_importances(lgbm, feature_names: list, output_dir: str) -> None:
    _plot_importances(lgbm.feature_importances_, feature_names, top_n=20,
                      title="LightGBM — Feature Importances (Gain)",
                      filename="22_lgbm_importance", output_dir=output_dir, color="seagreen")


# ---------------------------------------------------------------------------
# Vintage analysis, model selection, repayment formula
# ---------------------------------------------------------------------------

def plot_vintage_analysis(models_data: dict, y_test, occ_tier_test, output_dir: str) -> None:
    cohorts = ["Substandard", "Special Mention", "Pass"]
    model_names = list(models_data.keys())
    y_test_arr = np.asarray(y_test)
    tier_arr = np.asarray(occ_tier_test, dtype=str)

    recall_grid = np.zeros((len(model_names), len(cohorts)))
    auprg_grid = np.zeros((len(model_names), len(cohorts)))

    for i, name in enumerate(model_names):
        y_prob = models_data[name]["prob"]
        t = models_data[name]["threshold"]
        for j, cohort in enumerate(cohorts):
            mask = tier_arr == cohort
            if mask.sum() < 5:
                continue
            y_c = y_test_arr[mask]
            p_c = y_prob[mask]
            if y_c.sum() == 0:
                continue
            y_pred_c = (p_c >= t).astype(int) if t else np.zeros_like(p_c, dtype=int)
            recall_grid[i, j] = recall_score(y_c, y_pred_c, pos_label=1, zero_division=0)
            _, _, auprg = compute_prg_curve(y_c, p_c)
            auprg_grid[i, j] = auprg

    x = np.arange(len(cohorts))
    width = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, (name, color) in enumerate(zip(model_names, _MODEL_COLORS)):
        axes[0].bar(x + i * width, recall_grid[i], width, label=name,
                    color=color, edgecolor="white")
        axes[1].bar(x + i * width, auprg_grid[i], width, label=name,
                    color=color, edgecolor="white")

    for ax, title, ylabel, target in [
        (axes[0], "Recall by OCC FICO Tier Cohort", "Recall (default class)", 0.88),
        (axes[1], "AUPRG by OCC FICO Tier Cohort", "AUPRG", None),
    ]:
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(cohorts)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        if target:
            ax.axhline(target, color="crimson", linestyle="--", lw=1.2,
                       label=f"Target {target}")

    fig.suptitle("Vintage Analysis — OCC FICO Tier Cohort Performance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig("23_vintage_analysis", output_dir)


def plot_model_selection(models_data: dict, y_test, output_dir: str) -> None:
    model_names = list(models_data.keys())
    y_test_arr = np.asarray(y_test)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for name, color in zip(model_names, _MODEL_COLORS):
        pg = models_data[name]["pg"]
        rg = models_data[name]["rg"]
        auprg = models_data[name]["auprg"]
        order = np.argsort(rg)
        axes[0].plot(rg[order], pg[order], color=color, lw=2,
                     label=f"{name} (AUPRG={auprg:.3f})")
    axes[0].set_xlabel("Recall Gain")
    axes[0].set_ylabel("Precision Gain")
    axes[0].set_title("PRG Curves")
    axes[0].legend(fontsize=8)

    for name, color in zip(model_names, _MODEL_COLORS):
        y_prob = models_data[name]["prob"]
        fpr, tpr, _ = roc_curve(y_test_arr, y_prob)
        auc = models_data[name]["roc_auc"]
        axes[1].plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curves")
    axes[1].legend(fontsize=8)

    recalls = []
    for name in model_names:
        y_prob = models_data[name]["prob"]
        t = models_data[name]["threshold"]
        if t is None:
            recalls.append(0.0)
            continue
        y_pred = (y_prob >= t).astype(int)
        recalls.append(recall_score(y_test_arr, y_pred, pos_label=1, zero_division=0))

    bars = axes[2].bar(model_names, recalls, color=_MODEL_COLORS, edgecolor="white")
    axes[2].axhline(0.88, color="crimson", linestyle="--", lw=1.5, label="Target recall 0.88")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Recall (default class)")
    axes[2].set_title("Recall at Operating Threshold")
    axes[2].tick_params(axis="x", rotation=15)
    axes[2].legend(fontsize=9)
    for bar, r in zip(bars, recalls):
        axes[2].text(bar.get_x() + bar.get_width() / 2, r + 0.01,
                     f"{r:.3f}", ha="center", fontsize=9)

    fig.suptitle("Model Selection Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _savefig("24_model_selection", output_dir)


def plot_repayment_formula(y_test, y_prob_lr: np.ndarray, threshold: float, output_dir: str) -> None:
    y_test_arr = np.asarray(y_test)
    pi = y_test_arr.mean()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Score distributions
    for val, label, color in [(0, "Repaid", "steelblue"), (1, "Defaulted", "tomato")]:
        mask = y_test_arr == val
        pd.Series(y_prob_lr[mask]).plot.kde(ax=axes[0], color=color, lw=2, label=label)
    if threshold:
        axes[0].axvline(threshold, color="crimson", linestyle="--", lw=1.5,
                        label=f"T = {threshold:.4f}")
    axes[0].set_xlabel("P(cannot pay back)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Score Distribution by Class")
    axes[0].legend()

    # OCC tier distribution
    tier_labels = []
    for p in y_prob_lr:
        for i, boundary in enumerate(OCC_PROB_BINS[1:]):
            if p < boundary:
                tier_labels.append(OCC_PROB_LABELS[i])
                break
        else:
            tier_labels.append(OCC_PROB_LABELS[-1])
    tier_series = pd.Series(tier_labels)
    tier_counts = tier_series.value_counts().reindex(OCC_PROB_LABELS, fill_value=0)
    bars = axes[1].bar(OCC_PROB_LABELS, tier_counts.values,
                       color=OCC_PROB_COLORS, edgecolor="white")
    for bar, v in zip(bars, tier_counts.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                     f"{v}", ha="center", fontsize=9)
    axes[1].set_ylabel("Test Borrowers")
    axes[1].set_title("OCC Tier Distribution (Test Set)")
    axes[1].tick_params(axis="x", rotation=20)

    # Threshold sweep
    prec_all, rec_all, thresh = precision_recall_curve(y_test_arr, y_prob_lr, pos_label=1)
    prec_at_thresh = prec_all[:-1]
    rec_at_thresh = rec_all[:-1]
    axes[2].plot(thresh, rec_at_thresh, color="steelblue", lw=2, label="Recall")
    axes[2].plot(thresh, prec_at_thresh, color="darkorange", lw=2, label="Precision")
    if threshold:
        axes[2].axvline(threshold, color="crimson", linestyle="--", lw=1.5,
                        label=f"T = {threshold:.4f}")
    axes[2].set_xlabel("Threshold")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Precision / Recall vs Threshold")
    axes[2].legend()
    axes[2].set_xlim(0, 1)

    fig.suptitle("Repayment Likelihood Formula", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig("25_repayment_formula", output_dir)
