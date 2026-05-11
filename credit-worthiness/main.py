#!/usr/bin/env python3
"""
Credit Worthiness Assessment
Entry point: runs the selected model (Logistic Regression) end-to-end.
"""
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, recall_score, roc_auc_score

from src import (
    load_dataset,
    engineer_features, add_lof_flag, add_mahal_flag,
    iqr_cap, add_kmeans_cluster, encode_and_drop, split_and_scale, apply_smoteenn,
    train_logistic_regression,
    compute_prg_curve, find_operating_threshold, assign_occ_tier,
    OCC_PROB_LABELS, OCC_PROB_ACTIONS,
    plot_class_distribution, plot_univariate_numeric, plot_categorical_analysis,
    plot_correlation_heatmap, plot_boxplots_by_target, plot_violin_capacity,
    plot_purpose_stacked, plot_pairplot,
    plot_lof_outliers, plot_mahalanobis,
    plot_cluster_selection, plot_cluster_pca, plot_cluster_profiles,
    plot_smoteenn,
    plot_lr_evaluation, plot_lr_coefficients,
    plot_vintage_analysis, plot_repayment_formula,
)

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Credit worthiness loan default assessment")
    p.add_argument("--no-images", action="store_true",
                   help="Print results only, skip image generation")
    p.add_argument("--output-dir", default="output", metavar="DIR",
                   help="Directory for saved images (default: output/)")
    return p.parse_args()


def _save_eda_plots(df, lof_scores, d2, cutoff, k_range, inertias, sil_scores, out):
    plot_class_distribution(df["cannot_pay_back"], out)
    plot_univariate_numeric(df, out)
    plot_categorical_analysis(df, out)
    plot_correlation_heatmap(df, out)
    plot_boxplots_by_target(df, out)
    plot_violin_capacity(df, out)
    plot_purpose_stacked(df, out)
    plot_pairplot(df, out)
    plot_lof_outliers(df, lof_scores, out)
    plot_mahalanobis(df, d2, cutoff, out)
    plot_cluster_selection(k_range, inertias, sil_scores, out)
    plot_cluster_pca(df, out)
    plot_cluster_profiles(df, out)


def _print_vintage(cohorts, tier_arr, y_test_arr, y_prob, t):
    logger.info("      Recall by OCC FICO tier cohort (T = %.4f):", t if t else 0)
    for cohort in cohorts:
        mask = tier_arr == cohort
        if mask.sum() == 0 or y_test_arr[mask].sum() == 0:
            continue
        y_pred_c = (y_prob[mask] >= t).astype(int) if t else np.zeros(mask.sum(), dtype=int)
        r = recall_score(y_test_arr[mask], y_pred_c, pos_label=1, zero_division=0)
        _, _, auprg_c = compute_prg_curve(y_test_arr[mask], y_prob[mask])
        logger.info("      %-18s  n=%4d  defaults=%3d  recall=%.3f  AUPRG=%.3f",
                    cohort, mask.sum(), int(y_test_arr[mask].sum()), r, auprg_c)


def _print_borrowers(y_prob):
    logger.info("Sample borrower assessments:")
    logger.info("  %-8s  %-10s  %-10s  %-18s  %s",
                "Idx", "P(default)", "P(repay)", "OCC Tier", "Action")
    logger.info("  " + "-" * 78)
    sample_idx = np.random.default_rng(42).choice(len(y_prob), size=min(8, len(y_prob)), replace=False)
    for i in sample_idx:
        p = float(y_prob[i])
        tier = assign_occ_tier(p)
        action = OCC_PROB_ACTIONS[OCC_PROB_LABELS.index(tier)]
        logger.info("  %-8d  %-10.4f  %-10.4f  %-18s  %s", i, p, 1 - p, tier, action)


def main():
    args = parse_args()
    save = not args.no_images
    out = args.output_dir

    logger.info("=" * 60)
    logger.info("Credit Worthiness Assessment")
    logger.info("=" * 60)

    # 1. Load
    logger.info("[1/7] Loading data...")
    df_raw = load_dataset()
    logger.info("      %s loans, %d features", f"{len(df_raw):,}", df_raw.shape[1])

    # 2. Feature engineering
    logger.info("[2/7] Engineering features...")
    df = engineer_features(df_raw)
    class_counts = df["cannot_pay_back"].value_counts().sort_index()
    logger.info("      Repaid: %s (%.1f%%)  |  Defaulted: %s (%.1f%%)",
                f"{class_counts[0]:,}", class_counts[0] / len(df) * 100,
                f"{class_counts[1]:,}", class_counts[1] / len(df) * 100)
    logger.info("      DSCR proxy — mean %.2f  |  median %.2f",
                df["dscr_proxy"].mean(), df["dscr_proxy"].median())
    logger.info("      DTI flag (>43%%): %d  |  High util (>80%%): %d  |  "
                "Delinq: %d  |  Pub rec: %d",
                df["dti_flag"].sum(), df["high_utilization_flag"].sum(),
                df["delinquency_flag"].sum(), df["public_record_flag"].sum())

    # 3. Outlier detection
    logger.info("[3/7] Outlier detection...")
    df, lof_scores = add_lof_flag(df)
    df, d2, cutoff = add_mahal_flag(df)
    logger.info("      LOF flagged: %d (%.1f%%)  |  Mahal flagged: %d (%.1f%%) — all retained",
                df["lof_flag"].sum(), df["lof_flag"].mean() * 100,
                df["mahal_flag"].sum(), df["mahal_flag"].mean() * 100)

    # 4. IQR capping + clustering
    logger.info("[4/7] IQR capping and clustering...")
    df = iqr_cap(df)
    df, k_range, inertias, sil_scores = add_kmeans_cluster(df)
    cluster_dr = df.groupby("cluster_label")["cannot_pay_back"].mean()
    cluster_n = df["cluster_label"].value_counts().sort_index()
    for c in sorted(df["cluster_label"].unique()):
        logger.info("      Cluster %d — n=%d  default rate=%.1f%%",
                    c, cluster_n[c], cluster_dr[c] * 100)

    if save:
        _save_eda_plots(df, lof_scores, d2, cutoff, k_range, inertias, sil_scores, out)

    occ_tier_all = df["occ_tier"].copy()

    # 5. Encode, drop, split, scale
    logger.info("[5/7] Preprocessing and splitting...")
    df_model = encode_and_drop(df)
    X = df_model.drop(columns=["cannot_pay_back"])
    y = df_model["cannot_pay_back"]
    feature_names = list(X.columns)
    x_train_sc, x_test_sc, y_train, y_test, _, _, test_idx = split_and_scale(X, y)
    occ_tier_test = occ_tier_all.loc[test_idx]
    logger.info("      Train: %s  |  Test: %s  |  Features: %d",
                f"{len(y_train):,}", f"{len(y_test):,}", len(feature_names))

    # 6. SMOTE-ENN + train
    logger.info("[6/7] SMOTE-ENN resampling and training...")
    x_res, y_res = apply_smoteenn(x_train_sc, y_train)
    res_counts = pd.Series(y_res).value_counts().sort_index()
    logger.info("      Before — Repaid: %d  Defaulted: %d",
                int((y_train == 0).sum()), int((y_train == 1).sum()))
    logger.info("      After  — Repaid: %d  Defaulted: %d",
                res_counts.get(0, 0), res_counts.get(1, 0))

    if save:
        plot_smoteenn(y_train.values, y_res, out)

    lr = train_logistic_regression(x_res, y_res)
    y_prob = lr.predict_proba(x_test_sc)[:, 1]
    t = find_operating_threshold(y_test.values, y_prob)
    y_pred = (y_prob >= t).astype(int) if t else np.zeros(len(y_test), dtype=int)
    auc = roc_auc_score(y_test.values, y_prob)
    pg, rg, auprg = compute_prg_curve(y_test.values, y_prob)

    logger.info("      ROC-AUC %.4f  |  AUPRG %.4f  |  T = %.4f", auc, auprg, t if t else 0)
    logger.info("Classification report:\n%s",
                classification_report(y_test.values, y_pred, target_names=["Repaid", "Defaulted"]))

    if save:
        plot_lr_evaluation(y_test.values, y_pred, y_prob, out)
        plot_lr_coefficients(lr, feature_names, out)

    # 7. Vintage analysis + repayment formula
    logger.info("[7/7] Vintage analysis and repayment formula...")
    cohorts = ["Substandard", "Special Mention", "Pass"]
    tier_arr = np.asarray(occ_tier_test, dtype=str)
    y_test_arr = y_test.values

    _print_vintage(cohorts, tier_arr, y_test_arr, y_prob, t)
    _print_borrowers(y_prob)

    if save:
        lr_data = {"prob": y_prob, "threshold": t, "pg": pg, "rg": rg, "auprg": auprg, "roc_auc": auc}
        plot_vintage_analysis({"Logistic Regression": lr_data}, y_test_arr, occ_tier_test, out)
        plot_repayment_formula(y_test_arr, y_prob, t, out)

    logger.info("=" * 60)
    logger.info("Done.%s", f" Images saved to {out!r}." if save else "")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
