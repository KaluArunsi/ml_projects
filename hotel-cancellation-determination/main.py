#!/usr/bin/env python3
"""
Hotel Booking Cancellation Risk Assessment
Entry point: runs the full two-model pipeline.
"""
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, roc_auc_score

from src import (
    load_dataset,
    clean, cap_outliers, drop_leakage, encode, build_feature_matrix,
    remove_univariate, remove_multivariate,
    split_and_scale, train_lr, create_severity_labels, train_xgboost,
    SEVERITY_LABELS, SEVERITY_ACTIONS,
    plot_class_distribution, plot_feature_distributions,
    plot_lr_evaluation, plot_lr_learning_curve,
    plot_severity_distribution,
    plot_xgb_evaluation, plot_xgb_confidence,
    plot_xgb_pr_curves, plot_xgb_learning_curve,
)

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Hotel booking cancellation risk assessment")
    p.add_argument("--no-images", action="store_true",
                   help="Print results only, skip image generation")
    p.add_argument("--output-dir", default="output", metavar="DIR",
                   help="Directory for saved images (default: output/)")
    return p.parse_args()


def main():
    args = parse_args()
    save = not args.no_images
    out  = args.output_dir

    logger.info("=" * 50)
    logger.info("Hotel Booking Cancellation Risk Assessment")
    logger.info("=" * 50)

    # 1. Load
    logger.info("[1/7] Loading data...")
    df = load_dataset()
    logger.info("      %s bookings, %d features", f"{len(df):,}", df.shape[1])
    if save:
        plot_class_distribution(df["is_canceled"], out)
        plot_feature_distributions(df, out)

    # 2. Preprocess
    logger.info("[2/7] Preprocessing...")
    df_clean   = clean(df)
    df_capped  = cap_outliers(df_clean)
    df_capped  = drop_leakage(df_capped)
    df_encoded = encode(df_capped)
    df_final   = build_feature_matrix(df_encoded, df_capped)
    logger.info("      After preprocessing: %d features", df_final.shape[1])

    # 3. Remove outliers
    logger.info("[3/7] Removing outliers...")
    df_uni = remove_univariate(df_final)
    df_mv  = remove_multivariate(df_uni)
    removed = len(df_final) - len(df_mv)
    logger.info("      Removed %s rows (%.1f%%)", f"{removed:,}", removed / len(df_final) * 100)
    logger.info("      Clean dataset: %s rows", f"{len(df_mv):,}")

    # 4. Split and scale
    logger.info("[4/7] Splitting and scaling...")
    X = df_mv.drop(columns=["is_canceled"])
    y = df_mv["is_canceled"]
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    logger.info("      Train: %s | Test: %s", f"{len(y_train):,}", f"{len(y_test):,}")

    # 5. Logistic Regression
    logger.info("[5/7] Training Logistic Regression...")
    lr        = train_lr(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob_lr)
    f1  = f1_score(y_test, y_pred_lr)
    logger.info("      ROC-AUC        %.4f", auc)
    logger.info("      F1 (cancelled) %.4f", f1)
    logger.info("Classification report:\n%s",
                classification_report(y_test, y_pred_lr, target_names=["Not Cancelled", "Cancelled"]))
    if save:
        plot_lr_evaluation(y_test, y_pred_lr, y_prob_lr, out)
        plot_lr_learning_curve(X_train, y_train, out)

    # 6. Severity labels
    logger.info("[6/7] Generating severity labels...")
    y_train_sev, y_test_sev = create_severity_labels(X_train, y_train, y_prob_lr)
    tier_counts = pd.Series(y_test_sev).value_counts().sort_index()
    tier_table  = "\n".join(
        f"  Tier {t}  {SEVERITY_LABELS[t]:<22} count={tier_counts.get(t, 0):>5}  {SEVERITY_ACTIONS[t]}"
        for t in range(4)
    )
    logger.info("Severity tier breakdown:\n%s", tier_table)
    if save:
        y_prob_full = lr.predict_proba(scaler.transform(X))[:, 1]
        plot_severity_distribution(y_prob_full, out)

    # 7. XGBoost
    logger.info("[7/7] Training XGBoost severity classifier...")
    xgb        = train_xgboost(X_train, y_train_sev)
    y_pred_sev = xgb.predict(X_test)
    y_prob_sev = xgb.predict_proba(X_test)

    macro_auc = roc_auc_score(
        np.eye(4, dtype=int)[np.asarray(y_test_sev, dtype=int)],
        y_prob_sev, multi_class="ovr", average="macro",
    )
    macro_f1 = f1_score(y_test_sev, y_pred_sev, average="macro")
    logger.info("      Macro ROC-AUC  %.4f", macro_auc)
    logger.info("      Macro F1       %.4f", macro_f1)
    logger.info("XGBoost classification report:\n%s",
                classification_report(
                    y_test_sev, y_pred_sev,
                    target_names=["T0-Very Unlikely", "T1-Unlikely", "T2-Likely", "T3-Most Likely"],
                ))
    if save:
        plot_xgb_evaluation(y_test_sev, y_pred_sev, list(X.columns), xgb.feature_importances_, out)
        plot_xgb_confidence(y_test_sev, y_prob_sev, out)
        plot_xgb_pr_curves(y_test_sev, y_prob_sev, out)
        plot_xgb_learning_curve(X_train, y_train_sev, out)

    logger.info("=" * 50)
    logger.info("Done.%s", f" Images saved to {out!r}." if save else "")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
