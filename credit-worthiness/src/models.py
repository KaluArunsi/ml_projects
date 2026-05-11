import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing import Tuple

from .config import (
    LR_C, LR_SOLVER, LR_MAX_ITER,
    RF_PARAMS, XGB_PARAMS, LGBM_PARAMS,
    OCC_PROB_BINS, OCC_PROB_LABELS,
    TARGET_RECALL, RANDOM_STATE,
)


def compute_prg_curve(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    pi = np.asarray(y_true).mean()
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    with np.errstate(divide="ignore", invalid="ignore"):
        pg = np.where(precision > 0, (precision - pi) / ((1 - pi) * precision), 0)
        rg = np.where(recall > 0, (recall - pi) / ((1 - pi) * recall), 0)
    pg = np.clip(pg, 0, 1)
    rg = np.clip(rg, 0, 1)
    order = np.argsort(rg)
    auprg = float(np.trapezoid(pg[order], rg[order]))
    return pg, rg, auprg


def find_operating_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target: float = TARGET_RECALL,
) -> float:
    _, rec, thresh = precision_recall_curve(y_true, y_prob, pos_label=1)
    valid = thresh[rec[:-1] >= target]
    return float(valid[-1]) if len(valid) else None


def assign_occ_tier(prob: float) -> str:
    for i, boundary in enumerate(OCC_PROB_BINS[1:]):
        if prob < boundary:
            return OCC_PROB_LABELS[i]
    return OCC_PROB_LABELS[-1]


def train_logistic_regression(
    X_res: np.ndarray,
    y_res: np.ndarray,
    C: float = LR_C,
    solver: str = LR_SOLVER,
    max_iter: int = LR_MAX_ITER,
    random_state: int = RANDOM_STATE,
) -> LogisticRegression:
    lr = LogisticRegression(
        C=C,
        solver=solver,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=random_state,
    )
    lr.fit(X_res, y_res)
    return lr


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        **RF_PARAMS,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: float,
    random_state: int = RANDOM_STATE,
) -> XGBClassifier:
    xgb = XGBClassifier(
        **XGB_PARAMS,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    return xgb


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> LGBMClassifier:
    lgbm = LGBMClassifier(
        **LGBM_PARAMS,
        is_unbalance=True,
        objective="binary",
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm.fit(X_train, y_train)
    return lgbm
