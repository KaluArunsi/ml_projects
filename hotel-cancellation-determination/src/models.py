import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from typing import Tuple

from .config import SEVERITY_BINS


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple:
    """Stratified train/test split with StandardScaler fit on train only."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_lr(
    X_train: np.ndarray,
    y_train: pd.Series,
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """Train and return a calibrated Logistic Regression classifier."""
    lr = LogisticRegression(max_iter=max_iter, random_state=random_state)
    lr.fit(X_train, y_train)
    return lr


def create_severity_labels(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    y_prob_test: np.ndarray,
    cv: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create severity tier labels (0-3) for train and test sets.

    Training labels are generated via out-of-fold cross-validation to avoid
    the LR model scoring rows it was trained on (biased labels).
    """
    lr_cv = LogisticRegression(max_iter=1000, random_state=random_state)
    proba_train_cv = cross_val_predict(
        lr_cv, X_train_scaled, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    y_train_sev = np.asarray(
        pd.cut(proba_train_cv, bins=SEVERITY_BINS, labels=[0, 1, 2, 3]).astype(int)
    )
    y_test_sev = np.asarray(
        pd.cut(y_prob_test, bins=SEVERITY_BINS, labels=[0, 1, 2, 3]).astype(int)
    )
    return y_train_sev, y_test_sev


def train_xgboost(
    X_train: np.ndarray,
    y_train_sev: np.ndarray,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> XGBClassifier:
    """Train XGBoost multiclass severity classifier with balanced class weights."""
    sample_weights = compute_sample_weight("balanced", y_train_sev)
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="mlogloss",
        verbosity=0,
    )
    xgb.fit(X_train, y_train_sev, sample_weight=sample_weights)
    return xgb
