import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from typing import List, Tuple

from .config import (
    FICO_BINS, FICO_LABELS,
    IQR_CAP_COLS, CLUSTER_FEATURES, COLS_TO_DROP,
    N_CLUSTERS, RANDOM_STATE,
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns={"not.fully.paid": "cannot_pay_back"})

    df["monthly_income"] = np.exp(df["log.annual.inc"]) / 12
    df["payment_to_income_ratio"] = df["installment"] / df["monthly_income"]
    df["dscr_proxy"] = df["monthly_income"] / df["installment"]
    df["debt_burden_annual"] = (df["installment"] * 12) / np.exp(df["log.annual.inc"])

    df["dti_flag"] = (df["dti"] > 43).astype(int)
    df["high_utilization_flag"] = (df["revol.util"] > 80).astype(int)
    df["delinquency_flag"] = (df["delinq.2yrs"] > 0).astype(int)
    df["public_record_flag"] = (df["pub.rec"] > 0).astype(int)

    df["occ_tier"] = pd.cut(
        df["fico"],
        bins=FICO_BINS,
        labels=FICO_LABELS,
        right=False,
    )
    return df


def iqr_cap(df: pd.DataFrame, cols: List[str] = IQR_CAP_COLS) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
    return df


def add_kmeans_cluster(
    df: pd.DataFrame,
    cluster_cols: List[str] = CLUSTER_FEATURES,
    n_clusters: int = N_CLUSTERS,
    k_range: range = range(2, 9),
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, list, list, list]:
    df = df.copy()
    X_clust = StandardScaler().fit_transform(df[cluster_cols].values.astype(float))

    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_clust)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_clust, labels))

    km_final = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df["cluster_label"] = km_final.fit_predict(X_clust)
    return df, list(k_range), inertias, sil_scores


def encode_and_drop(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = pd.get_dummies(df, columns=["purpose"], drop_first=True)
    drop_cols = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=drop_cols)
    return df


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Tuple:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    return X_train_sc, X_test_sc, y_train, y_test, scaler, X_train.index, X_test.index


def apply_smoteenn(
    X_train_sc: np.ndarray,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray]:
    smoteenn = SMOTEENN(random_state=random_state)
    X_res, y_res = smoteenn.fit_resample(X_train_sc, y_train)
    return X_res, y_res
