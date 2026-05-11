import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from typing import Tuple

from .config import OUTLIER_COLS


def add_lof_flag(
    df: pd.DataFrame,
    cols: list = OUTLIER_COLS,
    n_neighbors: int = 20,
    contamination: float = 0.05,
) -> Tuple[pd.DataFrame, np.ndarray]:
    df = df.copy()
    X = StandardScaler().fit_transform(df[cols].values.astype(float))
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    flags = lof.fit_predict(X)
    scores = -lof.negative_outlier_factor_
    df["lof_flag"] = (flags == -1).astype(int)
    return df, scores


def add_mahal_flag(
    df: pd.DataFrame,
    cols: list = OUTLIER_COLS,
    p_threshold: float = 0.999,
) -> Tuple[pd.DataFrame, np.ndarray, float]:
    df = df.copy()
    X = df[cols].values.astype(float)
    mean = X.mean(axis=0)
    VI = np.linalg.inv(np.cov(X, rowvar=False))
    diff = X - mean
    D2 = np.einsum("ij,jk,ik->i", diff, VI, diff)
    cutoff = chi2.ppf(p_threshold, df=len(cols))
    df["mahal_flag"] = (D2 > cutoff).astype(int)
    return df, D2, cutoff
