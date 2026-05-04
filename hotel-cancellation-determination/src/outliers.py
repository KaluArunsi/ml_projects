import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2
from typing import List

from .config import ZSCORE_COLS, MAHAL_COLS


def _modified_zscore(series: pd.Series) -> pd.Series:
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return 0.6745 * (series - median).abs() / mad


def remove_univariate(
    df: pd.DataFrame,
    cols: List[str] = ZSCORE_COLS,
    z_threshold: float = 3.0,
    mz_threshold: float = 3.5,
) -> pd.DataFrame:
    """Remove rows flagged as outliers by BOTH Z-score AND Modified Z-score.

    Using the intersection (conservative) avoids over-pruning on large datasets.
    """
    z_df    = pd.DataFrame(np.abs(stats.zscore(df[cols])), columns=cols, index=df.index)
    mz_df   = pd.DataFrame({c: _modified_zscore(df[c]) for c in cols})
    z_mask  = (z_df  > z_threshold).any(axis=1)
    mz_mask = (mz_df > mz_threshold).any(axis=1)
    keep    = ~(z_mask & mz_mask)
    return df[keep].reset_index(drop=True)


def remove_multivariate(
    df: pd.DataFrame,
    cols: List[str] = MAHAL_COLS,
    p_threshold: float = 0.999,
) -> pd.DataFrame:
    """Remove rows beyond the chi-squared threshold of Mahalanobis distance.

    D^2 ~ chi2(k) under multivariate normality. Threshold: 99.9th percentile.
    
    You use Mahalanobis for non-linear cases (logistic reg). Euclidean is for linear cases (linear reg)
    """
    X       = df[cols].values.astype(float)
    mean    = X.mean(axis=0)
    VI      = np.linalg.inv(np.cov(X, rowvar=False))
    diff    = X - mean
    D2      = np.einsum("ij,jk,ik->i", diff, VI, diff)
    cutoff  = chi2.ppf(p_threshold, df=len(cols))
    return df[D2 <= cutoff].reset_index(drop=True)
