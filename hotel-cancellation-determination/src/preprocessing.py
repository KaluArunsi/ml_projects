import numpy as np
import pandas as pd
from typing import List

from .config import IQR_COLS, LEAKAGE_COLS


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with appropriate defaults."""
    df = df.copy()
    df["country"]  = df["country"].replace(np.nan, "Unknown")
    df["agent"]    = df["agent"].replace(np.nan, "no_agent")
    df["company"]  = df["company"].replace(np.nan, "no_company")
    df["children"] = df["children"].replace(np.nan, 0)
    return df


def cap_outliers(df: pd.DataFrame, cols: List[str] = IQR_COLS) -> pd.DataFrame:
    """Winsorise columns at 1.5x IQR (Tukey fencing). Preserves row count."""
    df = df.copy()
    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
    return df


def drop_leakage(df: pd.DataFrame, cols: List[str] = LEAKAGE_COLS) -> pd.DataFrame:
    """Drop columns that encode the future outcome (data leakage)."""
    return df.drop(columns=[c for c in cols if c in df.columns])


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Frequency-encode high-cardinality categoricals, one-hot the rest."""
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    high_card = [c for c in cat_cols if df[c].nunique() > 10]
    low_card  = [c for c in cat_cols if df[c].nunique() <= 10]

    for col in high_card:
        freq = df[col].value_counts(normalize=True)
        df[f"{col}_freq"] = df[col].map(freq)
        df.drop(col, axis=1, inplace=True)

    df = pd.get_dummies(df, columns=low_card, drop_first=True)
    return df


def build_feature_matrix(df_encoded: pd.DataFrame, df_capped: pd.DataFrame) -> pd.DataFrame:
    """Combine encoded features with any numeric columns not yet in the encoded set."""
    enc_num   = df_encoded.select_dtypes(include=[np.number])
    orig_num  = df_capped.select_dtypes(include=[np.number])
    extra     = list(set(orig_num.columns) - set(enc_num.columns))
    return pd.concat([enc_num, orig_num[extra]], axis=1).copy()
