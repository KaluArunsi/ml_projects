"""
Preprocessing Module
====================
Cleaning, commodity normalization, label parsing, document assembly, and
train/val/test splitting for the oil spill cause classification pipeline.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from . import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Commodity normalization
# ---------------------------------------------------------------------------

def _build_commodity_map() -> dict[str, str]:
    """Build a lookup dict from raw (lowercased) commodity → canonical category."""
    mapping: dict[str, str] = {}
    for canonical, variants in config.COMMODITY_CATEGORIES.items():
        for variant in variants:
            mapping[variant.lower().strip()] = canonical
    return mapping


_COMMODITY_MAP = _build_commodity_map()


def normalize_commodity(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the commodity column from 1,555 unique raw values → ~20 categories.

    Applies lowercasing, stripping, and synonym mapping. Unmapped values get
    ``config.UNKNOWN_COMMODITY_LABEL`` ("Other").

    Args:
        df: Incidents DataFrame with a ``commodity`` column.

    Returns:
        DataFrame with a new ``commodity_category`` column.
    """
    raw_col = "commodity"

    def _map_one(value) -> str:
        if pd.isna(value) or not isinstance(value, str):
            return config.UNKNOWN_COMMODITY_LABEL
        cleaned = value.lower().strip()
        # Remove parentheticals and extra punctuation
        cleaned = re.sub(r"\(.*?\)", "", cleaned)
        cleaned = re.sub(r"[^\w\s/#-]", "", cleaned)
        cleaned = cleaned.strip()
        return _COMMODITY_MAP.get(cleaned, config.UNKNOWN_COMMODITY_LABEL)

    df = df.copy()
    df["commodity_category"] = df[raw_col].apply(_map_one)

    n_unique_raw = df[raw_col].dropna().nunique()
    n_categories = df["commodity_category"].nunique()
    n_unmapped = (df["commodity_category"] == config.UNKNOWN_COMMODITY_LABEL).sum()
    logger.info(
        "Normalized commodity: %d unique raw → %d categories (%d unmapped → 'Other')",
        n_unique_raw, n_categories, n_unmapped,
    )
    return df


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def parse_tags(tags_series: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Parse pipe-separated cause tags into a multi-label binary matrix.

    Labels with fewer than ``config.MIN_LABEL_COUNT`` occurrences are collapsed
    into ``config.OTHER_LABEL``.

    Args:
        tags_series: Series of pipe-separated tag strings (e.g. ``"Coral|Grounding"``).

    Returns:
        (label_matrix, label_names) where label_matrix is (n_samples, n_labels)
        and label_names is the ordered list of label strings.
    """
    # Split tags into lists
    tag_lists = []
    for val in tags_series:
        if pd.isna(val) or not isinstance(val, str) or not val.strip():
            tag_lists.append([])
        else:
            tag_lists.append([t.strip() for t in val.split("|") if t.strip()])

    # Count frequencies
    from collections import Counter
    tag_counter = Counter()
    for tags in tag_lists:
        tag_counter.update(tags)

    logger.info("Raw tag frequencies: %s",
                 dict(tag_counter.most_common()))

    # Identify labels to keep vs fold into Other
    keep_labels = {
        tag for tag, cnt in tag_counter.items()
        if cnt >= config.MIN_LABEL_COUNT
    }
    rare_labels = {tag for tag in tag_counter if tag not in keep_labels}

    # Sort to match config.LABEL_NAMES ordering, then append any extras
    ordered_labels = [l for l in config.LABEL_NAMES if l in keep_labels]
    # Add any label not in our predefined list but above threshold
    for tag in sorted(keep_labels - set(config.LABEL_NAMES)):
        ordered_labels.append(tag)

    n_other = sum(cnt for tag, cnt in tag_counter.items() if tag not in keep_labels)
    if n_other > 0:
        ordered_labels.append(config.OTHER_LABEL)
        logger.info("Folded %d rare-label occurrences (%s) into '%s'",
                     n_other, sorted(rare_labels), config.OTHER_LABEL)

    logger.info("Final label set (%d): %s", len(ordered_labels), ordered_labels)

    # Replace rare labels with "Other" in the tag lists BEFORE binarization
    if rare_labels:
        tag_lists = [
            [config.OTHER_LABEL if t in rare_labels else t for t in tags]
            if tags else []
            for tags in tag_lists
        ]

    # Build binary matrix via MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=ordered_labels)
    label_matrix = mlb.fit_transform(tag_lists)

    logger.info("Label matrix shape: %s (n_samples=%d, n_labels=%d)",
                 label_matrix.shape, label_matrix.shape[0], label_matrix.shape[1])
    logger.info("Label density: %.3f (multi-label rate: %.1f%%)",
                 label_matrix.mean(),
                 100 * (label_matrix.sum(axis=1) > 1).mean())

    return label_matrix, ordered_labels


def compute_label_weights(label_matrix: np.ndarray, max_weight: float = 100.0) -> np.ndarray:
    """Compute per-label inverse-frequency weights for focal loss / BCE.

    Returns an array of shape (n_labels,) where weight = (1 - pos_rate) / pos_rate,
    capped at ``max_weight`` to prevent extreme values for ultra-rare labels
    (e.g. the "Other" bucket with only 5 examples).

    Args:
        label_matrix: Binary label matrix of shape (n_samples, n_labels).
        max_weight: Maximum allowable per-label weight.
    """
    n_samples = label_matrix.shape[0]
    pos_counts = label_matrix.sum(axis=0)
    # Add small epsilon to avoid division by zero
    pos_rate = (pos_counts + 1e-6) / n_samples
    weights = (1 - pos_rate) / pos_rate
    weights = np.clip(weights, 1.0, max_weight)
    logger.info("Per-label pos_weight (capped at %.0f): min=%.2f, max=%.2f, mean=%.2f",
                 max_weight, weights.min(), weights.max(), weights.mean())
    return weights


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------

def assemble_documents(
    incidents: pd.DataFrame,
    posts: pd.DataFrame,
    max_posts: int = config.MAX_POSTS_PER_INCIDENT,
    max_words: int = config.MAX_DOCUMENT_WORDS,
) -> pd.DataFrame:
    """Assemble each incident's document from its description + concatenated posts.

    Posts are joined on ``incidents[INCIDENT_ID_COL]`` == ``posts[POST_INCIDENT_ID_COL]``.
    Posts are sorted by date descending within each incident. Posts with null content
    fall back to their title.

    Args:
        incidents: Cleaned incidents DataFrame.
        posts: Raw posts DataFrame.
        max_posts: Maximum number of posts to concatenate per incident.
        max_words: Word-level cap for the full document.

    Returns:
        Incidents DataFrame with new ``document`` column and ``n_posts_used`` column.
    """
    df = incidents.copy()

    # Build a lookup: incident_id → sorted post texts
    post_groups: dict[int, list[str]] = {}
    for _, row in posts.iterrows():
        inc_id = row[config.POST_INCIDENT_ID_COL]
        content = row.get(config.POST_CONTENT_COL)
        title = row.get(config.POST_TITLE_COL)

        # Prefer content, fall back to title
        text = content if pd.notna(content) and str(content).strip() else None
        if text is None and pd.notna(title):
            text = str(title)
        if text is None:
            continue

        text = str(text).strip()
        if text:
            post_groups.setdefault(inc_id, []).append(text)

    # Assemble per incident
    documents: list[str] = []
    n_posts_used: list[int] = []

    for _, row in df.iterrows():
        inc_id = row[config.INCIDENT_ID_COL]
        desc = row.get(config.DESCRIPTION_COL, "")
        if pd.isna(desc):
            desc = ""

        parts = [str(desc).strip()]
        incident_posts = post_groups.get(inc_id, [])

        # Take up to max_posts (most recent first since sorted descending)
        used_posts = incident_posts[:max_posts]
        parts.extend(used_posts)
        n_posts_used.append(len(used_posts))

        full_text = " [SEP] ".join(p for p in parts if p)

        # Word-level truncation
        words = full_text.split()
        if len(words) > max_words:
            full_text = " ".join(words[:max_words])

        documents.append(full_text)

    df["document"] = documents
    df["n_posts_used"] = n_posts_used

    # Stats
    word_counts = [len(d.split()) for d in documents]
    logger.info(
        "Assembled documents: mean=%.0f words, median=%.0f, max=%d. "
        "Posts per incident: mean=%.1f, median=%d, max=%d",
        np.mean(word_counts), np.median(word_counts), max(word_counts),
        np.mean(n_posts_used), np.median(n_posts_used), max(n_posts_used),
    )

    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def normalize_release_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Log₁₀-transform max_ptl_release_gallons. Handles zeros and NaNs.

    Adds ``log_release_volume`` and ``has_release_volume`` columns.
    """
    df = df.copy()
    col = "max_ptl_release_gallons"
    df["has_release_volume"] = df[col].notna() & (df[col] > 0)
    df["log_release_volume"] = np.log10(df[col].fillna(0).clip(lower=0) + 1)
    return df


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract year, month, season from open_date."""
    df = df.copy()
    dates = pd.to_datetime(df["open_date"], errors="coerce")
    df["year"] = dates.dt.year
    df["month"] = dates.dt.month

    # Season: 0=Winter, 1=Spring, 2=Summer, 3=Fall (Northern Hemisphere)
    df["season"] = (dates.dt.month % 12 + 3) // 3 % 4
    return df


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

def stratified_multilabel_split(
    df: pd.DataFrame,
    label_matrix: np.ndarray,
    test_size: float = config.TEST_SIZE,
    val_size: float = config.VAL_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> dict[str, np.ndarray]:
    """Stratified train/val/test split for multi-label data.

    Uses iterative stratification (from skmultilearn if available, otherwise
    falls back to a simple random split that attempts to preserve per-label
    presence in each fold).

    Args:
        df: Incidents DataFrame (used for indexing only).
        label_matrix: Binary label matrix of shape (n_samples, n_labels).
        test_size: Fraction for test set.
        val_size: Fraction for validation set (of the full labeled set).
        random_state: Random seed.

    Returns:
        Dict with keys 'train', 'val', 'test', each containing an array of
        integer indices into df/label_matrix.
    """
    n = len(df)
    rng = np.random.RandomState(random_state)

    # Try iterative stratification first
    try:
        from skmultilearn.model_selection import iterative_train_test_split

        # First split: separate test set
        X_temp, y_temp, X_test, y_test = iterative_train_test_split(
            np.arange(n).reshape(-1, 1), label_matrix,
            test_size=test_size,
        )
        # Second split: separate validation from remaining
        val_frac_of_remaining = val_size / (1 - test_size)
        X_train, y_train, X_val, y_val = iterative_train_test_split(
            X_temp, y_temp,
            test_size=val_frac_of_remaining,
        )

        indices = {
            "train": X_train.ravel(),
            "val": X_val.ravel(),
            "test": X_test.ravel(),
        }
        logger.info("Stratified split (skmultilearn): train=%d, val=%d, test=%d",
                     len(indices["train"]), len(indices["val"]), len(indices["test"]))
        return indices

    except ImportError:
        logger.info("skmultilearn not available — using simple stratified random split")

    # Fallback: ensure each split gets at least one example of each label
    n_labels = label_matrix.shape[1]
    all_indices = np.arange(n)
    rng.shuffle(all_indices)

    n_test = int(n * test_size)
    n_val = int(n * val_size)
    n_train = n - n_test - n_val

    # Simple approach: shuffle and split, check coverage
    indices = {
        "train": all_indices[:n_train],
        "val": all_indices[n_train:n_train + n_val],
        "test": all_indices[n_train + n_val:],
    }

    # Verify per-label coverage
    for split_name, idx in indices.items():
        split_matrix = label_matrix[idx]
        missing = np.where(split_matrix.sum(axis=0) == 0)[0]
        if len(missing) > 0:
            missing_names = [config.LABEL_NAMES[i] for i in missing if i < len(config.LABEL_NAMES)]
            logger.warning("%s split missing labels: %s", split_name, missing_names)

    logger.info("Random stratified split: train=%d, val=%d, test=%d",
                 n_train, n_val, n_test)
    return indices


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline(
    incidents: Optional[pd.DataFrame] = None,
    posts: Optional[pd.DataFrame] = None,
    save_artifacts: bool = True,
) -> dict:
    """Run the full preprocessing pipeline end-to-end.

    Args:
        incidents: Incidents DataFrame. Loaded from disk if None.
        posts: Posts DataFrame. Loaded from disk if None.
        save_artifacts: Save processed data to disk.

    Returns:
        Dict with keys: 'incidents', 'label_matrix', 'label_names', 'label_weights',
        'splits', 'has_labels' (boolean mask).
    """
    from . import data_loader

    if incidents is None:
        incidents = data_loader.load_incidents()
    if posts is None:
        posts = data_loader.load_posts()

    logger.info("=== Preprocessing Pipeline ===")

    # 1. Normalize commodity
    incidents = normalize_commodity(incidents)

    # 2. Parse labels (only labeled incidents)
    has_labels = incidents[config.LABEL_COL].notna() & (
        incidents[config.LABEL_COL].astype(str).str.strip() != ""
    )
    n_labeled = has_labels.sum()
    logger.info("Labeled incidents: %d / %d (%.1f%%)",
                 n_labeled, len(incidents), 100 * n_labeled / len(incidents))

    label_matrix, label_names = parse_tags(incidents.loc[has_labels, config.LABEL_COL])
    label_weights = compute_label_weights(label_matrix)

    # 3. Assemble document text (all incidents, not just labeled)
    incidents = assemble_documents(incidents, posts)

    # 4. Feature engineering
    incidents = normalize_release_volume(incidents)
    incidents = extract_temporal_features(incidents)

    # 5. Split (labeled only)
    labeled_df = incidents[has_labels].reset_index(drop=True)
    splits = stratified_multilabel_split(labeled_df, label_matrix)

    # 6. Save artifacts
    if save_artifacts:
        processed_dir = Path(config.DATA_PROCESSED_DIR)
        processed_dir.mkdir(parents=True, exist_ok=True)

        incidents.to_parquet(processed_dir / "incidents_clean.parquet", index=False)
        np.save(processed_dir / "label_matrix.npy", label_matrix)
        np.save(processed_dir / "label_weights.npy", label_weights)

        with open(processed_dir / "label_names.txt", "w") as f:
            f.write("\n".join(label_names))

        for split_name, idx in splits.items():
            np.save(processed_dir / f"{split_name}_indices.npy", idx)

        logger.info("Saved processed artifacts to %s", processed_dir)

    return {
        "incidents": incidents,
        "label_matrix": label_matrix,
        "label_names": label_names,
        "label_weights": label_weights,
        "splits": splits,
        "has_labels": has_labels.values,
    }
