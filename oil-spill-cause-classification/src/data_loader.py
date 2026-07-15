"""
Data Loading Module
===================
Loads primary and external data sources for the oil spill cause
classification project. Each loader handles download-on-missing and
graceful fallback on failure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primary data sources
# ---------------------------------------------------------------------------

def load_incidents(path: Optional[str] = None) -> pd.DataFrame:
    """Load the primary incidents CSV.

    Args:
        path: Override path. Defaults to config.DATA_RAW_DIR / config.INCIDENTS_FILENAME.

    Returns:
        DataFrame with 4,473 incidents (id, open_date, name, location, lat, lon,
        threat, tags, commodity, measure_*, max_ptl_release_gallons, posts, description).
    """
    if path is None:
        path = str(Path(config.DATA_RAW_DIR) / config.INCIDENTS_FILENAME)

    logger.info("Loading incidents from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d incidents, %d columns", len(df), len(df.columns))
    return df


def load_posts(path: Optional[str] = None) -> pd.DataFrame:
    """Load the posts Excel file.

    Args:
        path: Override path. Defaults to config.DATA_RAW_DIR / config.POSTS_FILENAME.

    Returns:
        DataFrame with 25,733 posts (NPost id, post title, post date, post tags,
        post content, attachment availability, noaa id, post id, old_npost_id_2parts).
    """
    if path is None:
        path = str(Path(config.DATA_RAW_DIR) / config.POSTS_FILENAME)

    logger.info("Loading posts from %s", path)
    df = pd.read_excel(path)
    logger.info("Loaded %d posts, %d columns", len(df), len(df.columns))
    return df


def move_raw_data() -> None:
    """Move data files from the project-level data/ dir into data/raw/ if needed."""
    raw_dir = Path(config.DATA_RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)

    project_data = Path("data")
    for filename in [config.INCIDENTS_FILENAME, config.POSTS_FILENAME]:
        src = project_data / filename
        dst = raw_dir / filename
        if src.exists() and not dst.exists():
            logger.info("Moving %s → %s", src, dst)
            src.rename(dst)


# ---------------------------------------------------------------------------
# External datasets (optional — graceful fallback on failure)
# ---------------------------------------------------------------------------

def load_purdue_usmart(data_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Download and load the Purdue uSMART enhanced dataset from figshare.

    The Purdue dataset enriches the same NOAA IncidentNews base with
    NLP-extracted actual release volumes and cleaned labels.

    Args:
        data_dir: Cache directory. Defaults to config.DATA_EXTERNAL_DIR / "purdue_usmart".

    Returns:
        DataFrame or None if download fails or data is inaccessible.
    """
    if data_dir is None:
        data_dir = str(Path(config.DATA_EXTERNAL_DIR) / "purdue_usmart")

    logger.info("Attempting to load Purdue uSMART dataset (figshare %s) ...",
                config.EXTERNAL_DATASETS["purdue_usmart"]["figshare_id"])
    try:
        import requests
        from io import StringIO

        cache_dir = Path(data_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Check for cached version first
        cached_files = list(cache_dir.glob("*.csv"))
        if cached_files:
            logger.info("Using cached Purdue uSMART data: %s", cached_files[0])
            return pd.read_csv(cached_files[0])

        # Download from figshare
        figshare_id = config.EXTERNAL_DATASETS["purdue_usmart"]["figshare_id"]
        url = f"https://ndownloader.figshare.com/files/{figshare_id}"
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()

        # figshare may serve CSVs with different extensions — try to parse
        content = resp.content.decode("utf-8", errors="replace")
        df = pd.read_csv(StringIO(content))

        # Cache it
        cache_path = cache_dir / "purdue_usmart.csv"
        df.to_csv(cache_path, index=False)
        logger.info("Cached Purdue uSMART data to %s (%d rows)", cache_path, len(df))
        return df

    except Exception as exc:
        logger.warning("Could not load Purdue uSMART dataset: %s", exc)
        return None


def load_phmsa_data(path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load PHMSA pipeline incident data.

    PHMSA provides rich cause taxonomies (corrosion, excavation damage, etc.)
    that complement the maritime-focused NOAA tags.

    Args:
        path: File path to downloaded PHMSA CSV.

    Returns:
        DataFrame or None if file is missing or unreadable.
    """
    if path is None:
        # Check default download location
        default_path = Path(config.DATA_EXTERNAL_DIR) / "phmsa" / "pipeline_incidents.csv"
        if default_path.exists():
            path = str(default_path)
        else:
            logger.warning(
                "PHMSA data not found at %s. Download from %s",
                default_path, config.EXTERNAL_DATASETS["phmsa"]["url"],
            )
            return None

    logger.info("Loading PHMSA data from %s", path)
    try:
        df = pd.read_csv(path, low_memory=False)
        logger.info("Loaded %d PHMSA incidents", len(df))
        return df
    except Exception as exc:
        logger.warning("Could not load PHMSA data: %s", exc)
        return None


def load_kaggle_dataset(data_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load the Kaggle oil spill incidents dataset via kagglehub.

    Args:
        data_dir: Cache directory.

    Returns:
        DataFrame or None if kagglehub is unavailable or download fails.
    """
    if data_dir is None:
        data_dir = str(Path(config.DATA_EXTERNAL_DIR) / "kaggle")

    logger.info("Attempting to load Kaggle dataset '%s' ...",
                config.EXTERNAL_DATASETS["kaggle_oil_spill"]["dataset_id"])
    try:
        import kagglehub
        download_path = kagglehub.dataset_download(
            config.EXTERNAL_DATASETS["kaggle_oil_spill"]["dataset_id"],
            path=data_dir,
        )
        csv_files = list(Path(download_path).glob("*.csv"))
        if not csv_files:
            logger.warning("No CSV files found in Kaggle download at %s", download_path)
            return None

        df = pd.read_csv(csv_files[0])
        logger.info("Loaded %d Kaggle incidents from %s", len(df), csv_files[0])
        return df

    except Exception as exc:
        logger.warning("Could not load Kaggle dataset: %s", exc)
        return None


def load_all_data(use_external: bool = False) -> dict[str, pd.DataFrame]:
    """Convenience loader — returns all primary data plus optionally external.

    Args:
        use_external: If True, attempt to load Purdue, PHMSA, and Kaggle datasets.

    Returns:
        Dict with keys 'incidents', 'posts', and optionally 'purdue', 'phmsa', 'kaggle'.
    """
    # Ensure raw data is in the right place
    move_raw_data()

    data = {
        "incidents": load_incidents(),
        "posts": load_posts(),
    }

    if use_external:
        for name, loader in [
            ("purdue", load_purdue_usmart),
            ("phmsa", load_phmsa_data),
            ("kaggle", load_kaggle_dataset),
        ]:
            result = loader()
            if result is not None:
                data[name] = result

    return data
