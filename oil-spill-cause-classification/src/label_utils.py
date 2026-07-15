"""
Label Utilities
===============
Taxonomy mapping for external datasets and commodity synonym management.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from . import config

logger = logging.getLogger(__name__)


class TaxonomyMapper:
    """Maps external dataset cause labels → unified oil spill cause taxonomy.

    Uses YAML mapping files stored in ``config.LABEL_MAPS_DIR``.
    Each mapping file defines how an external dataset's cause labels
    translate into our unified 15-label schema.

    Supports multi-label mappings (one external cause → multiple unified labels)
    via list values, with optional confidence weights.
    """

    def __init__(self, mapping_dir: Optional[str] = None):
        self.mapping_dir = Path(mapping_dir or config.LABEL_MAPS_DIR)
        self.maps: dict[str, dict] = {}
        self.commodity_synonyms: dict[str, str] = {}

        # Load built-in mappings from config
        self._load_builtin_mappings()
        # Load any YAML overrides from disk
        self._load_yaml_mappings()

    def _load_builtin_mappings(self) -> None:
        """Load the taxonomy mappings defined in config.CAUSE_MAPPING."""
        self.maps = dict(config.CAUSE_MAPPING)

    def _load_yaml_mappings(self) -> None:
        """Load YAML mapping files from disk. These override/extend built-ins."""
        self.mapping_dir.mkdir(parents=True, exist_ok=True)

        # Load commodity synonyms if present
        synonyms_path = self.mapping_dir / "commodity_synonyms.yaml"
        if synonyms_path.exists():
            with open(synonyms_path) as f:
                self.commodity_synonyms = yaml.safe_load(f) or {}
            logger.info("Loaded %d commodity synonyms from %s",
                         len(self.commodity_synonyms), synonyms_path)

        # Load per-source taxonomy maps
        for yaml_file in self.mapping_dir.glob("*_to_unified.yaml"):
            source = yaml_file.stem.replace("_to_unified", "")
            with open(yaml_file) as f:
                mapping = yaml.safe_load(f)
            if mapping:
                self.maps[source] = mapping
                logger.info("Loaded taxonomy map '%s' with %d entries from %s",
                             source, len(mapping), yaml_file)

    def get_source_names(self) -> list[str]:
        """Return available external source names."""
        return list(self.maps.keys())

    def map_labels(
        self,
        source: str,
        external_labels: pd.Series,
        default_label: str = "Other",
    ) -> Optional[pd.DataFrame]:
        """Map external cause labels to the unified multi-label matrix.

        Args:
            source: Source name (e.g. "phmsa", "kaggle").
            external_labels: Series of external cause strings.
            default_label: Label to use when no mapping is found.

        Returns:
            Multi-label binary matrix of shape (n_samples, n_labels), or None
            if the source is unknown.
        """
        if source not in self.maps:
            logger.warning("Unknown source '%s'. Available: %s",
                           source, list(self.maps.keys()))
            return None

        source_map = self.maps[source]
        n = len(external_labels)
        n_labels = config.NUM_LABELS

        # Build label name → index lookup
        label_to_idx = {name: i for i, name in enumerate(config.LABEL_NAMES)}

        matrix = np.zeros((n, n_labels), dtype=np.int8)

        for i, raw_label in enumerate(external_labels):
            if pd.isna(raw_label) or not str(raw_label).strip():
                continue

            key = str(raw_label).lower().strip().replace(" ", "_")
            mapped = source_map.get(key)

            if mapped is None:
                # Try fuzzy: replace underscores and re-check
                key_alt = key.replace("_", " ")
                mapped = source_map.get(key_alt)

            if mapped is None:
                # Assign to default
                if default_label in label_to_idx:
                    matrix[i, label_to_idx[default_label]] = 1
                continue

            # Handle both single string and list values
            if isinstance(mapped, str):
                mapped = [mapped]

            for label_name in mapped:
                if label_name in label_to_idx:
                    matrix[i, label_to_idx[label_name]] = 1

        n_mapped = int((matrix.sum(axis=1) > 0).sum())
        logger.info("Mapped %d/%d rows for source '%s'", n_mapped, n, source)
        return pd.DataFrame(matrix, columns=config.LABEL_NAMES)

    def normalize_commodity(self, raw_series: pd.Series) -> pd.Series:
        """Apply commodity synonym map to a raw commodity series.

        Uses both the built-in config.COMMODITY_CATEGORIES and any YAML overrides.
        """
        # Build merged lookup
        merged = dict(config.COMMODITY_CATEGORIES)
        # YAML overrides: canonical → list of variants
        for canonical, variants in self.commodity_synonyms.items():
            if isinstance(variants, list):
                merged.setdefault(canonical, []).extend(variants)

        # Invert: variant → canonical
        lookup: dict[str, str] = {}
        for canonical, variants in merged.items():
            for v in variants:
                lookup[v.lower().strip()] = canonical

        def _map(val):
            if pd.isna(val) or not isinstance(val, str):
                return config.UNKNOWN_COMMODITY_LABEL
            cleaned = val.lower().strip()
            return lookup.get(cleaned, config.UNKNOWN_COMMODITY_LABEL)

        return raw_series.apply(_map)

    def export_unified_labels(
        self,
        external_labels: dict[str, pd.Series],
    ) -> dict[str, pd.DataFrame]:
        """Batch-map multiple external label sets and return a dict of matrices.

        Args:
            external_labels: Dict mapping source_name → label_series.

        Returns:
            Dict mapping source_name → binary label_matrix DataFrame.
        """
        results: dict[str, pd.DataFrame] = {}
        for source, series in external_labels.items():
            matrix = self.map_labels(source, series)
            if matrix is not None:
                results[source] = matrix
        return results
