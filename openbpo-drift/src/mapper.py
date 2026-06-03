# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import re
from typing import Dict, Iterable, List

import pandas as pd
import yaml

from src.schema import CANONICAL_COLUMNS, DEFAULT_ENTITY_TYPE, OPTIONAL_FIELDS


def slugify_column_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def _column_or_none(df: pd.DataFrame, column_name: str | None):
    if column_name and column_name in df.columns:
        return df[column_name]
    return None


def normalize_to_long(
    df: pd.DataFrame,
    field_mapping: Dict[str, str | None],
    kpi_mapping: List[Dict[str, object]],
    default_entity_type: str = DEFAULT_ENTITY_TYPE,
    include_raw_value: bool = False,
) -> pd.DataFrame:
    normalized_frames = []
    for kpi in kpi_mapping:
        if not kpi.get("include", True):
            continue

        source_column = str(kpi["source_column"])
        if source_column not in df.columns:
            continue

        frame = pd.DataFrame(
            {
                "date": _column_or_none(df, field_mapping.get("date")),
                "entity_id": _column_or_none(df, field_mapping.get("entity_id")),
                "entity_type": default_entity_type,
                "team": _column_or_none(df, field_mapping.get("team")),
                "site": _column_or_none(df, field_mapping.get("site")),
                "account": _column_or_none(df, field_mapping.get("account")),
                "shift": _column_or_none(df, field_mapping.get("shift")),
                "kpi_name": str(kpi.get("kpi_name") or slugify_column_name(source_column)),
                "kpi_value": df[source_column],
                "unit": str(kpi.get("unit") or ""),
                "source_column": source_column,
            }
        )
        if include_raw_value:
            frame["raw_kpi_value"] = df[source_column]
        normalized_frames.append(frame)

    if not normalized_frames:
        columns = CANONICAL_COLUMNS + (["raw_kpi_value"] if include_raw_value else [])
        return pd.DataFrame(columns=columns)

    normalized = pd.concat(normalized_frames, ignore_index=True)
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["entity_id"] = normalized["entity_id"].astype("string").str.strip()
    for column in OPTIONAL_FIELDS:
        normalized[column] = normalized[column].astype("string").str.strip()
    normalized["kpi_name"] = normalized["kpi_name"].astype("string").str.strip().str.lower()
    normalized["unit"] = normalized["unit"].astype("string").str.strip()
    normalized["source_column"] = normalized["source_column"].astype("string").str.strip()
    normalized["kpi_value"] = pd.to_numeric(normalized["kpi_value"], errors="coerce")

    ordered_columns = CANONICAL_COLUMNS + (["raw_kpi_value"] if include_raw_value else [])
    return normalized.loc[:, ordered_columns]


def build_mapping_yaml(field_mapping: Dict[str, str | None], kpi_mapping: List[Dict[str, object]]) -> str:
    payload = {
        "field_mapping": field_mapping,
        "kpis": [
            {
                "source_column": item["source_column"],
                "kpi_name": item["kpi_name"],
                "unit": item.get("unit") or "",
                "direction_bad": item.get("direction_bad", "up"),
                "drift_threshold_pct": float(item.get("drift_threshold_pct", 15)),
                "include": bool(item.get("include", True)),
            }
            for item in kpi_mapping
        ],
    }
    return yaml.safe_dump(payload, sort_keys=False)
