# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import re
from typing import Dict, List

import pandas as pd
import yaml

from src.schema import CANONICAL_COLUMNS, DEFAULT_ENTITY_TYPE, OPTIONAL_FIELDS


def slugify_column_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def build_default_kpi_mapping(
    columns: List[str],
    selected_kpis: List[str],
    default_rules: Dict[str, Dict[str, object]],
    default_threshold_pct: float,
) -> List[Dict[str, object]]:
    mapping = []
    for column in selected_kpis:
        if column not in columns:
            continue
        kpi_name = slugify_column_name(column)
        defaults = default_rules.get(kpi_name, {})
        mapping.append(
            {
                "source_column": column,
                "kpi_name": kpi_name,
                "unit": defaults.get("unit", ""),
                "direction_bad": defaults.get("direction_bad", "up"),
                "drift_threshold_pct": float(defaults.get("drift_threshold_pct", default_threshold_pct)),
                "include": True,
            }
        )
    return mapping


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
    included = [item for item in kpi_mapping if item.get("include", True) and str(item["source_column"]) in df.columns]
    if not included:
        columns = CANONICAL_COLUMNS + (["raw_kpi_value"] if include_raw_value else [])
        return pd.DataFrame(columns=columns)

    id_frame = pd.DataFrame(
        {
            "date": _column_or_none(df, field_mapping.get("date")),
            "entity_id": _column_or_none(df, field_mapping.get("entity_id")),
            "entity_type": default_entity_type,
            "team": _column_or_none(df, field_mapping.get("team")),
            "site": _column_or_none(df, field_mapping.get("site")),
            "account": _column_or_none(df, field_mapping.get("account")),
            "shift": _column_or_none(df, field_mapping.get("shift")),
        }
    )
    working = id_frame.join(df[[str(item["source_column"]) for item in included]])
    source_columns = [str(item["source_column"]) for item in included]
    metadata = {
        str(item["source_column"]): {
            "kpi_name": str(item.get("kpi_name") or slugify_column_name(str(item["source_column"]))),
            "unit": str(item.get("unit") or ""),
        }
        for item in included
    }

    normalized = working.melt(
        id_vars=["date", "entity_id", "entity_type", "team", "site", "account", "shift"],
        value_vars=source_columns,
        var_name="source_column",
        value_name="kpi_value",
    )
    if include_raw_value:
        normalized["raw_kpi_value"] = normalized["kpi_value"]

    normalized["kpi_name"] = normalized["source_column"].map(lambda column: metadata[column]["kpi_name"])
    normalized["unit"] = normalized["source_column"].map(lambda column: metadata[column]["unit"])
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
