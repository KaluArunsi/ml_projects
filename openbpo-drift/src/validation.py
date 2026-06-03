# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd


def _status(level: str, count: int, fail_when_positive: bool = False) -> str:
    if level == "info":
        return "Info"
    if count == 0:
        return "Pass"
    return "Fail" if fail_when_positive else "Warning"


def _append(rows: List[Dict[str, object]], status: str, check: str, details: str, affected_rows: int = 0) -> None:
    rows.append(
        {
            "status": status,
            "check": check,
            "details": details,
            "affected_rows": int(affected_rows),
        }
    )


def validate_normalized_data(
    df: pd.DataFrame,
    kpi_rules: Dict[str, Dict[str, object]] | None = None,
    raw_row_count: int | None = None,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    raw_count_text = "raw rows: {}".format(raw_row_count) if raw_row_count is not None else "raw rows: unknown"
    _append(rows, "Info", "row_count", "{}, normalized rows: {}".format(raw_count_text, len(df)))

    invalid_dates = int(df["date"].isna().sum())
    _append(rows, _status("check", invalid_dates, True), "date_parse", "Rows with invalid dates after parsing.", invalid_dates)

    missing_entity = int(df["entity_id"].fillna("").astype(str).str.strip().eq("").sum())
    _append(rows, _status("check", missing_entity, True), "entity_id_missing", "Rows missing entity IDs.", missing_entity)

    raw_values = df["raw_kpi_value"] if "raw_kpi_value" in df.columns else df["kpi_value"]
    raw_missing = raw_values.isna() | raw_values.astype(str).str.strip().eq("")
    non_numeric = int((~raw_missing & df["kpi_value"].isna()).sum())
    missing_kpi = int(raw_missing.sum())
    _append(rows, _status("check", non_numeric), "kpi_value_numeric", "Rows where KPI values could not be parsed as numeric.", non_numeric)
    _append(rows, _status("check", missing_kpi), "missing_kpi_values", "Rows where KPI values are blank or missing.", missing_kpi)

    duplicate_mask = df.duplicated(subset=["date", "entity_id", "kpi_name"], keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    _append(rows, _status("check", duplicate_rows), "duplicate_observations", "Duplicate date/entity/KPI observations.", duplicate_rows)

    if not df["date"].dropna().empty:
        date_min = df["date"].min().strftime("%Y-%m-%d")
        date_max = df["date"].max().strftime("%Y-%m-%d")
        _append(rows, "Info", "min_date_max_date", "Date range: {} to {}.".format(date_min, date_max))
    else:
        _append(rows, "Info", "min_date_max_date", "Date range unavailable because all dates are invalid.")

    _append(rows, "Info", "entities_count", "Unique entities: {}.".format(int(df["entity_id"].nunique(dropna=True))))
    _append(rows, "Info", "kpis_count", "Unique KPIs: {}.".format(int(df["kpi_name"].nunique(dropna=True))))

    if kpi_rules:
        out_of_range = 0
        details = []
        for kpi_name, rule in kpi_rules.items():
            validation = rule.get("validation", {})
            subset = df[df["kpi_name"] == kpi_name]
            if subset.empty:
                continue
            mask = pd.Series(False, index=subset.index)
            if "min" in validation:
                mask |= subset["kpi_value"] < float(validation["min"])
            if "max" in validation:
                mask |= subset["kpi_value"] > float(validation["max"])
            count = int(mask.sum())
            if count:
                out_of_range += count
                details.append("{}: {}".format(kpi_name, count))
        message = "Out-of-range KPI values by rule." if details else "No out-of-range KPI values found."
        if details:
            message = "{} {}".format(message, "; ".join(details))
        _append(rows, _status("check", out_of_range), "kpi_range", message, out_of_range)

    return pd.DataFrame(rows)
