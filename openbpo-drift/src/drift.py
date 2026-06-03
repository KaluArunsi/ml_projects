# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from typing import Dict

import pandas as pd

from src.config import DEFAULT_THRESHOLD_PCT


SEVERITY_RANK = {"High": 3, "Medium": 2, "Watch": 1}


def classify_severity(abs_drift_pct: float) -> str:
    if abs_drift_pct >= 30:
        return "High"
    if abs_drift_pct >= 15:
        return "Medium"
    return "Watch"


def _direction_is_bad(drift_pct: float, direction_bad: str, threshold_pct: float) -> bool:
    if direction_bad == "up":
        return drift_pct >= threshold_pct
    return drift_pct <= -threshold_pct


def _suggested_check(rule: Dict[str, object]) -> str:
    drivers = rule.get("drivers", [])
    if drivers:
        return "Review supporting KPIs: {}.".format(", ".join(str(driver) for driver in drivers))
    return "Review recent workflow, staffing, and process changes affecting this KPI."


def explain_alert(row: Dict[str, object]) -> str:
    direction_word = "increased" if float(row["drift_pct"]) > 0 else "decreased"
    return (
        "{entity_id}'s {kpi_name} {direction_word} by {drift:.1f}% versus baseline. "
        "Configured bad direction is '{direction_bad}'. Severity: {severity}. "
        "{suggested_check}"
    ).format(
        entity_id=row["entity_id"],
        kpi_name=row["kpi_name"],
        direction_word=direction_word,
        drift=abs(float(row["drift_pct"])),
        direction_bad=row["direction_bad"],
        severity=row["severity"],
        suggested_check=row["suggested_check"],
    )


def build_signal_frame(
    df: pd.DataFrame,
    entity_id: str,
    kpi_name: str,
    baseline_window: int = 14,
    current_window: int = 3,
) -> pd.DataFrame:
    subset = (
        df[(df["entity_id"] == entity_id) & (df["kpi_name"] == kpi_name)]
        .dropna(subset=["date", "kpi_value"])
        .groupby("date", as_index=False)["kpi_value"]
        .mean()
        .sort_values("date")
        .reset_index(drop=True)
    )
    subset["baseline_mean"] = pd.NA
    for idx in range(len(subset)):
        if idx < baseline_window:
            continue
        subset.loc[idx, "baseline_mean"] = float(subset.loc[idx - baseline_window : idx - 1, "kpi_value"].mean())
    subset["is_current_window"] = False
    if len(subset) >= current_window:
        subset.loc[subset.index[-current_window:], "is_current_window"] = True
    return subset


def detect_rolling_drift(
    df: pd.DataFrame,
    kpi_rules: Dict[str, Dict[str, object]],
    entity_col: str = "entity_id",
    baseline_window: int = 14,
    current_window: int = 3,
) -> pd.DataFrame:
    working = df.dropna(subset=["date", entity_col, "kpi_name", "kpi_value"]).copy()
    alerts = []
    observations_needed = baseline_window + current_window

    for (entity_id, kpi_name), group in working.groupby([entity_col, "kpi_name"], dropna=False):
        series = group.groupby("date")["kpi_value"].mean().sort_index()
        if len(series) < observations_needed:
            continue

        baseline = series.iloc[-observations_needed:-current_window]
        current = series.iloc[-current_window:]
        baseline_mean = float(baseline.mean())
        current_mean = float(current.mean())
        if abs(baseline_mean) < 1e-9:
            continue

        drift_pct = ((current_mean - baseline_mean) / baseline_mean) * 100.0
        rule = kpi_rules.get(kpi_name, {})
        direction_bad = str(rule.get("direction_bad", "up"))
        threshold_pct = float(rule.get("drift_threshold_pct", DEFAULT_THRESHOLD_PCT))
        if not _direction_is_bad(drift_pct, direction_bad, threshold_pct):
            continue

        latest_row = group.sort_values("date").iloc[-1]
        abs_drift = abs(drift_pct)
        severity = classify_severity(abs_drift)
        alert = {
            "severity": severity,
            "severity_rank": SEVERITY_RANK[severity],
            "entity_id": str(entity_id),
            "entity_type": latest_row.get("entity_type", "agent"),
            "team": latest_row.get("team"),
            "site": latest_row.get("site"),
            "account": latest_row.get("account"),
            "shift": latest_row.get("shift"),
            "kpi_name": kpi_name,
            "kpi_label": rule.get("label", kpi_name.replace("_", " ").title()),
            "baseline": baseline_mean,
            "current": current_mean,
            "drift_pct": drift_pct,
            "abs_drift_pct": abs_drift,
            "direction_bad": direction_bad,
            "threshold_pct": threshold_pct,
            "method": "rolling_baseline",
            "latest_date": series.index[-1].strftime("%Y-%m-%d"),
            "observations_used": observations_needed,
            "unit": rule.get("unit", ""),
            "suggested_check": _suggested_check(rule),
        }
        alert["explanation"] = explain_alert(alert)
        alerts.append(alert)

    alerts_df = pd.DataFrame(alerts)
    if alerts_df.empty:
        return alerts_df
    return alerts_df.sort_values(
        ["severity_rank", "abs_drift_pct", "kpi_name", "entity_id"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
