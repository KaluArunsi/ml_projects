# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from datetime import datetime

import pandas as pd


def summarize_monitoring(normalized_df: pd.DataFrame, alerts: pd.DataFrame) -> dict:
    date_range = "No data"
    date_days = 0
    if not normalized_df.empty and normalized_df["date"].notna().any():
        start = normalized_df["date"].min()
        end = normalized_df["date"].max()
        date_days = int((end - start).days) + 1
        date_range = "{} to {}".format(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    return {
        "entities_monitored": int(normalized_df["entity_id"].nunique(dropna=True)) if not normalized_df.empty else 0,
        "kpis_monitored": int(normalized_df["kpi_name"].nunique(dropna=True)) if not normalized_df.empty else 0,
        "active_alerts": int(len(alerts)),
        "high_severity_alerts": int((alerts["severity"] == "High").sum()) if not alerts.empty else 0,
        "date_range": date_range,
        "date_days": date_days,
    }


def alerts_for_display(alerts: pd.DataFrame) -> pd.DataFrame:
    if alerts.empty:
        return alerts
    columns = [
        "severity",
        "entity_id",
        "team",
        "account",
        "kpi_name",
        "baseline",
        "current",
        "drift_pct",
        "direction_bad",
        "threshold_pct",
        "observations_used",
        "latest_date",
        "explanation",
    ]
    return alerts.loc[:, columns].rename(
        columns={
            "severity": "Severity",
            "entity_id": "Entity ID",
            "team": "Team",
            "account": "Account",
            "kpi_name": "KPI",
            "baseline": "Baseline",
            "current": "Current",
            "drift_pct": "Drift %",
            "direction_bad": "Direction (Bad)",
            "threshold_pct": "Threshold %",
            "observations_used": "Observations",
            "latest_date": "Latest Date",
            "explanation": "Explanation",
        }
    )


def generate_markdown_report(alerts: pd.DataFrame, normalized_df: pd.DataFrame, quality_df: pd.DataFrame) -> str:
    summary = summarize_monitoring(normalized_df, alerts)
    lines = [
        "# OpenBPO Drift Report",
        "",
        "Generated: {}".format(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")),
        "",
        "## Summary",
        "",
        "- Entities monitored: {}".format(summary["entities_monitored"]),
        "- KPIs monitored: {}".format(summary["kpis_monitored"]),
        "- Date range: {}".format(summary["date_range"]),
        "- Total alerts: {}".format(summary["active_alerts"]),
        "- High severity alerts: {}".format(summary["high_severity_alerts"]),
        "",
        "## Top Alerts",
        "",
        "| Severity | Entity | KPI | Drift | Baseline | Current |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]

    if alerts.empty:
        lines.append("| None | - | - | - | - | - |")
    else:
        for row in alerts.head(10).itertuples(index=False):
            lines.append(
                "| {severity} | {entity_id} | {kpi_name} | {drift_pct:.1f}% | {baseline:.2f} | {current:.2f} |".format(
                    severity=row.severity,
                    entity_id=row.entity_id,
                    kpi_name=row.kpi_name,
                    drift_pct=row.drift_pct,
                    baseline=row.baseline,
                    current=row.current,
                )
            )

    lines.extend(
        [
            "",
            "## Data Quality",
            "",
            "| Status | Check | Details |",
            "| --- | --- | --- |",
        ]
    )
    for row in quality_df.itertuples(index=False):
        lines.append("| {status} | {check} | {details} |".format(status=row.status, check=row.check, details=row.details))

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "This report was generated locally using OpenBPO Drift.",
        ]
    )
    return "\n".join(lines)
