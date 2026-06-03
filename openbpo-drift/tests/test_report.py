import pandas as pd

from src.report import alerts_for_display, summarize_monitoring


def test_summarize_monitoring_includes_date_span_and_alert_counts():
    normalized = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-01", "2026-04-03"]),
            "entity_id": ["A001", "A002"],
            "kpi_name": ["aht", "qa"],
        }
    )
    alerts = pd.DataFrame(
        {
            "severity": ["High", "Watch"],
            "entity_id": ["A001", "A002"],
        }
    )

    summary = summarize_monitoring(normalized, alerts)

    assert summary["entities_monitored"] == 2
    assert summary["kpis_monitored"] == 2
    assert summary["active_alerts"] == 2
    assert summary["high_severity_alerts"] == 1
    assert summary["date_days"] == 3


def test_alerts_for_display_exposes_prd_table_columns():
    alerts = pd.DataFrame(
        [
            {
                "severity": "High",
                "entity_id": "A001",
                "team": "Team A",
                "account": "Telco",
                "kpi_name": "aht",
                "baseline": 420.0,
                "current": 560.0,
                "drift_pct": 33.3,
                "direction_bad": "up",
                "threshold_pct": 15.0,
                "observations_used": 17,
                "latest_date": "2026-06-01",
                "explanation": "AHT drifted.",
            }
        ]
    )

    display = alerts_for_display(alerts)

    assert list(display.columns) == [
        "Severity",
        "Entity ID",
        "Team",
        "Account",
        "KPI",
        "Baseline",
        "Current",
        "Drift %",
        "Direction (Bad)",
        "Threshold %",
        "Observations",
        "Latest Date",
        "Explanation",
    ]
