import pandas as pd
from pathlib import Path

from src.config import load_kpi_rules, merge_kpi_rules
from src.drift import detect_rolling_drift


def _normalized_frame() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2026-05-01", periods=24, freq="D")
    for date in dates:
        for agent_id in ["A001", "A002"]:
            drifting = agent_id == "A001" and date >= pd.Timestamp("2026-05-22")
            rows.append(
                {
                    "date": date,
                    "entity_id": agent_id,
                    "entity_type": "agent",
                    "team": "Team Manila A",
                    "site": "Manila",
                    "account": "Telco",
                    "shift": "Night",
                    "kpi_name": "aht",
                    "kpi_value": 560 if drifting else 420,
                    "unit": "seconds",
                    "source_column": "AHT",
                }
            )
            rows.append(
                {
                    "date": date,
                    "entity_id": agent_id,
                    "entity_type": "agent",
                    "team": "Team Manila A",
                    "site": "Manila",
                    "account": "Telco",
                    "shift": "Night",
                    "kpi_name": "qa",
                    "kpi_value": 91 if agent_id == "A001" else 90,
                    "unit": "percentage",
                    "source_column": "QA Score",
                }
            )
    return pd.DataFrame(rows)


def test_detect_rolling_drift_flags_known_drift_and_respects_direction():
    default_rules = load_kpi_rules(Path("configs/default_kpi_rules.yaml"))
    merged_rules = merge_kpi_rules(
        default_rules,
        [
            {"source_column": "AHT", "kpi_name": "aht", "unit": "seconds", "direction_bad": "up", "drift_threshold_pct": 15, "include": True},
            {"source_column": "QA Score", "kpi_name": "qa", "unit": "percentage", "direction_bad": "down", "drift_threshold_pct": 5, "include": True},
        ],
    )

    alerts = detect_rolling_drift(_normalized_frame(), merged_rules, baseline_window=14, current_window=3)

    aht_alert = alerts[(alerts["entity_id"] == "A001") & (alerts["kpi_name"] == "aht")]
    assert not aht_alert.empty
    assert aht_alert.iloc[0]["severity"] == "High"
    assert alerts[(alerts["entity_id"] == "A002") & (alerts["kpi_name"] == "qa")].empty
