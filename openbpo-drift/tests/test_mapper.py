import pandas as pd

from src.mapper import normalize_to_long
from src.schema import CANONICAL_COLUMNS


def test_normalize_to_long_creates_expected_canonical_rows():
    raw = {
        "Date": ["2026-06-01", "2026-06-02"],
        "Agent ID": ["A001", "A002"],
        "Team": ["Team A", "Team B"],
        "AHT": [420, 440],
        "QA Score": [91, 88],
    }
    frame = normalize_to_long(
        pd.DataFrame(raw),
        field_mapping={"date": "Date", "entity_id": "Agent ID", "team": "Team", "site": None, "account": None, "shift": None},
        kpi_mapping=[
            {"source_column": "AHT", "kpi_name": "aht", "unit": "seconds", "direction_bad": "up", "drift_threshold_pct": 15, "include": True},
            {"source_column": "QA Score", "kpi_name": "qa", "unit": "percentage", "direction_bad": "down", "drift_threshold_pct": 5, "include": True},
        ],
    )

    assert len(frame) == 4
    assert list(frame.columns) == CANONICAL_COLUMNS
    assert frame["date"].notna().all()
    assert frame["kpi_value"].dtype.kind in {"i", "f"}
