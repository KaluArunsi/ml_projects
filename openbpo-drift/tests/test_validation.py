import pandas as pd

from src.validation import validate_normalized_data


def test_validation_flags_invalid_dates_missing_entities_and_duplicates():
    frame = pd.DataFrame(
        {
            "date": [pd.NaT, pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-01")],
            "entity_id": [None, "A001", "A001"],
            "entity_type": ["agent", "agent", "agent"],
            "team": [None, "Team A", "Team A"],
            "site": [None, "Manila", "Manila"],
            "account": [None, "Telco", "Telco"],
            "shift": [None, "Night", "Night"],
            "kpi_name": ["aht", "aht", "aht"],
            "kpi_value": [None, 420, 420],
            "unit": ["seconds", "seconds", "seconds"],
            "source_column": ["AHT", "AHT", "AHT"],
            "raw_kpi_value": ["bad", 420, 420],
        }
    )

    quality = validate_normalized_data(frame)

    assert (quality["check"] == "date_parse").any()
    assert (quality["check"] == "entity_id_missing").any()
    duplicates = quality.loc[quality["check"] == "duplicate_observations", "affected_rows"].iloc[0]
    assert duplicates == 2
