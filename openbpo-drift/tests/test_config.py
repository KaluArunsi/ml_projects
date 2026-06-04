import pytest

from src.config import merge_kpi_rules


def test_merge_kpi_rules_reports_bad_mapping_entries():
    with pytest.raises(ValueError, match="Missing kpi_name"):
        merge_kpi_rules({}, [{"source_column": "AHT", "include": True}])

    with pytest.raises(ValueError, match="Invalid drift_threshold_pct"):
        merge_kpi_rules({}, [{"source_column": "AHT", "kpi_name": "aht", "drift_threshold_pct": "bad", "include": True}])

    with pytest.raises(ValueError, match="must be > 0 and <= 100"):
        merge_kpi_rules({}, [{"source_column": "AHT", "kpi_name": "aht", "drift_threshold_pct": 101, "include": True}])
