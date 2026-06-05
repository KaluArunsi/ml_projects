import pandas as pd


def test_streamlit_app_imports():
    import app

    assert app.PROJECT_ROOT.name == "openbpo-drift"


def test_alert_for_entity_kpi_returns_matching_alert():
    from app import alert_for_entity_kpi

    alerts = pd.DataFrame(
        [
            {"entity_id": "A001", "kpi_name": "qa", "severity": "High"},
            {"entity_id": "A002", "kpi_name": "late_count", "severity": "Medium"},
        ]
    )

    match = alert_for_entity_kpi(alerts, "A002", "late_count")

    assert match is not None
    assert match["severity"] == "Medium"
    assert alert_for_entity_kpi(alerts, "A003", "late_count") is None
