# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from src import (
    CANONICAL_COLUMNS,
    DEFAULT_BASELINE_WINDOW,
    DEFAULT_CURRENT_WINDOW,
    DEFAULT_THRESHOLD_PCT,
    FIELD_LABELS,
    REQUIRED_FIELDS,
    alerts_for_display,
    build_mapping_yaml,
    detect_rolling_drift,
    generate_markdown_report,
    generate_sample_bpo_kpis,
    guess_field_mapping,
    infer_kpi_candidates,
    list_excel_sheets,
    load_kpi_rules,
    load_tabular_file,
    make_kpi_trend_chart,
    merge_kpi_rules,
    normalize_to_long,
    slugify_column_name,
    summarize_monitoring,
    validate_normalized_data,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RULES_PATH = PROJECT_ROOT / "configs" / "default_kpi_rules.yaml"
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample_bpo_kpis.csv"
SAMPLE_MAPPING_PATH = PROJECT_ROOT / "configs" / "sample_mapping.yaml"
NONE_OPTION = "None"


def _ensure_sample_data() -> None:
    if not SAMPLE_DATA_PATH.exists():
        generate_sample_bpo_kpis(SAMPLE_DATA_PATH)


def _load_source_frame(uploaded_file, use_sample_data: bool, sheet_name: str | int = 0) -> pd.DataFrame:
    if use_sample_data or uploaded_file is None:
        return load_tabular_file(SAMPLE_DATA_PATH)
    return load_tabular_file(uploaded_file, sheet_name=sheet_name, filename=uploaded_file.name)


def _load_sample_mapping() -> dict:
    return yaml.safe_load(SAMPLE_MAPPING_PATH.read_text())


def _select_index(options: list[str], value: str | None) -> int:
    return options.index(value) if value in options else 0


def _build_kpi_mapping(raw_df: pd.DataFrame, guessed_fields: dict, default_rules: dict, use_sample_data: bool) -> list[dict]:
    if use_sample_data:
        return list(_load_sample_mapping()["kpis"])

    candidates = infer_kpi_candidates(raw_df.columns, guessed_fields)
    mapping = []
    for column in candidates:
        kpi_name = slugify_column_name(column)
        defaults = default_rules.get(kpi_name, {})
        mapping.append(
            {
                "source_column": column,
                "kpi_name": kpi_name,
                "unit": defaults.get("unit", ""),
                "direction_bad": defaults.get("direction_bad", "up"),
                "drift_threshold_pct": float(defaults.get("drift_threshold_pct", DEFAULT_THRESHOLD_PCT)),
                "include": True,
            }
        )
    return mapping


def _filter_alerts(alerts: pd.DataFrame) -> pd.DataFrame:
    filtered = alerts.copy()
    if filtered.empty:
        return filtered

    severity_options = ["All"] + sorted(filtered["severity"].unique().tolist(), key=lambda value: {"High": 0, "Medium": 1, "Watch": 2}.get(value, 3))
    selected_severity = st.selectbox("Severity", severity_options, index=0)
    if selected_severity != "All":
        filtered = filtered[filtered["severity"] == selected_severity]

    kpi_options = ["All"] + sorted(filtered["kpi_name"].dropna().unique().tolist())
    selected_kpi = st.selectbox("KPI", kpi_options, index=0)
    if selected_kpi != "All":
        filtered = filtered[filtered["kpi_name"] == selected_kpi]

    for column, label in [("team", "Team"), ("account", "Account")]:
        values = filtered[column].dropna().astype(str).unique().tolist() if column in filtered.columns else []
        if values:
            options = ["All"] + sorted(values)
            selected = st.selectbox(label, options, index=0)
            if selected != "All":
                filtered = filtered[filtered[column].astype(str) == selected]
    return filtered


def main() -> None:
    _ensure_sample_data()
    default_rules = load_kpi_rules(DEFAULT_RULES_PATH)

    st.set_page_config(page_title="OpenBPO Drift", page_icon=":chart_with_upwards_trend:", layout="wide")
    st.title("OpenBPO Drift")
    st.caption("Local-first KPI drift monitoring for BPO and contact center teams.")

    if "use_sample_data" not in st.session_state:
        st.session_state.use_sample_data = True

    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if st.sidebar.button("Load sample data"):
        st.session_state.use_sample_data = True
    elif uploaded_file is not None:
        st.session_state.use_sample_data = False

    sheet_name: str | int = 0
    if uploaded_file is not None and Path(uploaded_file.name).suffix.lower() in {".xlsx", ".xls"}:
        sheet_options = list(list_excel_sheets(uploaded_file))
        sheet_name = st.sidebar.selectbox("Excel sheet", sheet_options, index=0)

    baseline_window = st.sidebar.slider("Baseline window", min_value=7, max_value=30, value=DEFAULT_BASELINE_WINDOW)
    current_window = st.sidebar.slider("Current window", min_value=3, max_value=14, value=DEFAULT_CURRENT_WINDOW)
    default_threshold_pct = st.sidebar.number_input("Default drift threshold %", min_value=1.0, max_value=100.0, value=DEFAULT_THRESHOLD_PCT, step=1.0)
    st.sidebar.caption("Exports appear in the Export tab once data is normalized.")

    try:
        raw_df = _load_source_frame(uploaded_file, st.session_state.use_sample_data, sheet_name=sheet_name)
    except Exception as exc:
        st.error(str(exc))
        return

    guessed_fields = _load_sample_mapping()["field_mapping"] if st.session_state.use_sample_data else guess_field_mapping(raw_df.columns)
    default_kpi_mapping = _build_kpi_mapping(raw_df, guessed_fields, default_rules, st.session_state.use_sample_data)

    tabs = st.tabs(["Data Preview", "Schema Mapper", "Data Quality", "Drift Alerts", "KPI Explorer", "Export"])

    with tabs[0]:
        st.subheader("Data Preview")
        st.write("Source rows: {}".format(len(raw_df)))
        st.dataframe(raw_df.head(20), use_container_width=True, hide_index=True)

    with tabs[1]:
        st.subheader("Schema Mapper")
        options = [NONE_OPTION] + raw_df.columns.tolist()
        field_mapping = {}
        for field in ["date", "entity_id", "team", "site", "account", "shift"]:
            selected = st.selectbox(
                FIELD_LABELS[field],
                options=options,
                index=_select_index(options, guessed_fields.get(field)),
                key="field_{}".format(field),
            )
            field_mapping[field] = None if selected == NONE_OPTION else selected

        excluded_columns = {value for value in field_mapping.values() if value}
        available_kpis = [column for column in raw_df.columns if column not in excluded_columns]
        default_selected = [item["source_column"] for item in default_kpi_mapping if item["source_column"] in available_kpis]
        selected_kpis = st.multiselect("KPI columns", options=available_kpis, default=default_selected)

        kpi_mapping = []
        for source_column in selected_kpis:
            defaults = next((item for item in default_kpi_mapping if item["source_column"] == source_column), None)
            defaults = defaults or {
                "source_column": source_column,
                "kpi_name": slugify_column_name(source_column),
                "unit": "",
                "direction_bad": "up",
                "drift_threshold_pct": default_threshold_pct,
                "include": True,
            }
            with st.expander(source_column, expanded=True):
                kpi_name = st.text_input("Canonical KPI name", value=str(defaults["kpi_name"]), key="name_{}".format(source_column))
                unit = st.text_input("Unit", value=str(defaults.get("unit", "")), key="unit_{}".format(source_column))
                direction_bad = st.selectbox(
                    "Bad direction",
                    options=["up", "down"],
                    index=0 if defaults.get("direction_bad", "up") == "up" else 1,
                    key="direction_{}".format(source_column),
                )
                threshold = st.number_input(
                    "Drift threshold %",
                    min_value=1.0,
                    max_value=100.0,
                    value=float(defaults.get("drift_threshold_pct", default_threshold_pct)),
                    step=1.0,
                    key="threshold_{}".format(source_column),
                )
                include = st.checkbox("Include KPI", value=bool(defaults.get("include", True)), key="include_{}".format(source_column))
                kpi_mapping.append(
                    {
                        "source_column": source_column,
                        "kpi_name": slugify_column_name(kpi_name),
                        "unit": unit,
                        "direction_bad": direction_bad,
                        "drift_threshold_pct": threshold,
                        "include": include,
                    }
                )

        mapping_yaml = build_mapping_yaml(field_mapping, kpi_mapping)
        st.code(mapping_yaml, language="yaml")

    if any(field_mapping.get(field) is None for field in REQUIRED_FIELDS) or not any(item.get("include", True) for item in kpi_mapping):
        normalized_df = pd.DataFrame(columns=CANONICAL_COLUMNS + ["raw_kpi_value"])
        quality_df = pd.DataFrame(columns=["status", "check", "details", "affected_rows"])
        alerts = pd.DataFrame()
        active_rules = {}
    else:
        normalized_df = normalize_to_long(raw_df, field_mapping, kpi_mapping, include_raw_value=True)
        active_rules = merge_kpi_rules(default_rules, kpi_mapping, default_threshold_pct=float(default_threshold_pct))
        quality_df = validate_normalized_data(normalized_df, active_rules, raw_row_count=len(raw_df))
        alerts = detect_rolling_drift(normalized_df, active_rules, baseline_window=baseline_window, current_window=current_window)

    with tabs[2]:
        st.subheader("Data Quality")
        if normalized_df.empty:
            st.info("Map the required fields and at least one KPI to normalize the dataset.")
        else:
            st.dataframe(normalized_df[CANONICAL_COLUMNS].head(25), use_container_width=True, hide_index=True)
            st.dataframe(quality_df, use_container_width=True, hide_index=True)

    with tabs[3]:
        st.subheader("Drift Alerts")
        if normalized_df.empty:
            st.info("Normalize the dataset first.")
        else:
            summary = summarize_monitoring(normalized_df, alerts)
            cards = st.columns(5)
            cards[0].metric("Entities monitored", summary["entities_monitored"])
            cards[1].metric("KPIs monitored", summary["kpis_monitored"])
            cards[2].metric("Active alerts", summary["active_alerts"])
            cards[3].metric("High severity alerts", summary["high_severity_alerts"])
            cards[4].metric("Date range", summary["date_range"])

            if alerts.empty:
                st.info("No drift alerts were detected for the current window settings.")
            else:
                filtered_alerts = _filter_alerts(alerts)
                st.dataframe(alerts_for_display(filtered_alerts), use_container_width=True, hide_index=True)

    with tabs[4]:
        st.subheader("KPI Explorer")
        if normalized_df.empty:
            st.info("Normalize data first to explore KPI trends.")
        else:
            entity_options = sorted(normalized_df["entity_id"].dropna().astype(str).unique().tolist())
            kpi_options = sorted(normalized_df["kpi_name"].dropna().astype(str).unique().tolist())
            if not entity_options or not kpi_options:
                st.info("No entity or KPI selections are available.")
            else:
                entity_id = st.selectbox("Entity ID", entity_options, index=0)
                kpi_name = st.selectbox("KPI name", kpi_options, index=0)
                subset = normalized_df[(normalized_df["entity_id"] == entity_id) & (normalized_df["kpi_name"] == kpi_name)]
                if subset.empty:
                    st.info("No data exists for the selected entity and KPI.")
                else:
                    st.plotly_chart(
                        make_kpi_trend_chart(
                            normalized_df,
                            entity_id=entity_id,
                            kpi_name=kpi_name,
                            baseline_window=baseline_window,
                            current_window=current_window,
                        ),
                        use_container_width=True,
                    )

    with tabs[5]:
        st.subheader("Export")
        if normalized_df.empty:
            st.info("Normalize data first to unlock exports.")
        else:
            markdown_report = generate_markdown_report(alerts, normalized_df[CANONICAL_COLUMNS], quality_df)
            st.download_button(
                "Download normalized data",
                data=normalized_df[CANONICAL_COLUMNS].to_csv(index=False).encode("utf-8"),
                file_name="normalized_kpis.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download drift alerts",
                data=alerts.to_csv(index=False).encode("utf-8"),
                file_name="drift_alerts.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download mapping YAML",
                data=mapping_yaml.encode("utf-8"),
                file_name="mapping.yaml",
                mime="text/yaml",
            )
            st.download_button(
                "Download Markdown report",
                data=markdown_report.encode("utf-8"),
                file_name="openbpo_drift_report.md",
                mime="text/markdown",
            )


if __name__ == "__main__":
    main()
