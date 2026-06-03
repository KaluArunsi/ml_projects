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
    build_default_kpi_mapping,
    build_mapping_yaml,
    build_signal_frame,
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
SESSION_DEFAULTS = {
    "raw_df": None,
    "normalized_df": None,
    "field_mapping": {},
    "kpi_mapping": [],
    "kpi_rules": {},
    "quality_df": None,
    "alerts_df": None,
    "selected_alert": None,
    "sample_loaded": False,
    "source_key": None,
    "source_name": None,
    "normalized_mapping_yaml": None,
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          :root {
            --primary-blue: #2563EB;
            --blue-soft: #EFF6FF;
            --danger-red: #EF4444;
            --danger-soft: #FEE2E2;
            --warning-orange: #F97316;
            --warning-soft: #FFEDD5;
            --success-green: #22C55E;
            --success-soft: #DCFCE7;
            --text-dark: #111827;
            --text-muted: #6B7280;
            --border: #E5E7EB;
            --page-bg: #F8FAFC;
            --card-bg: #FFFFFF;
          }

          [data-testid="stAppViewContainer"] {
            background: var(--page-bg);
          }

          [data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid var(--border);
          }

          [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
            border: 1px dashed #93C5FD;
            border-radius: 16px;
            background: #F8FBFF;
          }

          [data-testid="stMetric"] {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
          }

          button[kind="primary"], .stDownloadButton button {
            background: var(--primary-blue);
            border-radius: 12px;
            border: none;
            color: white;
          }

          .stButton button {
            border-radius: 12px;
          }

          [data-baseweb="tab-list"] {
            gap: 8px;
          }

          [data-baseweb="tab"] {
            background: transparent;
            border-radius: 12px 12px 0 0;
            color: var(--text-muted);
            font-weight: 600;
            padding: 12px 18px;
          }

          [aria-selected="true"] {
            color: var(--primary-blue) !important;
            box-shadow: inset 0 -3px 0 0 var(--primary-blue);
          }

          .dashboard-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
          }

          .dashboard-card + .dashboard-card {
            margin-top: 16px;
          }

          .eyebrow {
            display: inline-block;
            font-size: 0.75rem;
            line-height: 1;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 10px;
          }

          .metric-title {
            color: var(--text-muted);
            font-size: 0.88rem;
            margin-bottom: 4px;
          }

          .metric-value {
            color: var(--text-dark);
            font-size: 1.8rem;
            font-weight: 700;
            line-height: 1.1;
          }

          .metric-subtitle {
            color: var(--text-muted);
            font-size: 0.88rem;
            margin-top: 6px;
          }

          .header-card {
            background: linear-gradient(180deg, #EFF6FF 0%, #F8FBFF 100%);
            border: 1px solid #BFDBFE;
            border-radius: 18px;
            padding: 18px 20px;
            color: var(--primary-blue);
          }

          .section-title {
            color: var(--text-dark);
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 10px;
          }

          .helper-text {
            color: var(--text-muted);
            font-size: 0.88rem;
          }

          .badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
          }

          .badge-high {
            background: var(--danger-soft);
            color: var(--danger-red);
          }

          .badge-medium {
            background: var(--warning-soft);
            color: var(--warning-orange);
          }

          .badge-watch {
            background: var(--blue-soft);
            color: var(--primary-blue);
          }

          .footer-note {
            color: var(--text-muted);
            font-size: 0.88rem;
            text-align: center;
            padding: 20px 0 8px 0;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def ensure_sample_data() -> None:
    if not SAMPLE_DATA_PATH.exists():
        generate_sample_bpo_kpis(SAMPLE_DATA_PATH)


def load_sample_mapping() -> dict:
    return yaml.safe_load(SAMPLE_MAPPING_PATH.read_text())


def reset_app_state() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith(("field_", "kpi_", "filter_", "explorer_", "chart_", "alert_")):
            del st.session_state[key]
    for key, value in SESSION_DEFAULTS.items():
        st.session_state[key] = value


def set_mapping_widget_defaults(field_mapping: dict, kpi_mapping: list[dict]) -> None:
    for field in ["date", "entity_id", "team", "site", "account", "shift"]:
        value = field_mapping.get(field)
        st.session_state["field_{}".format(field)] = value if value is not None else NONE_OPTION

    selected_kpis = [item["source_column"] for item in kpi_mapping if item.get("include", True)]
    st.session_state["kpi_selected_columns"] = selected_kpis
    for item in kpi_mapping:
        source = item["source_column"]
        st.session_state["kpi_name_{}".format(source)] = item["kpi_name"]
        st.session_state["kpi_unit_{}".format(source)] = item.get("unit", "")
        st.session_state["kpi_direction_{}".format(source)] = item.get("direction_bad", "up")
        st.session_state["kpi_threshold_{}".format(source)] = float(item.get("drift_threshold_pct", DEFAULT_THRESHOLD_PCT))
        st.session_state["kpi_include_{}".format(source)] = bool(item.get("include", True))


def store_source_frame(raw_df: pd.DataFrame, source_key, source_name: str, sample_loaded: bool, default_rules: dict) -> None:
    st.session_state.raw_df = raw_df
    st.session_state.source_key = source_key
    st.session_state.source_name = source_name
    st.session_state.sample_loaded = sample_loaded
    st.session_state.normalized_df = None
    st.session_state.quality_df = None
    st.session_state.kpi_rules = {}
    st.session_state.alerts_df = None
    st.session_state.selected_alert = None
    st.session_state.normalized_mapping_yaml = None

    if sample_loaded:
        sample_mapping = load_sample_mapping()
        field_mapping = sample_mapping["field_mapping"]
        kpi_mapping = sample_mapping["kpis"]
    else:
        field_mapping = guess_field_mapping(raw_df.columns)
        candidates = infer_kpi_candidates(raw_df.columns, field_mapping)
        kpi_mapping = build_default_kpi_mapping(raw_df.columns.tolist(), candidates, default_rules, DEFAULT_THRESHOLD_PCT)

    st.session_state.field_mapping = field_mapping
    st.session_state.kpi_mapping = kpi_mapping
    set_mapping_widget_defaults(field_mapping, kpi_mapping)


def load_source_if_needed(uploaded_file, sheet_name, default_rules: dict) -> None:
    if uploaded_file is not None:
        source_key = ("upload", uploaded_file.name, uploaded_file.size, sheet_name)
        if st.session_state.source_key != source_key:
            raw_df = load_tabular_file(uploaded_file, sheet_name=sheet_name, filename=uploaded_file.name)
            store_source_frame(raw_df, source_key, uploaded_file.name, False, default_rules)
        return

    source_key = ("sample", SAMPLE_DATA_PATH.name)
    if st.session_state.raw_df is None or st.session_state.source_key != source_key:
        raw_df = load_tabular_file(SAMPLE_DATA_PATH)
        store_source_frame(raw_df, source_key, SAMPLE_DATA_PATH.name, True, default_rules)


def build_current_mapping(raw_df: pd.DataFrame, default_threshold_pct: float) -> tuple[dict, list[dict], str]:
    field_mapping = {}
    for field in ["date", "entity_id", "team", "site", "account", "shift"]:
        selected = st.session_state.get("field_{}".format(field), NONE_OPTION)
        field_mapping[field] = None if selected == NONE_OPTION else selected

    available_kpis = st.session_state.get("kpi_selected_columns", [])
    kpi_mapping = []
    for source_column in available_kpis:
        if source_column not in raw_df.columns:
            continue
        kpi_mapping.append(
            {
                "source_column": source_column,
                "kpi_name": slugify_column_name(st.session_state.get("kpi_name_{}".format(source_column), source_column)),
                "unit": st.session_state.get("kpi_unit_{}".format(source_column), ""),
                "direction_bad": st.session_state.get("kpi_direction_{}".format(source_column), "up"),
                "drift_threshold_pct": float(st.session_state.get("kpi_threshold_{}".format(source_column), default_threshold_pct)),
                "include": bool(st.session_state.get("kpi_include_{}".format(source_column), True)),
            }
        )
    return field_mapping, kpi_mapping, build_mapping_yaml(field_mapping, kpi_mapping)


def normalize_current_state(field_mapping: dict, kpi_mapping: list[dict], mapping_yaml: str, default_rules: dict, default_threshold_pct: float) -> None:
    normalized_df = normalize_to_long(st.session_state.raw_df, field_mapping, kpi_mapping, include_raw_value=True)
    kpi_rules = merge_kpi_rules(default_rules, kpi_mapping, default_threshold_pct=default_threshold_pct)
    quality_df = validate_normalized_data(normalized_df, kpi_rules, raw_row_count=len(st.session_state.raw_df))

    st.session_state.field_mapping = field_mapping
    st.session_state.kpi_mapping = kpi_mapping
    st.session_state.kpi_rules = kpi_rules
    st.session_state.normalized_df = normalized_df
    st.session_state.quality_df = quality_df
    st.session_state.normalized_mapping_yaml = mapping_yaml


def refresh_alerts_state(baseline_window: int, current_window: int) -> None:
    if st.session_state.normalized_df is None or st.session_state.normalized_df.empty:
        st.session_state.alerts_df = pd.DataFrame()
        return
    st.session_state.alerts_df = detect_rolling_drift(
        st.session_state.normalized_df,
        st.session_state.kpi_rules,
        baseline_window=baseline_window,
        current_window=current_window,
    )


def render_header() -> None:
    left, right = st.columns([3, 2])
    with left:
        st.markdown("## OpenBPO Drift")
        st.markdown(
            "<div class='helper-text'>Local-first KPI drift monitoring for BPO and contact center teams.</div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="header-card">
              <div class="eyebrow">Local Processing</div>
              <div style="font-size:1rem;font-weight:700;">All processing happens locally.</div>
              <div style="margin-top:6px;font-size:0.92rem;">Your data never leaves your machine.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_metric_card(title: str, value: str, subtitle: str) -> None:
    st.markdown(
        """
        <div class="dashboard-card">
          <div class="eyebrow">{title}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-subtitle">{subtitle}</div>
        </div>
        """.format(title=title, value=value, subtitle=subtitle),
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, message: str) -> None:
    st.markdown(
        """
        <div class="dashboard-card" style="padding:28px 24px;">
          <div class="section-title">{title}</div>
          <div class="helper-text">{message}</div>
        </div>
        """.format(title=title, message=message),
        unsafe_allow_html=True,
    )


def style_quality_table(quality_df: pd.DataFrame):
    color_map = {
        "Pass": ("#DCFCE7", "#166534"),
        "Warning": ("#FFEDD5", "#C2410C"),
        "Fail": ("#FEE2E2", "#B91C1C"),
        "Info": ("#E5E7EB", "#374151"),
    }

    def status_style(value):
        background, color = color_map.get(value, ("#FFFFFF", "#111827"))
        return "background-color: {}; color: {}; font-weight: 700".format(background, color)

    return quality_df.style.map(status_style, subset=["status"])


def style_alert_table(alerts_df: pd.DataFrame):
    def severity_style(value):
        if value == "High":
            return "background-color: #FEE2E2; color: #B91C1C; font-weight: 700"
        if value == "Medium":
            return "background-color: #FFEDD5; color: #C2410C; font-weight: 700"
        return "background-color: #DBEAFE; color: #1D4ED8; font-weight: 700"

    def drift_style(value):
        color = "#B91C1C" if float(value) < 0 else "#1D4ED8"
        return "color: {}; font-weight: 700".format(color)

    return (
        alerts_for_display(alerts_df)
        .style.format(
            {
                "Baseline": "{:.2f}",
                "Current": "{:.2f}",
                "Drift %": "{:.1f}%",
                "Threshold %": "{:.1f}%",
            }
        )
        .map(severity_style, subset=["Severity"])
        .map(drift_style, subset=["Drift %"])
    )


def filter_alerts(alerts_df: pd.DataFrame) -> pd.DataFrame:
    filtered = alerts_df.copy()
    if filtered.empty:
        return filtered

    severity = st.session_state.get("filter_severity", "All")
    kpi = st.session_state.get("filter_kpi", "All")
    team = st.session_state.get("filter_team", "All")
    account = st.session_state.get("filter_account", "All")

    if severity != "All":
        filtered = filtered[filtered["severity"] == severity]
    if kpi != "All":
        filtered = filtered[filtered["kpi_name"] == kpi]
    if team != "All":
        filtered = filtered[filtered["team"].astype(str) == team]
    if account != "All":
        filtered = filtered[filtered["account"].astype(str) == account]
    return filtered


def alert_label(row: pd.Series) -> str:
    return "{severity} | {entity_id} | {kpi_name} | {drift_pct:.1f}%".format(**row.to_dict())


def main() -> None:
    st.set_page_config(page_title="OpenBPO Drift", page_icon="📈", layout="wide")
    inject_styles()
    init_session_state()
    ensure_sample_data()
    default_rules = load_kpi_rules(DEFAULT_RULES_PATH)

    st.sidebar.markdown("### 1. Load Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV / XLSX / XLS", type=["csv", "xlsx", "xls"])
    load_sample = st.sidebar.button("Load Sample Data", width="stretch")
    st.sidebar.caption("Load the included sample KPI dataset")

    if load_sample:
        raw_df = load_tabular_file(SAMPLE_DATA_PATH)
        store_source_frame(raw_df, ("sample", SAMPLE_DATA_PATH.name), SAMPLE_DATA_PATH.name, True, default_rules)

    sheet_name: str | int = 0
    if uploaded_file is not None and Path(uploaded_file.name).suffix.lower() in {".xlsx", ".xls"}:
        st.sidebar.markdown("### Sheet")
        sheet_options = list(list_excel_sheets(uploaded_file))
        sheet_name = st.sidebar.selectbox("Select worksheet", sheet_options, index=0)

    load_source_if_needed(uploaded_file, sheet_name, default_rules)

    st.sidebar.markdown("### 2. Analysis Windows")
    baseline_window = st.sidebar.number_input("Baseline window (days)", min_value=7, max_value=60, value=DEFAULT_BASELINE_WINDOW, step=1)
    current_window = st.sidebar.number_input("Current window (days)", min_value=3, max_value=30, value=DEFAULT_CURRENT_WINDOW, step=1)

    st.sidebar.markdown("### 3. Drift Settings")
    default_threshold_pct = st.sidebar.number_input("Default drift threshold (%)", min_value=1.0, max_value=100.0, value=DEFAULT_THRESHOLD_PCT, step=1.0)

    st.sidebar.markdown("### 4. Export")
    if st.session_state.normalized_df is not None and not st.session_state.normalized_df.empty:
        st.sidebar.success("Exports are ready in the Export tab")
    else:
        st.sidebar.caption("Exports will appear in the Export tab after normalization")
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset App", width="stretch"):
        reset_app_state()
        st.rerun()
    st.sidebar.caption("OpenBPO Drift v1.0.0")

    render_header()

    if st.session_state.raw_df is None or st.session_state.raw_df.empty:
        render_empty_state("Welcome", "Upload a CSV or Excel file, or load the sample dataset to begin.")
        return

    raw_df = st.session_state.raw_df
    source_field_mapping = st.session_state.field_mapping or guess_field_mapping(raw_df.columns)
    source_kpi_defaults = st.session_state.kpi_mapping or build_default_kpi_mapping(
        raw_df.columns.tolist(),
        infer_kpi_candidates(raw_df.columns, source_field_mapping),
        default_rules,
        default_threshold_pct,
    )

    tabs = st.tabs(["Data Preview", "Schema Mapper", "Data Quality", "Drift Alerts", "KPI Explorer", "Export"])

    with tabs[0]:
        st.markdown("<div class='section-title'>Data Preview</div>", unsafe_allow_html=True)
        date_range = "Unavailable"
        if "date" in raw_df.columns:
            parsed_dates = pd.to_datetime(raw_df["date"], errors="coerce")
            if parsed_dates.notna().any():
                date_range = "{} to {}".format(parsed_dates.min().strftime("%Y-%m-%d"), parsed_dates.max().strftime("%Y-%m-%d"))
        preview_cards = st.columns(4)
        with preview_cards[0]:
            render_metric_card("Rows Loaded", "{:,}".format(len(raw_df)), "raw rows")
        with preview_cards[1]:
            render_metric_card("Columns", str(len(raw_df.columns)), "detected columns")
        with preview_cards[2]:
            render_metric_card("Date Range", date_range, "source coverage")
        with preview_cards[3]:
            render_metric_card("Source", st.session_state.source_name or "unknown", "current file")
        st.markdown(
            "<div class='dashboard-card'><div class='section-title'>Detected Columns</div><div class='helper-text'>{}</div></div>".format(
                " ".join("<code>{}</code>".format(column) for column in raw_df.columns)
            ),
            unsafe_allow_html=True,
        )
        st.dataframe(raw_df.head(20), width="stretch", hide_index=True)

    with tabs[1]:
        st.markdown("<div class='section-title'>Schema Mapper</div>", unsafe_allow_html=True)
        options = [NONE_OPTION] + raw_df.columns.tolist()

        required_col, optional_col = st.columns(2)
        with required_col:
            with st.container(border=True):
                st.markdown("<div class='section-title'>Required Fields</div>", unsafe_allow_html=True)
                for field in ["date", "entity_id"]:
                    st.selectbox(FIELD_LABELS[field], options, key="field_{}".format(field))
        with optional_col:
            with st.container(border=True):
                st.markdown("<div class='section-title'>Optional Metadata</div>", unsafe_allow_html=True)
                for field in ["team", "site", "account", "shift"]:
                    st.selectbox(FIELD_LABELS[field], options, key="field_{}".format(field))

        selected_fields = {field: st.session_state.get("field_{}".format(field), NONE_OPTION) for field in ["date", "entity_id", "team", "site", "account", "shift"]}
        excluded_columns = {value for value in selected_fields.values() if value != NONE_OPTION}
        available_kpis = [column for column in raw_df.columns if column not in excluded_columns]
        default_selected_kpis = [item["source_column"] for item in source_kpi_defaults if item["source_column"] in available_kpis]
        if not st.session_state.get("kpi_selected_columns"):
            st.session_state["kpi_selected_columns"] = default_selected_kpis

        with st.container(border=True):
            st.markdown("<div class='section-title'>KPI Columns</div>", unsafe_allow_html=True)
            st.multiselect("Select KPI columns", options=available_kpis, key="kpi_selected_columns")
            default_lookup = {item["source_column"]: item for item in source_kpi_defaults}
            for source_column in st.session_state.get("kpi_selected_columns", []):
                defaults = default_lookup.get(
                    source_column,
                    {
                        "source_column": source_column,
                        "kpi_name": source_column.lower(),
                        "unit": "",
                        "direction_bad": "up",
                        "drift_threshold_pct": default_threshold_pct,
                        "include": True,
                    },
                )
                st.session_state.setdefault("kpi_name_{}".format(source_column), defaults["kpi_name"])
                st.session_state.setdefault("kpi_unit_{}".format(source_column), defaults.get("unit", ""))
                st.session_state.setdefault("kpi_direction_{}".format(source_column), defaults.get("direction_bad", "up"))
                st.session_state.setdefault("kpi_threshold_{}".format(source_column), float(defaults.get("drift_threshold_pct", default_threshold_pct)))
                st.session_state.setdefault("kpi_include_{}".format(source_column), bool(defaults.get("include", True)))
                with st.expander(source_column, expanded=True):
                    st.text_input("KPI name", key="kpi_name_{}".format(source_column))
                    st.text_input("Unit", key="kpi_unit_{}".format(source_column))
                    st.selectbox("Bad direction", ["up", "down"], key="kpi_direction_{}".format(source_column))
                    st.number_input("Drift threshold (%)", min_value=1.0, max_value=100.0, step=1.0, key="kpi_threshold_{}".format(source_column))
                    st.checkbox("Include", key="kpi_include_{}".format(source_column))

        field_mapping, kpi_mapping, mapping_yaml = build_current_mapping(raw_df, default_threshold_pct)
        action_cols = st.columns([1, 1, 3])
        with action_cols[0]:
            normalize_clicked = st.button("Normalize Data", type="primary", width="stretch")
        with action_cols[1]:
            st.download_button(
                "Export Mapping YAML",
                data=mapping_yaml.encode("utf-8"),
                file_name="mapping.yaml",
                mime="text/yaml",
                width="stretch",
            )

        st.code(mapping_yaml, language="yaml")
        if normalize_clicked:
            if any(field_mapping.get(field) is None for field in REQUIRED_FIELDS):
                st.error("Date column and Entity ID column are required.")
            elif not any(item.get("include", True) for item in kpi_mapping):
                st.error("Select at least one KPI column before normalizing.")
            else:
                normalize_current_state(field_mapping, kpi_mapping, mapping_yaml, default_rules, float(default_threshold_pct))
                st.success("Data normalized and validation refreshed.")

        if st.session_state.sample_loaded and st.session_state.normalized_df is None:
            normalize_current_state(field_mapping, kpi_mapping, mapping_yaml, default_rules, float(default_threshold_pct))

    mapping_is_stale = (
        st.session_state.normalized_mapping_yaml is not None
        and st.session_state.normalized_mapping_yaml != mapping_yaml
    )

    if st.session_state.normalized_df is not None and not mapping_is_stale:
        refresh_alerts_state(int(baseline_window), int(current_window))

    normalized_df = st.session_state.normalized_df
    quality_df = st.session_state.quality_df if st.session_state.quality_df is not None else pd.DataFrame(columns=["status", "check", "details", "affected_rows"])
    alerts_df = st.session_state.alerts_df if st.session_state.alerts_df is not None else pd.DataFrame()

    with tabs[2]:
        st.markdown("<div class='section-title'>Data Quality</div>", unsafe_allow_html=True)
        if normalized_df is None:
            render_empty_state("Normalize Data First", "Normalize your data first to view validation results.")
        else:
            warnings_count = int((quality_df["status"] == "Warning").sum()) if not quality_df.empty else 0
            failures_count = int((quality_df["status"] == "Fail").sum()) if not quality_df.empty else 0
            cards = st.columns(4)
            with cards[0]:
                render_metric_card("Rows Loaded", "{:,}".format(len(raw_df)), "raw rows")
            with cards[1]:
                render_metric_card("Normalized Observations", "{:,}".format(len(normalized_df)), "long-format rows")
            with cards[2]:
                render_metric_card("Validation Warnings", str(warnings_count), "warning checks")
            with cards[3]:
                render_metric_card("Validation Failures", str(failures_count), "fail checks")
            if mapping_is_stale:
                st.warning("Mapping changes are pending. Click Normalize Data to refresh these results.")
            st.dataframe(style_quality_table(quality_df), width="stretch", hide_index=True)
            st.dataframe(normalized_df[CANONICAL_COLUMNS].head(25), width="stretch", hide_index=True)

    with tabs[3]:
        st.markdown("<div class='section-title'>Drift Alerts</div>", unsafe_allow_html=True)
        if normalized_df is None:
            render_empty_state("Normalize Data First", "Normalize your data first to view drift alerts.")
        else:
            if mapping_is_stale:
                st.warning("Mapping or KPI rule changes are pending. Click Normalize Data to refresh drift outputs.")
            summary = summarize_monitoring(normalized_df, alerts_df)
            cards = st.columns(5)
            with cards[0]:
                render_metric_card("Entities Monitored", str(summary["entities_monitored"]), "agent")
            with cards[1]:
                render_metric_card("KPIs Monitored", str(summary["kpis_monitored"]), "metrics")
            with cards[2]:
                render_metric_card("Active Alerts", str(summary["active_alerts"]), "drifting now")
            with cards[3]:
                render_metric_card("High Severity Alerts", str(summary["high_severity_alerts"]), "needs attention")
            with cards[4]:
                render_metric_card("Date Range", summary["date_range"], "{} days".format(summary["date_days"]))

            if alerts_df.empty:
                render_empty_state("No Active Drift Alerts", "No entities currently breach the configured drift thresholds for the selected analysis windows.")
            else:
                with st.container(border=True):
                    st.markdown("<div class='section-title'>Filters</div>", unsafe_allow_html=True)
                    filter_cols = st.columns([1, 1, 1, 1, 0.8])
                    with filter_cols[0]:
                        severity_options = ["All"] + sorted(alerts_df["severity"].unique().tolist(), key=lambda value: {"High": 0, "Medium": 1, "Watch": 2}.get(value, 3))
                        st.selectbox("Filter by severity", severity_options, key="filter_severity")
                    with filter_cols[1]:
                        st.selectbox("Filter by KPI", ["All"] + sorted(alerts_df["kpi_name"].dropna().unique().tolist()), key="filter_kpi")
                    with filter_cols[2]:
                        st.selectbox("Filter by team", ["All"] + sorted(alerts_df["team"].dropna().astype(str).unique().tolist()), key="filter_team")
                    with filter_cols[3]:
                        st.selectbox("Filter by account", ["All"] + sorted(alerts_df["account"].dropna().astype(str).unique().tolist()), key="filter_account")
                    with filter_cols[4]:
                        st.write("")
                        st.write("")
                        if st.button("Reset Filters", width="stretch"):
                            for key in ["filter_severity", "filter_kpi", "filter_team", "filter_account"]:
                                st.session_state[key] = "All"
                            st.rerun()

                filtered_alerts = filter_alerts(alerts_df)
                st.markdown("#### Drift Alerts ({})".format(len(filtered_alerts)))
                st.dataframe(style_alert_table(filtered_alerts), width="stretch", hide_index=True)
                st.caption("Showing {} of {} alerts. Page 1 of 1.".format(len(filtered_alerts), len(alerts_df)))

                if not filtered_alerts.empty:
                    labels = [alert_label(row) for _, row in filtered_alerts.iterrows()]
                    if st.session_state.selected_alert not in labels:
                        st.session_state.selected_alert = labels[0]
                    selected_label = st.selectbox(
                        "Inspect alert",
                        labels,
                        index=labels.index(st.session_state.selected_alert) if st.session_state.selected_alert in labels else 0,
                        key="alert_selected_label",
                    )
                    st.session_state.selected_alert = selected_label
                    selected_alert = filtered_alerts.iloc[labels.index(selected_label)]

                    chart_entity_default = selected_alert["entity_id"]
                    chart_kpi_default = selected_alert["kpi_name"]
                    if st.session_state.get("chart_entity") not in normalized_df["entity_id"].dropna().astype(str).unique().tolist():
                        st.session_state["chart_entity"] = chart_entity_default
                    if st.session_state.get("chart_kpi") not in normalized_df["kpi_name"].dropna().astype(str).unique().tolist():
                        st.session_state["chart_kpi"] = chart_kpi_default

                    lower, right = st.columns([1.4, 1])
                    with lower:
                        with st.container(border=True):
                            st.markdown("<div class='section-title'>KPI Trend Preview</div>", unsafe_allow_html=True)
                            chart_controls = st.columns(2)
                            with chart_controls[0]:
                                st.selectbox("Entity", sorted(normalized_df["entity_id"].dropna().astype(str).unique().tolist()), key="chart_entity")
                            with chart_controls[1]:
                                st.selectbox("KPI", sorted(normalized_df["kpi_name"].dropna().astype(str).unique().tolist()), key="chart_kpi")
                            st.plotly_chart(
                                make_kpi_trend_chart(
                                    normalized_df,
                                    entity_id=st.session_state["chart_entity"],
                                    kpi_name=st.session_state["chart_kpi"],
                                    baseline_window=int(baseline_window),
                                    current_window=int(current_window),
                                ),
                                width="stretch",
                            )
                    with right:
                        badge_class = "badge-high" if selected_alert["severity"] == "High" else "badge-medium" if selected_alert["severity"] == "Medium" else "badge-watch"
                        st.markdown(
                            """
                            <div class="dashboard-card">
                              <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">
                                <div class="section-title" style="margin-bottom:0;">Alert Explanation</div>
                                <span class="badge {badge_class}">{severity}</span>
                              </div>
                              <div style="margin-top:14px;color:#111827;line-height:1.65;">{explanation}</div>
                              <div style="margin-top:18px;display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                                <div class="dashboard-card" style="padding:12px 14px;">
                                  <div class="metric-title">Baseline ({baseline_window}d)</div>
                                  <div style="font-size:1.1rem;font-weight:700;">{baseline:.2f}</div>
                                </div>
                                <div class="dashboard-card" style="padding:12px 14px;">
                                  <div class="metric-title">Current ({current_window}d)</div>
                                  <div style="font-size:1.1rem;font-weight:700;">{current:.2f}</div>
                                </div>
                                <div class="dashboard-card" style="padding:12px 14px;">
                                  <div class="metric-title">Drift</div>
                                  <div style="font-size:1.1rem;font-weight:700;">{drift_pct:.1f}%</div>
                                </div>
                                <div class="dashboard-card" style="padding:12px 14px;">
                                  <div class="metric-title">Threshold</div>
                                  <div style="font-size:1.1rem;font-weight:700;">{threshold_pct:.1f}% {direction_bad}</div>
                                </div>
                              </div>
                            </div>
                            """.format(
                                badge_class=badge_class,
                                severity=selected_alert["severity"],
                                explanation=selected_alert["explanation"],
                                baseline_window=int(baseline_window),
                                current_window=int(current_window),
                                baseline=selected_alert["baseline"],
                                current=selected_alert["current"],
                                drift_pct=selected_alert["drift_pct"],
                                threshold_pct=selected_alert["threshold_pct"],
                                direction_bad=selected_alert["direction_bad"],
                            ),
                            unsafe_allow_html=True,
                        )

    with tabs[4]:
        st.markdown("<div class='section-title'>KPI Explorer</div>", unsafe_allow_html=True)
        if normalized_df is None:
            render_empty_state("Normalize Data First", "Normalize your data first to explore KPI trends.")
        else:
            explorer_filters = normalized_df.copy()
            controls = st.columns(4)
            team_options = ["All"] + sorted(explorer_filters["team"].dropna().astype(str).unique().tolist())
            account_options = ["All"] + sorted(explorer_filters["account"].dropna().astype(str).unique().tolist())
            with controls[0]:
                st.selectbox("Team filter", team_options, key="explorer_team")
            with controls[1]:
                st.selectbox("Account filter", account_options, key="explorer_account")

            if st.session_state.get("explorer_team") not in {None, "All"}:
                explorer_filters = explorer_filters[explorer_filters["team"].astype(str) == st.session_state["explorer_team"]]
            if st.session_state.get("explorer_account") not in {None, "All"}:
                explorer_filters = explorer_filters[explorer_filters["account"].astype(str) == st.session_state["explorer_account"]]

            entity_options = sorted(explorer_filters["entity_id"].dropna().astype(str).unique().tolist())
            kpi_options = sorted(explorer_filters["kpi_name"].dropna().astype(str).unique().tolist())
            if not entity_options or not kpi_options:
                render_empty_state("No Matching Data", "The current KPI explorer filters do not match any normalized observations.")
            else:
                if st.session_state.get("explorer_entity") not in entity_options:
                    st.session_state["explorer_entity"] = entity_options[0]
                if st.session_state.get("explorer_kpi") not in kpi_options:
                    st.session_state["explorer_kpi"] = kpi_options[0]
                with controls[2]:
                    st.selectbox("Entity", entity_options, key="explorer_entity")
                with controls[3]:
                    st.selectbox("KPI", kpi_options, key="explorer_kpi")

                explorer_chart = build_signal_frame(
                    normalized_df,
                    entity_id=st.session_state["explorer_entity"],
                    kpi_name=st.session_state["explorer_kpi"],
                    baseline_window=int(baseline_window),
                    current_window=int(current_window),
                )
                latest_value = float(explorer_chart["kpi_value"].iloc[-1])
                baseline_series = explorer_chart["baseline_mean"].dropna()
                baseline_value = float(baseline_series.iloc[-1]) if not baseline_series.empty else latest_value
                current_series = explorer_chart.loc[explorer_chart["is_current_window"], "kpi_value"]
                current_value = float(current_series.mean()) if not current_series.empty else latest_value
                drift_pct = ((current_value - baseline_value) / baseline_value) * 100 if baseline_value else 0.0

                stat_cards = st.columns(4)
                with stat_cards[0]:
                    render_metric_card("Latest Value", "{:.2f}".format(latest_value), "most recent")
                with stat_cards[1]:
                    render_metric_card("Baseline", "{:.2f}".format(baseline_value), "{} day mean".format(int(baseline_window)))
                with stat_cards[2]:
                    render_metric_card("Current Value", "{:.2f}".format(current_value), "{} day mean".format(int(current_window)))
                with stat_cards[3]:
                    render_metric_card("Drift Percent", "{:.1f}%".format(drift_pct), "current vs baseline")

                st.plotly_chart(
                    make_kpi_trend_chart(
                        normalized_df,
                        entity_id=st.session_state["explorer_entity"],
                        kpi_name=st.session_state["explorer_kpi"],
                        baseline_window=int(baseline_window),
                        current_window=int(current_window),
                    ),
                    width="stretch",
                )

    with tabs[5]:
        st.markdown("<div class='section-title'>Export</div>", unsafe_allow_html=True)
        if normalized_df is None:
            render_empty_state("Normalize Data First", "Normalize your data first to unlock exports.")
        else:
            if mapping_is_stale:
                st.warning("The current mapping has changed since the last normalization. Exported results still reflect the last normalized state.")
            markdown_report = generate_markdown_report(alerts_df, normalized_df[CANONICAL_COLUMNS], quality_df)
            export_cards = st.columns(2)
            with export_cards[0]:
                st.markdown("<div class='dashboard-card'><div class='section-title'>Normalized KPI Data</div><div class='helper-text'>Export the canonical long-format KPI dataset used for drift detection.</div></div>", unsafe_allow_html=True)
                st.download_button(
                    "Download Normalized KPI Data",
                    data=normalized_df[CANONICAL_COLUMNS].to_csv(index=False).encode("utf-8"),
                    file_name="normalized_kpis.csv",
                    mime="text/csv",
                    width="stretch",
                )
                st.markdown("<div class='dashboard-card'><div class='section-title'>Mapping YAML</div><div class='helper-text'>Export the current schema and KPI mapping for reuse on similar files.</div></div>", unsafe_allow_html=True)
                st.download_button(
                    "Download Mapping YAML",
                    data=mapping_yaml.encode("utf-8"),
                    file_name="mapping.yaml",
                    mime="text/yaml",
                    width="stretch",
                )
            with export_cards[1]:
                st.markdown("<div class='dashboard-card'><div class='section-title'>Drift Alerts</div><div class='helper-text'>Export the current drift alert table with explanations and thresholds.</div></div>", unsafe_allow_html=True)
                st.download_button(
                    "Download Drift Alerts",
                    data=alerts_df.to_csv(index=False).encode("utf-8"),
                    file_name="drift_alerts.csv",
                    mime="text/csv",
                    width="stretch",
                )
                st.markdown("<div class='dashboard-card'><div class='section-title'>Markdown Report</div><div class='helper-text'>Generate a portable local report summarizing quality checks and top alerts.</div></div>", unsafe_allow_html=True)
                st.download_button(
                    "Download Markdown Report",
                    data=markdown_report.encode("utf-8"),
                    file_name="openbpo_drift_report.md",
                    mime="text/markdown",
                    width="stretch",
                )

    st.markdown(
        "<div class='footer-note'>OpenBPO Drift runs locally. No external services, no telemetry, and no data leaves your machine.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
