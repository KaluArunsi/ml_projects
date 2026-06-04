# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from html import escape
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
    SSA_N8NN_SOURCE_URL,
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
QUALITY_LABELS = {
    "row_count": "Row count",
    "date_parse": "Date parse check",
    "entity_id_missing": "Missing entity IDs",
    "kpi_value_numeric": "Numeric KPI values",
    "missing_kpi_values": "Missing KPI values",
    "duplicate_observations": "Duplicate observations",
    "min_date_max_date": "Date range check",
    "entities_count": "Unique entities",
    "kpis_count": "Unique KPIs",
    "kpi_range": "KPI range check",
}
UNIT_LABELS = {
    "percentage": "%",
    "seconds": "sec",
    "score": "score",
    "rate": "rate",
    "count": "count",
    "number": "number",
}
UNIT_OPTIONS = ["percentage", "seconds", "score", "rate", "count", "number"]
ACCENT_STYLES = {
    "blue": {"soft": "#EEF4FF", "line": "#2454D6", "text": "#2454D6"},
    "green": {"soft": "#ECF8F4", "line": "#17745B", "text": "#17745B"},
    "amber": {"soft": "#FFF6E8", "line": "#B36B00", "text": "#B36B00"},
    "red": {"soft": "#FFF1F1", "line": "#C83737", "text": "#C83737"},
    "purple": {"soft": "#F2F0FF", "line": "#6857C8", "text": "#6857C8"},
}
SESSION_DEFAULTS = {
    "raw_df": None,
    "normalized_df": None,
    "field_mapping": {},
    "kpi_mapping": [],
    "kpi_editor_rows": [],
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
            --ink: #141820;
            --muted: #687182;
            --line: #D9DEE8;
            --panel: #FFFFFF;
            --canvas: #F4F5F7;
            --accent: #244A9B;
            --accent-soft: #EEF4FF;
            --good: #176B55;
            --good-soft: #ECF8F4;
            --warn: #A15C00;
            --warn-soft: #FFF6E8;
            --bad: #B4232A;
            --bad-soft: #FFF1F1;
            --violet: #6857C8;
            --violet-soft: #F2F0FF;
          }

          [data-testid="stDecoration"] {
            display: none !important;
          }

          [data-testid="stHeader"] {
            background: transparent;
          }

          [data-testid="stToolbar"] {
            background: transparent;
          }

          [data-testid="stAppViewContainer"] {
            background: var(--canvas);
          }

          [data-testid="stMainBlockContainer"] {
            max-width: 1560px;
            padding-top: 1.25rem;
            padding-bottom: 2rem;
          }

          [data-testid="stSidebar"] {
            background: #F8F9FB;
            border-right: 1px solid var(--line);
          }

          [data-testid="stSidebarContent"] {
            padding: 1.15rem 1rem 1.4rem;
          }

          [data-testid="stSidebar"] hr {
            margin: 1.4rem 0;
          }

          [data-testid="stFileUploader"] section {
            border-radius: 8px !important;
            border-color: #BAC3D2 !important;
            background: #FFFFFF !important;
          }

          [data-baseweb="tab-list"] {
            gap: 12px;
            border-bottom: 1px solid var(--line);
            margin: 0.45rem 0 1.15rem;
          }

          [data-baseweb="tab"] {
            padding: 0.7rem 0.25rem 0.85rem;
            color: var(--muted);
            font-weight: 600;
          }

          [aria-selected="true"] {
            color: var(--accent) !important;
            box-shadow: inset 0 -2px 0 0 var(--accent);
          }

          [data-testid="stMetric"] {
            background: #FFFFFF;
            border-color: var(--line) !important;
            box-shadow: 0 1px 2px rgba(20, 24, 32, 0.04);
          }

          [data-testid="stMetric"] label p {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            text-transform: uppercase;
          }

          .app-header {
            display: grid;
            grid-template-columns: minmax(0, 1fr) 360px;
            gap: 1.25rem;
            align-items: start;
            margin-bottom: 0.35rem;
          }

          .app-kicker {
            color: #475467;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
          }

          .app-title {
            color: var(--ink);
            font-size: clamp(2rem, 1.5rem + 1vw, 2.75rem);
            font-weight: 800;
            line-height: 1.05;
          }

          .app-subtitle {
            color: var(--muted);
            font-size: 0.98rem;
            margin-top: 0.45rem;
            max-width: 760px;
          }

          .status-panel {
            background: #FFFFFF;
            border: 1px solid var(--line);
            border-left: 3px solid var(--good);
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
          }

          .status-panel strong {
            color: var(--ink);
          }

          .status-panel span {
            display: block;
            color: var(--muted);
            font-size: 0.86rem;
            margin-top: 0.25rem;
          }

          .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            margin-bottom: 1.6rem;
          }

          .sidebar-brand__mark {
            width: 34px;
            height: 34px;
            border-radius: 8px;
            background: var(--ink);
            color: #FFFFFF;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.78rem;
            font-weight: 800;
          }

          .sidebar-brand__text {
            color: var(--ink);
            font-size: 1.05rem;
            font-weight: 800;
          }

          .sidebar-section {
            color: #4B5565;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            margin: 1.2rem 0 0.55rem;
          }

          .sidebar-note,
          .helper-text,
          .section-subtitle {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.5;
          }

          .section-title {
            color: var(--ink);
            font-size: 1.02rem;
            font-weight: 800;
            margin: 0.35rem 0 0.35rem;
          }

          .chip-cloud {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.35rem 0 0.85rem;
          }

          .chip {
            border: 1px solid var(--line);
            border-radius: 6px;
            background: #FFFFFF;
            color: #344054;
            padding: 0.35rem 0.55rem;
            font-size: 0.78rem;
            font-weight: 600;
          }

          .mapping-overview {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.75rem 0 1rem;
          }

          .summary-tile,
          .dashboard-card,
          .mapping-summary-card,
          .export-hero {
            background: #FFFFFF;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 1px 2px rgba(20, 24, 32, 0.04);
          }

          [data-testid="stMetric"] {
            border-top: 1px solid var(--line) !important;
          }

          [data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: var(--ink);
          }

          .summary-tile__label,
          .metric-title,
          .mapping-col-label,
          .detail-stat__label {
            color: var(--muted);
            font-size: 0.73rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
          }

          .summary-tile__value,
          .detail-stat__value {
            color: var(--ink);
            font-size: 1.4rem;
            font-weight: 800;
            margin-top: 0.4rem;
          }

          .summary-tile__note {
            color: var(--muted);
            font-size: 0.82rem;
            margin-top: 0.25rem;
          }

          .mapping-shell {
            display: grid;
            grid-template-columns: minmax(280px, 0.9fr) minmax(0, 2.1fr);
            gap: 1rem;
          }

          .mapping-pair {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            padding: 0.65rem 0;
            border-bottom: 1px solid #EEF1F5;
          }

          .mapping-pair__label {
            color: var(--muted);
            font-weight: 600;
          }

          .mapping-pair__value {
            color: var(--ink);
            font-weight: 700;
          }

          .mapping-table {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
          }

          .mapping-table th,
          .mapping-table td {
            text-align: left;
            border-bottom: 1px solid #EEF1F5;
            padding: 0.55rem 0.35rem;
            overflow-wrap: anywhere;
          }

          .mapping-table th {
            color: var(--muted);
            font-size: 0.72rem;
            text-transform: uppercase;
          }

          .mapping-tag,
          .badge,
          .deliverable-type {
            display: inline-flex;
            align-items: center;
            border-radius: 6px;
            padding: 0.25rem 0.45rem;
            font-size: 0.74rem;
            font-weight: 800;
          }

          .mapping-tag--up,
          .badge-watch {
            background: var(--accent-soft);
            color: var(--accent);
          }

          .mapping-tag--down,
          .badge-high {
            background: var(--bad-soft);
            color: var(--bad);
          }

          .mapping-tag--include {
            background: var(--good-soft);
            color: var(--good);
          }

          .mapping-tag--exclude,
          .badge-medium {
            background: var(--warn-soft);
            color: var(--warn);
          }

          .export-hero {
            background: #FFFFFF;
            border-color: var(--line);
          }

          .detail-stat-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 0.75rem;
          }

          .detail-stat {
            background: #FFFFFF;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.8rem;
          }

          .footer-note {
            color: var(--muted);
            text-align: center;
            padding: 1.25rem 0 0.25rem;
          }

          @media (max-width: 1100px) {
            .app-header,
            .mapping-overview,
            .mapping-shell {
              grid-template-columns: 1fr;
            }
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
        if key.startswith(("field_", "kpi_", "kpi_row_", "filter_", "explorer_", "chart_", "alert_")):
            del st.session_state[key]
    for key, value in SESSION_DEFAULTS.items():
        st.session_state[key] = value


def set_mapping_widget_defaults(field_mapping: dict, kpi_mapping: list[dict]) -> None:
    for field in ["date", "entity_id", "team", "site", "account", "shift", "channel", "program"]:
        value = field_mapping.get(field)
        st.session_state["field_{}".format(field)] = value if value is not None else NONE_OPTION

    st.session_state["kpi_editor_rows"] = [dict(item) for item in kpi_mapping]


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


def ensure_kpi_editor_rows(default_rows: list[dict], available_kpis: list[str]) -> list[dict]:
    valid_sources = set(available_kpis)
    existing_rows = [dict(row) for row in st.session_state.get("kpi_editor_rows", [])]
    filtered_rows = []
    used_sources = set()

    for row in existing_rows:
        source = row.get("source_column")
        if source not in valid_sources or source in used_sources:
            continue
        filtered_rows.append(row)
        used_sources.add(source)

    if not filtered_rows:
        for row in default_rows:
            source = row.get("source_column")
            if source in valid_sources and source not in used_sources:
                filtered_rows.append(dict(row))
                used_sources.add(source)

    st.session_state["kpi_editor_rows"] = filtered_rows
    return filtered_rows


def add_kpi_editor_row(available_kpis: list[str], default_lookup: dict[str, dict], default_threshold_pct: float) -> None:
    rows = [dict(row) for row in st.session_state.get("kpi_editor_rows", [])]
    used_sources = {row.get("source_column") for row in rows}
    next_source = next((column for column in available_kpis if column not in used_sources), None)
    if next_source is None:
        return
    defaults = dict(
        default_lookup.get(
            next_source,
            {
                "source_column": next_source,
                "kpi_name": slugify_column_name(next_source),
                "unit": "",
                "direction_bad": "up",
                "drift_threshold_pct": default_threshold_pct,
                "include": True,
            },
        )
    )
    rows.append(defaults)
    st.session_state["kpi_editor_rows"] = rows


def remove_kpi_editor_row(index: int) -> None:
    rows = [dict(row) for row in st.session_state.get("kpi_editor_rows", [])]
    if 0 <= index < len(rows):
        rows.pop(index)
        st.session_state["kpi_editor_rows"] = rows


def build_current_mapping(raw_df: pd.DataFrame, default_threshold_pct: float) -> tuple[dict, list[dict], str]:
    field_mapping = {}
    for field in ["date", "entity_id", "team", "site", "account", "shift", "channel", "program"]:
        selected = st.session_state.get("field_{}".format(field), NONE_OPTION)
        field_mapping[field] = None if selected == NONE_OPTION else selected

    available_kpis = st.session_state.get("kpi_editor_rows", [])
    kpi_mapping = []
    for index, row in enumerate(available_kpis):
        source_column = st.session_state.get("kpi_row_source_{}".format(index), row.get("source_column"))
        if source_column not in raw_df.columns:
            continue
        kpi_mapping.append(
            {
                "source_column": source_column,
                "kpi_name": slugify_column_name(st.session_state.get("kpi_row_name_{}".format(index), source_column)),
                "unit": st.session_state.get("kpi_row_unit_{}".format(index), ""),
                "direction_bad": st.session_state.get("kpi_row_direction_{}".format(index), "up"),
                "drift_threshold_pct": float(st.session_state.get("kpi_row_threshold_{}".format(index), default_threshold_pct)),
                "include": bool(st.session_state.get("kpi_row_include_{}".format(index), True)),
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
    st.markdown(
        """
        <div class="app-header">
          <div>
            <div class="app-kicker">Operations intelligence</div>
            <div class="app-title">OpenBPO Drift</div>
            <div class="app-subtitle">Monitor KPI drift across teams, sites, accounts, and agents with local-first analysis.</div>
          </div>
          <div class="status-panel">
            <strong>Local processing active</strong>
            <span>Data is parsed, normalized, analyzed, and exported on this machine.</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def html_text(value: object) -> str:
    return escape(str(value)).replace("\n", "<br>")


def material_icon(name: str) -> str:
    icon_map = {
        "R": "database",
        "C": "view_column",
        "D": "calendar_month",
        "F": "description",
        "!": "warning",
        "N": "fact_check",
        "W": "report_problem",
        "E": "groups",
        "K": "monitoring",
        "A": "notifications_active",
        "H": "priority_high",
        "L": "show_chart",
        "B": "horizontal_rule",
        "T": "shield",
        "Δ": "trending_down",
    }
    symbol = icon_map.get(name, name or "analytics")
    return "<span class='material-symbols-rounded'>{}</span>".format(escape(symbol))


def render_metric_card(title: str, value: str, subtitle: str, accent: str = "blue", icon: str = "•") -> None:
    st.metric(title, value, help=subtitle, border=True)


def render_metric_grid(cards: list[dict[str, object]]) -> None:
    if not cards:
        return
    with st.container(horizontal=True):
        for card in cards:
            title = str(card["title"])
            value = str(card["value"])
            subtitle = str(card["subtitle"])
            delta = None
            delta_color = "normal"
            if title.lower().startswith("drift"):
                delta = subtitle
                delta_color = "inverse" if "drift" in subtitle.lower() else "off"
            st.metric(title, value, delta=delta, delta_color=delta_color, help=subtitle, border=True)


def render_empty_state(title: str, message: str) -> None:
    st.markdown(
        """
        <div class="dashboard-card" style="padding:28px 24px;">
          <div class="section-title">{title}</div>
          <div class="helper-text">{message}</div>
        </div>
        """.format(title=escape(title), message=escape(message)),
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
          <div class="sidebar-brand__mark">OB</div>
          <div class="sidebar-brand__text">OpenBPO Drift</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chip_cloud(values: list[str]) -> None:
    chips = "".join("<span class='chip'>{}</span>".format(escape(str(value))) for value in values)
    st.markdown("<div class='chip-cloud'>{}</div>".format(chips), unsafe_allow_html=True)


def render_panel_header(title: str, subtitle: str | None = None) -> None:
    st.markdown("<div class='section-title'>{}</div>".format(escape(title)), unsafe_allow_html=True)
    if subtitle:
        st.markdown("<div class='section-subtitle'>{}</div>".format(escape(subtitle)), unsafe_allow_html=True)


def unit_format(value: str) -> str:
    return UNIT_LABELS.get(value, value.title())


def kpi_display_name(kpi_name: str, kpi_rules: dict[str, dict[str, object]] | None = None) -> str:
    if kpi_rules and kpi_name in kpi_rules:
        return str(kpi_rules[kpi_name].get("label", "") or kpi_name.replace("_", " ").title())
    return kpi_name.replace("_", " ").title()


def format_kpi_metric_value(value: float, unit: str) -> str:
    if unit == "percentage":
        return "{:.1f}%".format(value)
    return "{:.1f}".format(value)


def render_visual_toolbar() -> None:
    return


def render_alert_detail_stats(selected_alert: pd.Series, baseline_window: int, current_window: int) -> None:
    st.markdown(
        """
        <div class="detail-stat-grid">
          <div class="detail-stat">
            <div class="detail-stat__label">Baseline ({baseline_window}d)</div>
            <div class="detail-stat__value">{baseline:.1f}</div>
          </div>
          <div class="detail-stat">
            <div class="detail-stat__label">Current ({current_window}d)</div>
            <div class="detail-stat__value">{current:.1f}</div>
          </div>
          <div class="detail-stat">
            <div class="detail-stat__label">Drift</div>
            <div class="detail-stat__value">{drift_pct:+.1f}%</div>
          </div>
          <div class="detail-stat">
            <div class="detail-stat__label">Threshold</div>
            <div class="detail-stat__value">{threshold_pct:.1f}% {direction_bad}</div>
          </div>
        </div>
        """.format(
            baseline_window=baseline_window,
            current_window=current_window,
            baseline=float(selected_alert["baseline"]),
            current=float(selected_alert["current"]),
            drift_pct=float(selected_alert["drift_pct"]),
            threshold_pct=float(selected_alert["threshold_pct"]),
            direction_bad=escape(str(selected_alert["direction_bad"])),
        ),
        unsafe_allow_html=True,
    )


def source_file_type(source_name: str | None) -> str:
    if not source_name:
        return "File"
    return Path(source_name).suffix.replace(".", "").upper() or "File"


def source_file_label(source_name: str | None) -> str:
    if not source_name:
        return "unknown"
    suffix = Path(source_name).suffix.lower()
    if suffix == ".csv":
        return "text/csv"
    if suffix in {".xlsx", ".xls"}:
        return "excel workbook"
    return suffix.replace(".", "") or "local file"


def missing_cells_summary(frame: pd.DataFrame) -> tuple[str, str]:
    missing = int(frame.isna().sum().sum())
    total = int(frame.shape[0] * frame.shape[1]) if not frame.empty else 0
    ratio = (missing / total * 100.0) if total else 0.0
    return "{:.1f}%".format(ratio), "{} cells".format(missing)


def summary_date_days(frame: pd.DataFrame) -> int:
    if "date" not in frame.columns:
        return 0
    parsed_dates = pd.to_datetime(frame["date"], errors="coerce")
    if parsed_dates.notna().sum() == 0:
        return 0
    return int((parsed_dates.max() - parsed_dates.min()).days) + 1


def compact_date_range(date_range: str) -> str:
    if " to " not in date_range:
        return date_range
    start, end = date_range.split(" to ", 1)
    start_date = pd.to_datetime(start, errors="coerce")
    end_date = pd.to_datetime(end, errors="coerce")
    if pd.isna(start_date) or pd.isna(end_date):
        return date_range
    def label(value: pd.Timestamp, include_year: bool) -> str:
        base = "{} {}".format(value.strftime("%b"), value.day)
        return "{}, {}".format(base, value.year) if include_year else base

    if start_date.year == end_date.year:
        return "{} to {}".format(label(start_date, False), label(end_date, True))
    return "{} to {}".format(label(start_date, True), label(end_date, True))


def quality_display_frame(quality_df: pd.DataFrame) -> pd.DataFrame:
    if quality_df.empty:
        return quality_df
    display = quality_df.copy()
    display["check"] = display["check"].map(lambda value: QUALITY_LABELS.get(value, str(value).replace("_", " ").title()))
    return display.rename(
        columns={
            "status": "Status",
            "check": "Check",
            "details": "Details",
            "affected_rows": "Affected Rows",
        }
    )


def render_mapping_summary(field_mapping: dict[str, str | None], kpi_mapping: list[dict]) -> None:
    field_parts = [
        "<div class='mapping-summary-card'>",
        "<div class='section-title'>Field Mapping</div>",
        "<div class='section-subtitle'>Canonical fields mapped to your source columns.</div>",
    ]
    for field in ["date", "entity_id", "team", "site", "account", "shift"]:
        label = FIELD_LABELS[field]
        value = field_mapping.get(field) or "Not mapped"
        field_parts.append(
            (
                "<div class=\"mapping-pair\">"
                "<div class=\"mapping-pair__label\">{label}</div>"
                "<div class=\"mapping-pair__value\">{value}</div>"
                "</div>"
            ).format(label=escape(label), value=escape(str(value)))
        )
    field_parts.append("</div>")

    kpi_parts = [
        "<div class='mapping-summary-card'>",
        "<div class='section-title'>KPI Mapping</div>",
        "<div class='section-subtitle'>How each selected KPI will be normalized and monitored.</div>",
        "<table class='mapping-table'>",
        "<thead><tr><th>Source</th><th>Canonical KPI</th><th>Unit</th><th>Direction</th><th>Threshold</th><th>Status</th></tr></thead>",
        "<tbody>",
    ]
    for item in kpi_mapping:
        direction_bad = str(item.get("direction_bad", "up"))
        direction_label = "Higher" if direction_bad == "up" else "Lower"
        direction_class = "mapping-tag--up" if direction_bad == "up" else "mapping-tag--down"
        include = bool(item.get("include", True))
        include_label = "Included" if include else "Excluded"
        include_class = "mapping-tag--include" if include else "mapping-tag--exclude"
        threshold = "{:.1f}%".format(float(item.get("drift_threshold_pct", 0.0)))
        kpi_parts.append(
            (
                "<tr>"
                "<td>{source}</td>"
                "<td><strong>{kpi_name}</strong></td>"
                "<td>{unit}</td>"
                "<td><span class=\"mapping-tag {direction_class}\">{direction_label}</span></td>"
                "<td>{threshold}</td>"
                "<td><span class=\"mapping-tag {include_class}\">{include_label}</span></td>"
                "</tr>"
            ).format(
                source=escape(str(item.get("source_column", ""))),
                kpi_name=escape(str(item.get("kpi_name", ""))),
                unit=escape(str(item.get("unit", "")) or "Unspecified"),
                direction_class=direction_class,
                direction_label=escape(direction_label),
                threshold=escape(threshold),
                include_class=include_class,
                include_label=escape(include_label),
            )
        )
    kpi_parts.append("</tbody></table></div>")

    st.markdown(
        "<div class='mapping-shell'>{left}{right}</div>".format(left="".join(field_parts), right="".join(kpi_parts)),
        unsafe_allow_html=True,
    )


def render_mapping_overview(field_mapping: dict[str, str | None], kpi_mapping: list[dict]) -> None:
    mapped_required = sum(1 for field in REQUIRED_FIELDS if field_mapping.get(field))
    included_kpis = [item for item in kpi_mapping if item.get("include", True)]
    unmapped_optional = sum(1 for field in ["channel", "program"] if not field_mapping.get(field))
    st.markdown(
        """
        <div class="mapping-overview">
          <div class="summary-tile">
            <div class="summary-tile__label">Required Fields</div>
            <div class="summary-tile__value">{mapped_required}/6</div>
            <div class="summary-tile__note">Core schema mapped</div>
          </div>
          <div class="summary-tile">
            <div class="summary-tile__label">Active KPIs</div>
            <div class="summary-tile__value">{active_kpis}</div>
            <div class="summary-tile__note">Included in drift analysis</div>
          </div>
          <div class="summary-tile">
            <div class="summary-tile__label">Optional Metadata</div>
            <div class="summary-tile__value">{remaining}</div>
            <div class="summary-tile__note">Fields still available to map</div>
          </div>
        </div>
        """.format(
            mapped_required=mapped_required,
            active_kpis=len(included_kpis),
            remaining=unmapped_optional,
        ),
        unsafe_allow_html=True,
    )


def render_export_card(title: str, body: str, icon: str, accent_bg: str, accent_text: str) -> None:
    st.markdown(
        """
        <div class="dashboard-card deliverable-card" style="--accent-line:{accent_text};">
          <div>
            <div class="deliverable-type">{icon}</div>
            <div class="section-title" style="margin-top:16px;">{title}</div>
            <div class="helper-text">{body}</div>
          </div>
        </div>
        """.format(
            title=escape(title),
            body=escape(body),
            icon=escape(icon),
            accent_text=accent_text,
        ),
        unsafe_allow_html=True,
    )


def render_pagination(active: int = 1, pages: list[str] | None = None) -> str:
    if pages is None:
        pages = ["1", "2", "3", "...", "625"]
    parts = ["<div class='pager'>", "<span class='pager-pill'>&lt;</span>"]
    for page in pages:
        css_class = "pager-pill pager-pill--active" if str(page) == str(active) else "pager-pill"
        parts.append("<span class='{css}'>{page}</span>".format(css=css_class, page=escape(str(page))))
    parts.append("<span class='pager-pill'>&gt;</span></div>")
    return "".join(parts)


def render_html_table(title: str, headers: list[str], rows: list[list[str]], footer_left: str, footer_right: str = "") -> None:
    head_html = "".join("<th>{}</th>".format(escape(header)) for header in headers)
    body_html = "".join("<tr>{}</tr>".format("".join("<td>{}</td>".format(cell) for cell in row)) for row in rows)
    st.markdown(
        """
        <div class="app-table-card">
          <div class="app-table-head">
            <div class="section-title" style="margin-bottom:0;">{title}</div>
          </div>
          <div class="app-table-wrap">
            <table class="app-table">
              <thead><tr>{head}</tr></thead>
              <tbody>{body}</tbody>
            </table>
          </div>
          <div class="table-footer">
            <div>{footer_left}</div>
            <div>{footer_right}</div>
          </div>
        </div>
        """.format(
            title=escape(title),
            head=head_html,
            body=body_html,
            footer_left=html_text(footer_left),
            footer_right=footer_right,
        ),
        unsafe_allow_html=True,
    )


def format_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return escape(value.strftime("%Y-%m-%d"))
    if isinstance(value, float):
        if abs(value) >= 100:
            return escape("{:.1f}".format(value))
        if abs(value) >= 1:
            return escape("{:.2f}".format(value))
        return escape("{:.3f}".format(value).rstrip("0").rstrip("."))
    return escape(str(value))


def render_preview_table(frame: pd.DataFrame) -> None:
    preview = frame.head(20)
    st.markdown("<div class='section-title'>Data preview</div>", unsafe_allow_html=True)
    st.caption("Showing the first {} of {:,} rows.".format(min(len(frame), 20), len(frame)))
    column_config = {}
    for column in preview.columns:
        if pd.api.types.is_datetime64_any_dtype(preview[column]) or str(column).lower() == "date":
            column_config[column] = st.column_config.DateColumn(str(column), format="YYYY-MM-DD")
        elif pd.api.types.is_numeric_dtype(preview[column]):
            column_config[column] = st.column_config.NumberColumn(str(column), format="%.2f")
        else:
            column_config[column] = st.column_config.TextColumn(str(column))
    st.dataframe(
        preview,
        width="stretch",
        height=420,
        hide_index=True,
        column_config=column_config,
    )


def render_quality_html_table(quality_df: pd.DataFrame, normalized_count: int) -> None:
    display = quality_display_frame(quality_df)
    st.markdown("<div class='section-title'>Data quality checks</div>", unsafe_allow_html=True)
    st.caption("Coverage: {:,} normalized observations.".format(normalized_count))

    def status_color(value):
        colors = {
            "Pass": "background-color: #ECF8F4; color: #176B55; font-weight: 700;",
            "Warning": "background-color: #FFF6E8; color: #A15C00; font-weight: 700;",
            "Fail": "background-color: #FFF1F1; color: #B4232A; font-weight: 700;",
            "Info": "background-color: #EEF4FF; color: #244A9B; font-weight: 700;",
        }
        return colors.get(value, "")

    styled = display.style.map(status_color, subset=["Status"])
    st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
        height=410,
        column_config={
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Check": st.column_config.TextColumn("Check", width="medium"),
            "Details": st.column_config.TextColumn("Details", width="large"),
            "Affected Rows": st.column_config.NumberColumn("Affected rows", format="%d", width="small"),
        },
    )


def render_alerts_html_table(alerts_df: pd.DataFrame) -> None:
    display = alerts_for_display(alerts_df).copy()
    st.markdown("<div class='section-title'>Drift alerts ({})</div>".format(len(display)), unsafe_allow_html=True)
    st.caption("Sorted by severity and absolute drift. Use the filters above to narrow the queue.")

    def severity_color(value):
        if value == "High":
            return "background-color: #FFF1F1; color: #B4232A; font-weight: 700;"
        if value == "Medium":
            return "background-color: #FFF6E8; color: #A15C00; font-weight: 700;"
        return "background-color: #EEF4FF; color: #244A9B; font-weight: 700;"

    def drift_color(value):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return ""
        if abs(numeric) >= 30:
            return "color: #B4232A; font-weight: 800;"
        if abs(numeric) >= 15:
            return "color: #A15C00; font-weight: 800;"
        return "color: #244A9B; font-weight: 700;"

    styled = display.style.map(severity_color, subset=["Severity"]).map(drift_color, subset=["Drift %"])
    st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
        height=360,
        column_config={
            "Severity": st.column_config.TextColumn("Severity", width="small"),
            "Entity ID": st.column_config.TextColumn("Entity ID", pinned=True, width="small"),
            "Team": st.column_config.TextColumn("Team", width="medium"),
            "Account": st.column_config.TextColumn("Account", width="medium"),
            "KPI": st.column_config.TextColumn("KPI", width="small"),
            "Baseline": st.column_config.NumberColumn("Baseline", format="%.2f"),
            "Current": st.column_config.NumberColumn("Current", format="%.2f"),
            "Drift %": st.column_config.NumberColumn("Drift %", format="%+.1f%%"),
            "Threshold %": st.column_config.NumberColumn("Threshold %", format="%.1f%%"),
            "Observations": st.column_config.NumberColumn("Observations", format="%d"),
            "Latest Date": st.column_config.TextColumn("Latest date", width="small"),
            "Explanation": st.column_config.TextColumn("Explanation", width="large"),
        },
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
        return "background-color: {}; color: {}; font-weight: 700; border-radius: 999px".format(background, color)

    display = quality_display_frame(quality_df)
    return display.style.map(status_style, subset=["Status"])


def style_alert_table(alerts_df: pd.DataFrame):
    def severity_style(value):
        if value == "High":
            return "background-color: #FEE2E2; color: #B91C1C; font-weight: 700"
        if value == "Medium":
            return "background-color: #FFEDD5; color: #C2410C; font-weight: 700"
        return "background-color: #DBEAFE; color: #1D4ED8; font-weight: 700"

    def drift_style(value):
        try:
            drift_value = float(value)
        except (TypeError, ValueError):
            drift_value = 0.0
        color = "#B91C1C" if drift_value < 0 else "#1D4ED8"
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
    st.set_page_config(page_title="OpenBPO Drift", page_icon=":material/monitoring:", layout="wide", initial_sidebar_state="expanded")
    inject_styles()
    init_session_state()
    ensure_sample_data()
    default_rules = load_kpi_rules(DEFAULT_RULES_PATH)

    render_sidebar_brand()
    st.sidebar.markdown("<div class='sidebar-section'>Load Data</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-note' style='margin-bottom:6px;color:#0F172A;font-weight:600;'>Upload CSV or Excel</div>", unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("Upload CSV / XLSX / XLS", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
    load_sample = st.sidebar.button("Load sample data", icon=":material/database:", width="stretch")
    st.sidebar.markdown("<div class='sidebar-note'>Loads a deterministic BPO demo with SSA National 800 Number workload context.</div>", unsafe_allow_html=True)

    if load_sample:
        raw_df = load_tabular_file(SAMPLE_DATA_PATH)
        store_source_frame(raw_df, ("sample", SAMPLE_DATA_PATH.name), SAMPLE_DATA_PATH.name, True, default_rules)

    sheet_name: str | int = 0
    if uploaded_file is not None and Path(uploaded_file.name).suffix.lower() in {".xlsx", ".xls"}:
        st.sidebar.markdown("<div class='sidebar-section'>Worksheet</div>", unsafe_allow_html=True)
        sheet_options = list(list_excel_sheets(uploaded_file))
        sheet_name = st.sidebar.selectbox("Select worksheet", sheet_options, index=0)

    load_source_if_needed(uploaded_file, sheet_name, default_rules)

    st.sidebar.markdown("<div class='sidebar-section'>Analysis Windows</div>", unsafe_allow_html=True)
    baseline_window = st.sidebar.number_input("Baseline window (days)", min_value=7, max_value=60, value=DEFAULT_BASELINE_WINDOW, step=1)
    current_window = st.sidebar.number_input("Current window (days)", min_value=3, max_value=30, value=DEFAULT_CURRENT_WINDOW, step=1)

    st.sidebar.markdown("<div class='sidebar-section'>Drift Settings</div>", unsafe_allow_html=True)
    default_threshold_pct = st.sidebar.number_input("Default drift threshold (%)", min_value=1.0, max_value=100.0, value=DEFAULT_THRESHOLD_PCT, step=1.0)

    st.sidebar.markdown("<div class='sidebar-section'>Export</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-note'>Exports appear in the Export tab after analysis.</div>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset app", icon=":material/refresh:", width="stretch"):
        reset_app_state()
        st.rerun()
    st.sidebar.markdown("<div class='sidebar-note' style='margin-top:4px;'>OpenBPO Drift v1.0.0</div>", unsafe_allow_html=True)

    render_header()

    if st.session_state.raw_df is None or st.session_state.raw_df.empty:
        render_empty_state("Welcome", "Upload a CSV/Excel file or load sample data to begin.")
        return

    raw_df = st.session_state.raw_df
    source_field_mapping = st.session_state.field_mapping or guess_field_mapping(raw_df.columns)
    source_kpi_defaults = st.session_state.kpi_mapping or build_default_kpi_mapping(
        raw_df.columns.tolist(),
        infer_kpi_candidates(raw_df.columns, source_field_mapping),
        default_rules,
        default_threshold_pct,
    )

    tabs = st.tabs(
        [
            ":material/table_chart: Data preview",
            ":material/schema: Schema mapper",
            ":material/verified: Data quality",
            ":material/notifications_active: Drift alerts",
            ":material/query_stats: KPI explorer",
            ":material/download: Export",
        ]
    )

    with tabs[0]:
        date_range = "Unavailable"
        if "date" in raw_df.columns:
            parsed_dates = pd.to_datetime(raw_df["date"], errors="coerce")
            if parsed_dates.notna().any():
                date_range = "{} to {}".format(parsed_dates.min().strftime("%Y-%m-%d"), parsed_dates.max().strftime("%Y-%m-%d"))
        missing_pct, missing_cells = missing_cells_summary(raw_df)
        with st.container():
            render_panel_header("Data overview")
            render_metric_grid(
                [
                    {"title": "Rows loaded", "value": "{:,}".format(len(raw_df)), "subtitle": "Source rows"},
                    {"title": "Columns detected", "value": str(len(raw_df.columns)), "subtitle": "Raw fields"},
                    {"title": "Date range", "value": compact_date_range(date_range), "subtitle": "{} days".format(summary_date_days(raw_df))},
                    {"title": "File type", "value": source_file_type(st.session_state.source_name), "subtitle": source_file_label(st.session_state.source_name)},
                    {"title": "Missing values", "value": missing_pct, "subtitle": missing_cells},
                ]
            )
        with st.container():
            render_panel_header("Detected columns")
            render_chip_cloud(raw_df.columns.tolist())
        if st.session_state.sample_loaded:
            st.caption(
                "Sample data is synthetic BPO operational detail with SSA National 800 Number workload context: {}".format(
                    SSA_N8NN_SOURCE_URL
                )
            )
        render_preview_table(raw_df)

    with tabs[1]:
        options = [NONE_OPTION] + raw_df.columns.tolist()
        with st.container():
            render_panel_header("Schema mapping", "Map your raw dataset columns to the standardized schema used by OpenBPO Drift.")
            preview_field_mapping, preview_kpi_mapping, _ = build_current_mapping(raw_df, default_threshold_pct)
            render_mapping_overview(preview_field_mapping, preview_kpi_mapping)

            required_col, kpi_col = st.columns([0.95, 2.55])
            with required_col:
                with st.container():
                    render_panel_header("Required fields")
                    for field in ["date", "entity_id", "team", "site", "account", "shift"]:
                        st.selectbox("{} column".format(FIELD_LABELS[field]), options, key="field_{}".format(field))
                    st.markdown("<div class='section-title' style='margin-top:18px;'>Optional metadata</div>", unsafe_allow_html=True)
                    st.selectbox("Channel column", options, key="field_channel")
                    st.selectbox("Program column", options, key="field_program")
                    st.caption("Optional fields help with deeper slicing and context, but are not required.")

            selected_fields = {
                field: st.session_state.get("field_{}".format(field), NONE_OPTION)
                for field in ["date", "entity_id", "team", "site", "account", "shift", "channel", "program"]
            }
            excluded_columns = {value for value in selected_fields.values() if value != NONE_OPTION}
            available_kpis = [column for column in raw_df.columns if column not in excluded_columns]
            default_selected_kpis = [item["source_column"] for item in source_kpi_defaults if item["source_column"] in available_kpis]
            default_lookup = {item["source_column"]: item for item in source_kpi_defaults}
            kpi_rows = ensure_kpi_editor_rows(
                [default_lookup[source] for source in default_selected_kpis if source in default_lookup],
                available_kpis,
            )

            with kpi_col:
                with st.container():
                    header_left, header_right = st.columns([2.65, 1.35])
                    with header_left:
                        render_panel_header("KPI columns", "Select and configure the KPI columns you want to monitor for drift.")
                    with header_right:
                        st.write("")
                        if st.button("Add KPI column", icon=":material/add:", width="stretch", disabled=len(kpi_rows) >= len(available_kpis)):
                            add_kpi_editor_row(available_kpis, default_lookup, float(default_threshold_pct))
                            st.rerun()

                    header = st.columns([0.42, 1.45, 1.95, 0.88, 0.98, 1.02, 0.34])
                    header_labels = ["Inc", "Source<br>Column", "KPI Name", "Unit", "Bad<br>Direction", "Threshold %", ""]
                    for label, column in zip(header_labels, header):
                        with column:
                            if label:
                                st.markdown("<div class='mapping-col-label'>{}</div>".format(label), unsafe_allow_html=True)
                    for index, row_data in enumerate(kpi_rows):
                        source_column = row_data["source_column"]
                        defaults = default_lookup.get(
                            source_column,
                            {
                                "source_column": source_column,
                                "kpi_name": source_column.lower(),
                                "unit": "number",
                                "direction_bad": "up",
                                "drift_threshold_pct": default_threshold_pct,
                                "include": True,
                            },
                        )
                        row = st.columns([0.42, 1.45, 1.95, 0.88, 0.98, 1.02, 0.34])
                        taken_sources = {item["source_column"] for idx, item in enumerate(kpi_rows) if idx != index}
                        available_for_row = [source_column] + [column for column in available_kpis if column == source_column or column not in taken_sources]
                        current_unit = str(defaults.get("unit", "number") or "number")
                        available_units = [current_unit] + [unit for unit in UNIT_OPTIONS if unit != current_unit]
                        st.session_state.setdefault("kpi_row_include_{}".format(index), bool(defaults.get("include", True)))
                        st.session_state.setdefault("kpi_row_source_{}".format(index), source_column)
                        st.session_state.setdefault("kpi_row_name_{}".format(index), defaults["kpi_name"])
                        st.session_state.setdefault("kpi_row_unit_{}".format(index), current_unit)
                        st.session_state.setdefault("kpi_row_direction_{}".format(index), defaults.get("direction_bad", "up"))
                        st.session_state.setdefault("kpi_row_threshold_{}".format(index), float(defaults.get("drift_threshold_pct", default_threshold_pct)))
                        with row[0]:
                            st.checkbox("Include row {}".format(index), key="kpi_row_include_{}".format(index), label_visibility="collapsed")
                        with row[1]:
                            st.selectbox("Source row {}".format(index), available_for_row, key="kpi_row_source_{}".format(index), label_visibility="collapsed")
                        with row[2]:
                            st.text_input("KPI name row {}".format(index), key="kpi_row_name_{}".format(index), label_visibility="collapsed")
                        with row[3]:
                            st.selectbox("Unit row {}".format(index), available_units, format_func=unit_format, key="kpi_row_unit_{}".format(index), label_visibility="collapsed")
                        with row[4]:
                            st.selectbox("Direction row {}".format(index), ["up", "down"], format_func=lambda value: "Higher" if value == "up" else "Lower", key="kpi_row_direction_{}".format(index), label_visibility="collapsed")
                        with row[5]:
                            st.number_input("Threshold row {}".format(index), min_value=1.0, max_value=100.0, step=1.0, key="kpi_row_threshold_{}".format(index), label_visibility="collapsed")
                        with row[6]:
                            if st.button(" ", icon=":material/delete:", key="kpi_row_delete_{}".format(index), width="stretch", help="Remove KPI row"):
                                remove_kpi_editor_row(index)
                                st.rerun()
                    st.markdown("<div class='mapping-grid-note'>Thresholds default to Drift Settings but can be overridden per KPI.</div>", unsafe_allow_html=True)

            field_mapping, kpi_mapping, mapping_yaml = build_current_mapping(raw_df, default_threshold_pct)
            action_cols = st.columns([1.15, 1.2, 2.6])
            with action_cols[0]:
                preview_clicked = st.button("Preview normalized data", icon=":material/visibility:", width="stretch")
            with action_cols[1]:
                st.download_button(
                    "Export mapping YAML",
                    icon=":material/download:",
                    data=mapping_yaml.encode("utf-8"),
                    file_name="mapping.yaml",
                    mime="text/yaml",
                    width="stretch",
                )
            with action_cols[2]:
                normalize_clicked = st.button("Normalize data", icon=":material/play_arrow:", type="primary", width="stretch")

            if preview_clicked:
                preview_frame = normalize_to_long(st.session_state.raw_df, field_mapping, kpi_mapping, include_raw_value=True)
                st.dataframe(preview_frame[CANONICAL_COLUMNS].head(20), width="stretch", hide_index=True)

            with st.expander("Review current mapping and YAML"):
                render_mapping_summary(field_mapping, kpi_mapping)
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
        render_panel_header("Data quality")
        if normalized_df is None:
            render_empty_state("Data Quality", "Normalize your data first to view data quality checks.")
        else:
            warnings_count = int((quality_df["status"] == "Warning").sum()) if not quality_df.empty else 0
            failures_count = int((quality_df["status"] == "Fail").sum()) if not quality_df.empty else 0
            warning_rows = int(quality_df.loc[quality_df["status"] == "Warning", "affected_rows"].sum()) if not quality_df.empty else 0
            failure_rows = int(quality_df.loc[quality_df["status"] == "Fail", "affected_rows"].sum()) if not quality_df.empty else 0
            render_metric_grid(
                [
                    {"title": "Rows loaded", "value": "{:,}".format(len(raw_df)), "subtitle": "From 1 file"},
                    {
                        "title": "Normalized observations",
                        "value": "{:,}".format(len(normalized_df)),
                        "subtitle": "{:.1f}% of source rows".format((len(normalized_df) / max(len(raw_df), 1)) * 100.0 / max(len(st.session_state.kpi_mapping), 1)),
                        "accent": "green",
                        "icon": "N",
                        "basis": "260px",
                    },
                    {
                        "title": "Validation warnings",
                        "value": str(warnings_count),
                        "subtitle": "{:.1f}% of rows".format((warning_rows / max(len(normalized_df), 1)) * 100.0),
                        "accent": "amber",
                        "icon": "W",
                        "basis": "260px",
                    },
                    {
                        "title": "Validation failures",
                        "value": str(failures_count),
                        "subtitle": "{:.1f}% of rows".format((failure_rows / max(len(normalized_df), 1)) * 100.0),
                        "accent": "red",
                        "icon": "F",
                        "basis": "260px",
                    },
                ]
            )
            if mapping_is_stale:
                st.warning("Mapping changes are pending. Click Normalize Data to refresh these results.")
            render_quality_html_table(quality_df, len(normalized_df))
            st.markdown("<div class='footer-note'>All calculations are deterministic and run locally on your machine.</div>", unsafe_allow_html=True)

    with tabs[3]:
        render_panel_header("Drift alerts")
        if normalized_df is None:
            render_empty_state("Drift Alerts", "Run drift detection to view alerts.")
        else:
            if mapping_is_stale:
                st.warning("Mapping or KPI rule changes are pending. Click Normalize Data to refresh drift outputs.")
            summary = summarize_monitoring(normalized_df, alerts_df)
            render_metric_grid(
                [
                    {"title": "Entities monitored", "value": str(summary["entities_monitored"]), "subtitle": "Agents"},
                    {"title": "KPIs monitored", "value": str(summary["kpis_monitored"]), "subtitle": "Metrics"},
                    {"title": "Active alerts", "value": str(summary["active_alerts"]), "subtitle": "Drifting now"},
                    {"title": "High severity", "value": str(summary["high_severity_alerts"]), "subtitle": "Needs attention"},
                    {"title": "Date range", "value": compact_date_range(summary["date_range"]), "subtitle": "{} days".format(summary["date_days"])},
                ]
            )

            if alerts_df.empty:
                render_empty_state("No Active Drift Alerts", "No entities currently breach the configured drift thresholds for the selected analysis windows.")
            else:
                with st.container():
                    st.markdown("<div class='section-title'>Alert filters</div>", unsafe_allow_html=True)
                    st.markdown("<div class='section-subtitle'>Focus the alert queue by severity, KPI, team, or account.</div>", unsafe_allow_html=True)
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
                        if st.button("Reset filters", icon=":material/filter_alt_off:", width="stretch"):
                            for key in ["filter_severity", "filter_kpi", "filter_team", "filter_account"]:
                                st.session_state[key] = "All"
                            st.rerun()

                filtered_alerts = filter_alerts(alerts_df)
                render_alerts_html_table(filtered_alerts)

                if not filtered_alerts.empty:
                    labels = [alert_label(row) for _, row in filtered_alerts.iterrows()]
                    if st.session_state.selected_alert not in labels:
                        st.session_state.selected_alert = labels[0]
                    selected_alert = filtered_alerts.iloc[labels.index(st.session_state.selected_alert)]

                    chart_entity_default = selected_alert["entity_id"]
                    chart_kpi_default = selected_alert["kpi_name"]
                    if st.session_state.get("chart_entity") not in normalized_df["entity_id"].dropna().astype(str).unique().tolist():
                        st.session_state["chart_entity"] = chart_entity_default
                    if st.session_state.get("chart_kpi") not in normalized_df["kpi_name"].dropna().astype(str).unique().tolist():
                        st.session_state["chart_kpi"] = chart_kpi_default

                    lower, right = st.columns([1.4, 1])
                    with lower:
                        with st.container():
                            render_panel_header("Trend context", "Inspect the selected entity and KPI against the current analysis windows.")
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
                                config={"displayModeBar": False},
                            )
                    with right:
                        badge_class = "badge-high" if selected_alert["severity"] == "High" else "badge-medium" if selected_alert["severity"] == "Medium" else "badge-watch"
                        st.markdown(
                            """
                            <div class="dashboard-card">
                              <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">
                                <div class="section-title" style="margin-bottom:0;">Alert explanation</div>
                                <span class="badge {badge_class}">{severity}</span>
                              </div>
                              <div style="margin-top:14px;color:#111827;line-height:1.65;">{explanation}</div>
                            </div>
                            """.format(
                                badge_class=escape(badge_class),
                                severity=escape(str(selected_alert["severity"])),
                                explanation=escape(str(selected_alert["explanation"])),
                            ),
                            unsafe_allow_html=True,
                        )
                        render_alert_detail_stats(selected_alert, int(baseline_window), int(current_window))

    with tabs[4]:
        render_panel_header("KPI explorer")
        if normalized_df is None:
            render_empty_state("KPI Explorer", "Run drift detection to explore KPI trends.")
        else:
            explorer_filters = normalized_df.copy()
            team_options = ["All"] + sorted(explorer_filters["team"].dropna().astype(str).unique().tolist())
            account_options = ["All"] + sorted(explorer_filters["account"].dropna().astype(str).unique().tolist())
            shift_options = ["All"] + sorted(explorer_filters["shift"].dropna().astype(str).unique().tolist())

            if st.session_state.get("explorer_team") not in {None, "All"}:
                explorer_filters = explorer_filters[explorer_filters["team"].astype(str) == st.session_state["explorer_team"]]
            if st.session_state.get("explorer_account") not in {None, "All"}:
                explorer_filters = explorer_filters[explorer_filters["account"].astype(str) == st.session_state["explorer_account"]]
            if st.session_state.get("explorer_shift") not in {None, "All"}:
                explorer_filters = explorer_filters[explorer_filters["shift"].astype(str) == st.session_state["explorer_shift"]]

            entity_options = sorted(explorer_filters["entity_id"].dropna().astype(str).unique().tolist())
            kpi_options = sorted(explorer_filters["kpi_name"].dropna().astype(str).unique().tolist())
            if not entity_options or not kpi_options:
                render_empty_state("No Matching Data", "The current KPI explorer filters do not match any normalized observations.")
            else:
                if st.session_state.get("explorer_entity") not in entity_options:
                    st.session_state["explorer_entity"] = entity_options[0]
                if st.session_state.get("explorer_kpi") not in kpi_options:
                    st.session_state["explorer_kpi"] = kpi_options[0]
                left_panel, chart_panel = st.columns([1.05, 2.35])
                with left_panel:
                    with st.container():
                        render_panel_header("Explore KPI")
                        st.selectbox("Entity", entity_options, key="explorer_entity")
                        st.selectbox("KPI", kpi_options, key="explorer_kpi")
                        st.selectbox("Team", team_options, key="explorer_team")
                        st.selectbox("Account", account_options, key="explorer_account")
                        st.selectbox("Shift", shift_options, key="explorer_shift")
                        st.button("Apply filters", icon=":material/filter_list:", width="stretch")

                explorer_chart = build_signal_frame(
                    normalized_df,
                    entity_id=st.session_state["explorer_entity"],
                    kpi_name=st.session_state["explorer_kpi"],
                    baseline_window=int(baseline_window),
                    current_window=int(current_window),
                )
                if explorer_chart.empty:
                    render_empty_state("No KPI History", "Selected entity and KPI have no observations for the current filters.")
                    return
                latest_value = float(explorer_chart["kpi_value"].iloc[-1])
                baseline_series = explorer_chart["baseline_mean"].dropna()
                baseline_value = float(baseline_series.iloc[-1]) if not baseline_series.empty else latest_value
                current_series = explorer_chart.loc[explorer_chart["is_current_window"], "kpi_value"]
                current_value = float(current_series.mean()) if not current_series.empty else latest_value
                drift_pct = ((current_value - baseline_value) / baseline_value) * 100 if baseline_value else 0.0
                rule = st.session_state.kpi_rules.get(st.session_state["explorer_kpi"], {})
                kpi_title = kpi_display_name(st.session_state["explorer_kpi"], st.session_state.kpi_rules)
                chart_unit = str(rule.get("unit", "") or "")
                latest_date = explorer_chart["date"].iloc[-1].strftime("%b %d, %Y")
                baseline_range_label = (
                    "{} \u2013 {}".format(
                        explorer_chart["date"].iloc[max(len(explorer_chart) - int(baseline_window) - int(current_window), 0)].strftime("%b %d, %Y"),
                        explorer_chart["date"].iloc[max(len(explorer_chart) - int(current_window) - 1, 0)].strftime("%b %d, %Y"),
                    )
                    if len(explorer_chart) > int(current_window)
                    else "{} day baseline".format(int(baseline_window))
                )
                current_dates = explorer_chart.loc[explorer_chart["is_current_window"], "date"]
                current_range_label = (
                    "{} \u2013 {}".format(current_dates.iloc[0].strftime("%b %d, %Y"), current_dates.iloc[-1].strftime("%b %d, %Y"))
                    if not current_dates.empty
                    else "{} day current".format(int(current_window))
                )

                with chart_panel:
                    with st.container():
                        render_panel_header(
                            "{} for {}".format(kpi_title, st.session_state["explorer_entity"]),
                            "Baseline window: {} days  |  Current window: {} days".format(int(baseline_window), int(current_window)),
                        )
                        render_visual_toolbar()
                        st.plotly_chart(
                            make_kpi_trend_chart(
                                normalized_df,
                                entity_id=st.session_state["explorer_entity"],
                                kpi_name=st.session_state["explorer_kpi"],
                                baseline_window=int(baseline_window),
                                current_window=int(current_window),
                                yaxis_title=kpi_title if not chart_unit else "{} ({})".format(kpi_title, unit_format(chart_unit)),
                            ),
                            width="stretch",
                            config={"displayModeBar": False},
                        )
                threshold_direction = "Up" if str(rule.get("direction_bad", "up")) == "up" else "Down"
                render_metric_grid(
                    [
                        {"title": "Latest value", "value": format_kpi_metric_value(latest_value, chart_unit), "subtitle": latest_date},
                        {"title": "Baseline mean ({}d)".format(int(baseline_window)), "value": format_kpi_metric_value(baseline_value, chart_unit), "subtitle": baseline_range_label},
                        {"title": "Current mean ({}d)".format(int(current_window)), "value": format_kpi_metric_value(current_value, chart_unit), "subtitle": current_range_label},
                        {"title": "Drift %", "value": "{:+.1f}%".format(drift_pct), "subtitle": "High drift" if abs(drift_pct) >= float(rule.get("drift_threshold_pct", default_threshold_pct)) else "Within range"},
                        {"title": "Threshold", "value": "{:.1f}%".format(float(rule.get("drift_threshold_pct", default_threshold_pct))), "subtitle": threshold_direction, "accent": "blue", "icon": "T", "basis": "220px"},
                    ]
                )

    with tabs[5]:
        render_panel_header("Export")
        if normalized_df is None:
            render_empty_state("Export", "Run analysis first to enable exports.")
        else:
            if mapping_is_stale:
                st.warning("The current mapping has changed since the last normalization. Exported results still reflect the last normalized state.")
            markdown_report = generate_markdown_report(alerts_df, normalized_df[CANONICAL_COLUMNS], quality_df)
            export_cards = st.columns(4)
            with export_cards[0]:
                render_export_card("Normalized KPI Data", "Export the normalized KPI dataset used as the analysis foundation.", "CSV", "#ECFDF5", "#15803D")
                st.download_button(
                    "Download CSV",
                    icon=":material/download:",
                    data=normalized_df[CANONICAL_COLUMNS].to_csv(index=False).encode("utf-8"),
                    file_name="normalized_kpis.csv",
                    mime="text/csv",
                    width="stretch",
                )
            with export_cards[1]:
                render_export_card("Drift Alerts", "Export the current alert queue with severity, baselines, and explanations.", "ALT", "#FEF2F2", "#B91C1C")
                st.download_button(
                    "Download CSV",
                    icon=":material/download:",
                    data=alerts_df.to_csv(index=False).encode("utf-8"),
                    file_name="drift_alerts.csv",
                    mime="text/csv",
                    width="stretch",
                )
            with export_cards[2]:
                render_export_card("Mapping YAML", "Export the field and KPI mapping used to normalize the source dataset.", "YML", "#F5F3FF", "#6D28D9")
                st.download_button(
                    "Download YAML",
                    icon=":material/download:",
                    data=mapping_yaml.encode("utf-8"),
                    file_name="mapping.yaml",
                    mime="text/yaml",
                    width="stretch",
                )
            with export_cards[3]:
                render_export_card("Markdown Report", "Export a concise write-up of quality signals, drift findings, and outputs.", "MD", "#EFF6FF", "#1D4ED8")
                st.download_button(
                    "Download MD",
                    icon=":material/download:",
                    data=markdown_report.encode("utf-8"),
                    file_name="openbpo_drift_report.md",
                    mime="text/markdown",
                    width="stretch",
                )
            st.markdown(
                """
                <div class="export-hero" style="margin-top:18px;">
                  <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:24px;">
                    <div style="display:flex;gap:16px;align-items:flex-start;">
                      <div style="width:42px;height:42px;border-radius:999px;background:#DBEAFE;color:#1D4ED8;display:flex;align-items:center;justify-content:center;font-size:1rem;font-weight:800;">i</div>
                      <div>
                        <div style="font-size:1.3rem;font-weight:800;color:#0F172A;margin-bottom:6px;">All files are generated locally and stay on your machine.</div>
                        <div style="font-size:0.96rem;color:#475569;">Rerun the analysis anytime with updated data, thresholds, or mappings.</div>
                      </div>
                    </div>
                    <div style="font-size:1.85rem;color:#94A3B8;line-height:1;">LOCAL</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        "<div class='footer-note'>OpenBPO Drift runs locally. No external services, no telemetry, and no data leaves your machine.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
