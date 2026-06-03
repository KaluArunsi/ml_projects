# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import plotly.graph_objects as go

from src.drift import build_signal_frame


def make_kpi_trend_chart(df, entity_id: str, kpi_name: str, baseline_window: int = 14, current_window: int = 3) -> go.Figure:
    chart = build_signal_frame(df, entity_id, kpi_name, baseline_window=baseline_window, current_window=current_window)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=chart["date"], y=chart["kpi_value"], mode="lines+markers", name="Actual"))
    figure.add_trace(go.Scatter(x=chart["date"], y=chart["baseline_mean"], mode="lines", name="Baseline", line={"dash": "dash"}))

    current = chart[chart["is_current_window"]]
    if not current.empty:
        figure.add_trace(
            go.Scatter(
                x=current["date"],
                y=current["kpi_value"],
                mode="markers",
                marker={"size": 10, "symbol": "diamond"},
                name="Current window",
            )
        )

    figure.update_layout(
        title="{} - {} Trend".format(entity_id, kpi_name.upper()),
        xaxis_title="Date",
        yaxis_title=kpi_name,
        margin={"l": 24, "r": 24, "t": 56, "b": 24},
        template="plotly_white",
    )
    return figure
