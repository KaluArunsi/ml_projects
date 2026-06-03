# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import plotly.graph_objects as go

from src.drift import build_signal_frame


def make_kpi_trend_chart(df, entity_id: str, kpi_name: str, baseline_window: int = 14, current_window: int = 3) -> go.Figure:
    chart = build_signal_frame(df, entity_id, kpi_name, baseline_window=baseline_window, current_window=current_window)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=chart["date"],
            y=chart["kpi_value"],
            mode="lines+markers",
            name="Actual",
            line={"color": "#2563EB", "width": 3},
            marker={"size": 6, "color": "#2563EB"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=chart["date"],
            y=chart["baseline_mean"],
            mode="lines",
            name="Baseline",
            line={"dash": "dash", "color": "#94A3B8", "width": 2},
        )
    )

    current = chart[chart["is_current_window"]]
    if not current.empty:
        figure.add_vrect(
            x0=current["date"].min(),
            x1=current["date"].max(),
            fillcolor="#DBEAFE",
            opacity=0.35,
            line_width=0,
            layer="below",
        )
        figure.add_trace(
            go.Scatter(
                x=current["date"],
                y=current["kpi_value"],
                mode="markers",
                marker={"size": 10, "symbol": "diamond", "color": "#1D4ED8"},
                name="Current window",
            )
        )

    figure.update_layout(
        title="{} - {} Trend".format(entity_id, kpi_name.upper()),
        xaxis_title="Date",
        yaxis_title=kpi_name,
        margin={"l": 24, "r": 24, "t": 56, "b": 24},
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        hovermode="x unified",
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor="#E2E8F0", zeroline=False)
    return figure
