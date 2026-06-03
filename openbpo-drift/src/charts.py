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
            name="Baseline Mean ({:d}d)".format(int(baseline_window)),
            line={"dash": "dash", "color": "#22C55E", "width": 2},
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
                name="Current Window ({:d}d)".format(int(current_window)),
            )
        )

    figure.update_layout(
        xaxis_title="Date",
        yaxis_title=kpi_name,
        margin={"l": 24, "r": 24, "t": 20, "b": 24},
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        hovermode="x unified",
        legend={
            "orientation": "h",
            "x": 0.5,
            "y": 1.1,
            "xanchor": "center",
            "yanchor": "bottom",
        },
    )
    figure.update_xaxes(showgrid=False, linecolor="#CBD5E1", tickfont={"size": 12, "color": "#475569"})
    figure.update_yaxes(gridcolor="#E2E8F0", zeroline=False, tickfont={"size": 12, "color": "#475569"})
    return figure
