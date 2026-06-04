# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import plotly.graph_objects as go

from src.drift import build_signal_frame


def make_kpi_trend_chart(
    df,
    entity_id: str,
    kpi_name: str,
    baseline_window: int = 14,
    current_window: int = 3,
    yaxis_title: str | None = None,
) -> go.Figure:
    chart = build_signal_frame(df, entity_id, kpi_name, baseline_window=baseline_window, current_window=current_window)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=chart["date"],
            y=chart["kpi_value"],
            mode="lines",
            name="Actual",
            line={"color": "#244A9B", "width": 2.4},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=chart["date"],
            y=chart["baseline_mean"],
            mode="lines",
            name="Baseline Mean ({:d}d)".format(int(baseline_window)),
            line={"dash": "dash", "color": "#176B55", "width": 1.8},
        )
    )

    current = chart[chart["is_current_window"]]
    if not current.empty:
        figure.add_vrect(
            x0=current["date"].min(),
            x1=current["date"].max(),
            fillcolor="#F2F0FF",
            opacity=0.75,
            line_width=0,
            layer="below",
        )
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 12, "symbol": "square", "color": "#C8C0F2"},
                name="Current Window ({:d}d)".format(int(current_window)),
            )
        )

    figure.update_layout(
        xaxis_title="Date",
        yaxis_title=yaxis_title or kpi_name,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        hovermode="x unified",
        height=400,
        legend={
            "orientation": "h",
            "x": 0.5,
            "y": 1.06,
            "xanchor": "center",
            "yanchor": "bottom",
            "font": {"size": 12, "color": "#475569"},
        },
        hoverlabel={"bgcolor": "#FFFFFF", "bordercolor": "#CBD5E1", "font_size": 12},
    )
    figure.update_xaxes(
        showgrid=False,
        linecolor="#D9DEE8",
        tickfont={"size": 12, "color": "#475569"},
        title_font={"size": 13, "color": "#334155"},
    )
    figure.update_yaxes(
        gridcolor="#E8ECF2",
        gridwidth=1,
        zeroline=False,
        tickfont={"size": 12, "color": "#475569"},
        title_font={"size": 13, "color": "#334155"},
    )
    return figure
