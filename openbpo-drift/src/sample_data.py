# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


SSA_N8NN_SOURCE_URL = "https://www.ssa.gov/open/data/800-number-call-volume-and-agent-busy-rate.html"

# Small benchmark seed inspired by SSA National 800 Number workload reporting.
# The app expands this aggregate context into synthetic BPO rows so the demo
# remains deterministic, privacy-safe, and usable without network access.
SSA_N8NN_BENCHMARK = [
    {"month": "2026-04", "calls_offered": 2_715_000, "busy_rate_pct": 7.9},
    {"month": "2026-05", "calls_offered": 2_865_000, "busy_rate_pct": 8.6},
    {"month": "2026-06", "calls_offered": 2_540_000, "busy_rate_pct": 6.8},
]


def _benchmark_for_day(day: datetime) -> dict[str, float | int | str]:
    month_key = day.strftime("%Y-%m")
    return next((item for item in SSA_N8NN_BENCHMARK if item["month"] == month_key), SSA_N8NN_BENCHMARK[-1])


def generate_sample_bpo_kpis(output_path: str | Path = "data/sample_bpo_kpis.csv") -> pd.DataFrame:
    output_path = Path(output_path)
    teams = [
        {"team": "Team Manila A", "site": "Manila", "account": "Telco", "channel": "Voice", "program": "Retention"},
        {"team": "Team Manila B", "site": "Manila", "account": "Retail", "channel": "Voice", "program": "Order Support"},
        {"team": "Team Cebu C", "site": "Cebu", "account": "Travel", "channel": "Voice", "program": "Reservations"},
        {"team": "Team Davao D", "site": "Davao", "account": "Healthcare", "channel": "Voice", "program": "Benefits"},
    ]
    shifts = ("Day", "Swing", "Night")
    start_date = datetime(2026, 4, 1)
    rows = []

    for offset in range(75):
        day = start_date + timedelta(days=offset)
        benchmark = _benchmark_for_day(day)
        weekday_pressure = [1.08, 1.04, 1.0, 0.99, 0.94, 0.72, 0.66][day.weekday()]
        month_pressure = float(benchmark["busy_rate_pct"]) / 8.0
        seasonality = (offset % 7) - 3

        for agent_number in range(48):
            team_config = teams[agent_number // 12]
            agent_id = "A{:03d}".format(agent_number + 1)
            shift = shifts[agent_number % len(shifts)]
            tenure_band = ["new", "steady", "senior"][agent_number % 3]
            skill_factor = {"new": 1.08, "steady": 1.0, "senior": 0.94}[tenure_band]
            account_pressure = {
                "Telco": 1.06,
                "Retail": 0.98,
                "Travel": 1.1,
                "Healthcare": 1.02,
            }[team_config["account"]]

            daily_contacts = (float(benchmark["calls_offered"]) / 30.0) * weekday_pressure
            occupancy = min(0.96, 0.72 + month_pressure * 0.08 + account_pressure * 0.04 + (agent_number % 4) * 0.01)
            aht = 405 * skill_factor * account_pressure + seasonality * 5 + month_pressure * 14
            acw = 62 * skill_factor + seasonality * 1.2 + month_pressure * 4
            qa = 93.5 - (skill_factor - 0.94) * 18 - abs(seasonality) * 0.18 - month_pressure * 0.35
            csat = 4.62 - (account_pressure - 0.98) * 0.7 - abs(seasonality) * 0.01 - month_pressure * 0.025
            fcr = 0.84 - (skill_factor - 0.94) * 0.08 - (account_pressure - 0.98) * 0.05
            escalation_rate = 0.055 + (account_pressure - 0.98) * 0.12 + month_pressure * 0.01 + (agent_number % 4) * 0.004
            transfer_rate = 0.085 + (skill_factor - 0.94) * 0.08 + month_pressure * 0.008 + (agent_number % 5) * 0.003
            late_count = max(0.0, 0.2 + (agent_number % 4) * 0.15 + max(seasonality, 0) * 0.08)
            absent_flag = 1 if (offset + agent_number * 3) % 53 == 0 else 0

            # Known incidents make the sample useful for demos and regression tests.
            if team_config["team"] == "Team Manila A" and offset >= 66:
                step = offset - 65
                aht += 16 * step
                acw += 3.8 * step
                transfer_rate += 0.006 * step

            if agent_id == "A007" and offset >= 68:
                step = offset - 67
                qa -= 3.8 * step
                csat -= 0.05 * step

            if team_config["account"] == "Travel" and offset >= 64:
                step = offset - 63
                csat -= 0.08 * step
                fcr -= 0.008 * step
                escalation_rate += 0.009 * step

            if team_config["site"] == "Davao" and offset >= 70:
                step = offset - 69
                late_count += 0.35 * step
                occupancy += 0.015 * step

            rows.append(
                {
                    "date": day.strftime("%Y-%m-%d"),
                    "agent_id": agent_id,
                    "team": team_config["team"],
                    "site": team_config["site"],
                    "account": team_config["account"],
                    "shift": shift,
                    "channel": team_config["channel"],
                    "program": team_config["program"],
                    "tenure_band": tenure_band,
                    "ssa_n8nn_month": benchmark["month"],
                    "ssa_n8nn_calls_offered": int(benchmark["calls_offered"]),
                    "ssa_n8nn_busy_rate_pct": round(float(benchmark["busy_rate_pct"]), 2),
                    "daily_contacts": round(daily_contacts / 48.0 * (0.85 + (agent_number % 6) * 0.06), 0),
                    "occupancy": round(min(0.99, occupancy), 3),
                    "aht": round(aht, 2),
                    "acw": round(acw, 2),
                    "qa": round(max(0.0, qa), 2),
                    "csat": round(max(1.0, csat), 2),
                    "fcr": round(max(0.0, min(1.0, fcr)), 3),
                    "escalation_rate": round(max(0.0, escalation_rate), 3),
                    "transfer_rate": round(max(0.0, transfer_rate), 3),
                    "late_count": round(late_count, 2),
                    "absent_flag": absent_flag,
                }
            )

    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame
