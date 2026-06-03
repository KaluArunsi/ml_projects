# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd


def generate_sample_bpo_kpis(output_path: str | Path = "data/sample_bpo_kpis.csv") -> pd.DataFrame:
    output_path = Path(output_path)
    teams = [
        {"team": "Team Manila A", "site": "Manila", "account": "Telco"},
        {"team": "Team Manila B", "site": "Manila", "account": "Retail"},
        {"team": "Team Cebu C", "site": "Cebu", "account": "Travel"},
    ]
    shifts = ("Day", "Night")
    start_date = datetime(2026, 4, 1)
    rows = []

    for offset in range(60):
        day = start_date + timedelta(days=offset)
        seasonality = (offset % 7) - 3
        for agent_number in range(30):
            team_config = teams[agent_number // 10]
            agent_id = "A{:03d}".format(agent_number + 1)
            shift = shifts[agent_number % 2]
            aht = 430 + (agent_number % 5) * 10 + seasonality * 4
            acw = 68 + (agent_number % 4) * 3 + seasonality
            qa = 92 - (agent_number % 6) * 0.8 - abs(seasonality) * 0.2
            csat = 4.55 - (agent_number % 5) * 0.05 - abs(seasonality) * 0.01
            fcr = 0.82 - (agent_number % 6) * 0.015
            escalation_rate = 0.07 + (agent_number % 5) * 0.01
            transfer_rate = 0.10 + (agent_number % 4) * 0.01
            late_count = 0.4 + (agent_number % 3) * 0.2
            absent_flag = 1 if (offset + agent_number) % 29 == 0 else 0

            if team_config["team"] == "Team Manila A" and offset >= 53:
                step = offset - 52
                aht += 18 * step
                acw += 4 * step
                transfer_rate += 0.008 * step

            if agent_id == "A007" and offset >= 55:
                step = offset - 54
                qa -= 4.5 * step

            if team_config["account"] == "Travel" and offset >= 53:
                step = offset - 52
                csat -= 0.12 * step
                fcr -= 0.01 * step

            if team_config["team"] == "Team Cebu C" and offset >= 50:
                step = offset - 49
                escalation_rate += 0.015 * step

            rows.append(
                {
                    "date": day.strftime("%Y-%m-%d"),
                    "agent_id": agent_id,
                    "team": team_config["team"],
                    "site": team_config["site"],
                    "account": team_config["account"],
                    "shift": shift,
                    "aht": round(aht, 2),
                    "acw": round(acw, 2),
                    "qa": round(max(0.0, qa), 2),
                    "csat": round(max(1.0, csat), 2),
                    "fcr": round(max(0.0, fcr), 3),
                    "escalation_rate": round(escalation_rate, 3),
                    "transfer_rate": round(transfer_rate, 3),
                    "late_count": round(max(0.0, late_count + seasonality * 0.1), 2),
                    "absent_flag": absent_flag,
                }
            )

    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame
