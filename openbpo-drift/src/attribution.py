# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from typing import Dict


def suggested_check(rule: Dict[str, object]) -> str:
    drivers = rule.get("drivers", [])
    if drivers:
        return "Review supporting KPIs: {}.".format(", ".join(str(driver).replace("_", " ") for driver in drivers))
    return "Review recent workflow, staffing, and process changes affecting this KPI."


def explain_alert(row: Dict[str, object]) -> str:
    direction_word = "increased" if float(row["drift_pct"]) > 0 else "decreased"
    configured_direction = "higher" if row["direction_bad"] == "up" else "lower"
    return (
        "{entity_id}'s {kpi_name} {direction_word} by {drift:.1f}% versus baseline. "
        "Since {configured_direction} {kpi_name} is configured as bad, this has been flagged as a {severity} severity drift. "
        "{suggested_check}"
    ).format(
        entity_id=row["entity_id"],
        kpi_name=row["kpi_name"],
        direction_word=direction_word,
        drift=abs(float(row["drift_pct"])),
        configured_direction=configured_direction,
        severity=row["severity"],
        suggested_check=row["suggested_check"],
    )
