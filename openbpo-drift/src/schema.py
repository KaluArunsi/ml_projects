# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from typing import Dict, Iterable, List


CANONICAL_COLUMNS = [
    "date",
    "entity_id",
    "entity_type",
    "team",
    "site",
    "account",
    "shift",
    "channel",
    "program",
    "kpi_name",
    "kpi_value",
    "unit",
    "source_column",
]

REQUIRED_FIELDS = ["date", "entity_id"]
OPTIONAL_FIELDS = ["team", "site", "account", "shift", "channel", "program"]
DEFAULT_ENTITY_TYPE = "agent"

FIELD_LABELS = {
    "date": "Date",
    "entity_id": "Entity ID",
    "team": "Team",
    "site": "Site",
    "account": "Account",
    "shift": "Shift",
    "channel": "Channel",
    "program": "Program",
}

FIELD_ALIASES = {
    "date": ("date", "day", "report_date"),
    "entity_id": ("agent_id", "agent", "entity_id", "advisor_id", "employee_id"),
    "team": ("team", "squad"),
    "site": ("site", "location", "center"),
    "account": ("account", "client", "program", "campaign"),
    "shift": ("shift", "schedule"),
    "channel": ("channel", "contact_channel", "queue_channel"),
    "program": ("program", "lob", "line_of_business"),
}


def guess_field_mapping(columns: Iterable[str]) -> Dict[str, str | None]:
    available = {column.lower(): column for column in columns}
    mapping: Dict[str, str | None] = {}
    for field, aliases in FIELD_ALIASES.items():
        mapping[field] = next((available[alias] for alias in aliases if alias in available), None)
    return mapping


def infer_kpi_candidates(columns: Iterable[str], mapped_fields: Dict[str, str | None]) -> List[str]:
    excluded = {value for value in mapped_fields.values() if value}
    candidates = []
    for column in columns:
        if column in excluded:
            continue
        lowered = column.lower()
        if any(token in lowered for token in ("aht", "acw", "qa", "csat", "fcr", "rate", "count", "absent", "score")):
            candidates.append(column)
    return candidates
