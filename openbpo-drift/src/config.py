# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import yaml


DEFAULT_BASELINE_WINDOW = 14
DEFAULT_CURRENT_WINDOW = 3
DEFAULT_THRESHOLD_PCT = 15.0


def load_kpi_rules(config_path: Path) -> Dict[str, Dict[str, object]]:
    payload = yaml.safe_load(config_path.read_text()) or {}
    return payload if "kpis" not in payload else payload["kpis"]


def merge_kpi_rules(
    default_rules: Dict[str, Dict[str, object]],
    kpi_mapping: List[Dict[str, object]],
    default_threshold_pct: float = DEFAULT_THRESHOLD_PCT,
) -> Dict[str, Dict[str, object]]:
    merged: Dict[str, Dict[str, object]] = {}
    for index, item in enumerate(kpi_mapping):
        if not item.get("include", True):
            continue
        kpi_raw = item.get("kpi_name")
        if not kpi_raw or not str(kpi_raw).strip():
            raise ValueError("Missing kpi_name in KPI mapping item {}: {!r}".format(index, item))
        kpi_name = str(kpi_raw).strip().lower()
        rule = deepcopy(default_rules.get(kpi_name, {}))
        rule["label"] = rule.get("label", kpi_name.replace("_", " ").title())
        rule["direction_bad"] = item.get("direction_bad") or rule.get("direction_bad", "up")
        rule["unit"] = item.get("unit") or rule.get("unit", "")
        threshold = item.get("drift_threshold_pct") or rule.get("drift_threshold_pct", default_threshold_pct)
        try:
            threshold_pct = float(threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid drift_threshold_pct for KPI {!r}: {!r}".format(kpi_name, threshold)) from exc
        if threshold_pct <= 0 or threshold_pct > 100:
            raise ValueError("drift_threshold_pct for KPI {!r} must be > 0 and <= 100: {!r}".format(kpi_name, threshold_pct))
        rule["drift_threshold_pct"] = threshold_pct
        rule["drivers"] = list(rule.get("drivers", []))
        rule["validation"] = dict(rule.get("validation", {}))
        merged[kpi_name] = rule
    return merged
