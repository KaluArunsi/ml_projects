# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable

import pandas as pd


def _source_name(source, filename: str | None = None) -> str:
    if filename:
        return filename
    if isinstance(source, (str, Path)):
        return str(source)
    return getattr(source, "name", "")


def _rewind(source):
    if hasattr(source, "seek"):
        source.seek(0)
    return source


def _to_excel_source(source):
    if isinstance(source, (str, Path)):
        return source
    if isinstance(source, BytesIO):
        source.seek(0)
        return source
    if hasattr(source, "read"):
        return _rewind(source)
    return BytesIO(Path(source).read_bytes())


def load_csv(file) -> pd.DataFrame:
    _rewind(file)
    return pd.read_csv(file)


def load_excel(file, sheet_name: str | int = 0) -> pd.DataFrame:
    return pd.read_excel(_to_excel_source(file), sheet_name=sheet_name)


def list_excel_sheets(file) -> Iterable[str]:
    workbook = pd.ExcelFile(_to_excel_source(file))
    return workbook.sheet_names


def load_tabular_file(file, sheet_name: str | int = 0, filename: str | None = None) -> pd.DataFrame:
    suffix = Path(_source_name(file, filename)).suffix.lower()
    if suffix == ".csv":
        return load_csv(file)
    if suffix in {".xlsx", ".xls"}:
        return load_excel(file, sheet_name=sheet_name)
    raise ValueError("Unsupported file type: {}".format(suffix or "unknown"))
