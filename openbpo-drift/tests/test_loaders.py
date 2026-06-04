from io import BytesIO

import pandas as pd
import pytest

from src.loaders import list_excel_sheets, load_excel


def _excel_bytes() -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        pd.DataFrame({"a": [1]}).to_excel(writer, sheet_name="Ops", index=False)
    return buffer.getvalue()


def test_excel_loader_accepts_bytes_and_closes_workbook():
    payload = _excel_bytes()

    assert list(list_excel_sheets(payload)) == ["Ops"]
    assert load_excel(payload, sheet_name="Ops").to_dict("records") == [{"a": 1}]


def test_excel_loader_rejects_unsupported_source_type():
    with pytest.raises(TypeError, match="Unsupported source type"):
        load_excel(object())
