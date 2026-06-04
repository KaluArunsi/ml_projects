from __future__ import annotations

from pathlib import Path
import sys
import importlib.util


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA_MODULE = PROJECT_ROOT / "src" / "sample_data.py"

spec = importlib.util.spec_from_file_location("openbpo_sample_data", SAMPLE_DATA_MODULE)
if spec is None or spec.loader is None:
    raise ImportError("Could not load openbpo_sample_data from {}; spec={!r}, loader={!r}".format(SAMPLE_DATA_MODULE, spec, getattr(spec, "loader", None)))

sample_data = importlib.util.module_from_spec(spec)
sys.modules["openbpo_sample_data"] = sample_data
spec.loader.exec_module(sample_data)
generate_sample_bpo_kpis = sample_data.generate_sample_bpo_kpis


OUTPUT_PATH = PROJECT_ROOT / "data" / "sample_bpo_kpis.csv"


def main() -> None:
    generate_sample_bpo_kpis(OUTPUT_PATH)


if __name__ == "__main__":
    main()
