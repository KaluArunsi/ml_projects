from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sample_data import generate_sample_bpo_kpis


OUTPUT_PATH = PROJECT_ROOT / "data" / "sample_bpo_kpis.csv"


def main() -> None:
    generate_sample_bpo_kpis(OUTPUT_PATH)


if __name__ == "__main__":
    main()
