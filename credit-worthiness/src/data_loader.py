import os
import pandas as pd

_DEFAULT_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "loan_data.csv")


def load_dataset(csv_path: str = _DEFAULT_CSV) -> pd.DataFrame:
    return pd.read_csv(csv_path)
