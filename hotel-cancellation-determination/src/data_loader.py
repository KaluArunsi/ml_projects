import os
import pandas as pd

_DEFAULT_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hotel_bookings.csv")


def load_dataset(csv_path: str = _DEFAULT_CSV) -> pd.DataFrame:
    """Load the hotel bookings CSV. Defaults to hotel_bookings.csv in the project root."""
    return pd.read_csv(csv_path)
