"""
Data loading and initial preprocessing
"""
import pandas as pd


def load_and_prepare_data(url):
    """Load bike sharing data and perform initial preprocessing"""
    bike_data = pd.read_csv(url, encoding='latin1')
    
    # Standardize column names
    bike_data.columns = bike_data.columns.str.replace(' ', '_').str.lower()
    
    # Parse dates
    bike_data['date'] = pd.to_datetime(bike_data['date'], errors='coerce')
    bike_data = bike_data[~bike_data['date'].isnull()]
    
    # Encode categorical variables
    bike_data['seasons'] = bike_data['seasons'].map({
        'Spring': 1, 'Summer': 2, 'Autumn': 3, 'Winter': 4
    })
    bike_data['holiday'] = bike_data['holiday'].map({'Holiday': 1, 'No Holiday': 0})
    bike_data['functioning_day'] = bike_data['functioning_day'].map({'Yes': 1, 'No': 0})
    
    # Drop categorical columns (preference for engineered time-based signals)
    bike_data = bike_data.drop(columns=['seasons', 'holiday', 'functioning_day'])
    
    return bike_data