"""
Feature engineering for bike demand prediction
"""
import numpy as np
import pandas as pd


def engineer_features(bike_data):
    """
    Create cyclical time features and lag-based demand features
    
    Cyclical encoding preserves temporal relationships (e.g., hour 23 is close to hour 0)
    Lag features capture short-term momentum and recent usage behavior
    """
    # Cyclical encoding for day of year
    days_in_year = bike_data['date'].dt.is_leap_year.map({True: 366, False: 365})
    date_theta = 2 * np.pi * (bike_data['date'].dt.day_of_year - 1) / days_in_year
    bike_data['date_sin'] = np.sin(date_theta)
    bike_data['date_cos'] = np.cos(date_theta)
    
    # Cyclical encoding for hour of day
    hours_theta = 2 * np.pi * bike_data['hour'] / 24
    bike_data['hour_sin'] = np.sin(hours_theta)
    bike_data['hour_cos'] = np.cos(hours_theta)
    
    # Temperature-hour interaction
    bike_data['temp_hour'] = bike_data['temperature(°c)'] / 24
    
    # Lag features (short-term momentum)
    bike_data['rbc_1'] = bike_data['rented_bike_count'].shift(1)
    bike_data['rbc_2'] = bike_data['rented_bike_count'].shift(2)
    bike_data['rbc_3'] = bike_data['rented_bike_count'].shift(3)
    
    # Rolling statistics (smoothing and volatility)
    bike_data['rbc_rolling_mean'] = bike_data['rented_bike_count'].shift(1).rolling(window=7).mean()
    bike_data['rbc_rolling_std'] = bike_data['rented_bike_count'].shift(1).rolling(window=7).std()
    
    # Trend indicator
    bike_data['rbc_trend'] = bike_data['rented_bike_count'].rolling(3).apply(
        lambda x: x.iloc[-1] - x.iloc[0]
    )
    
    # Drop NaN rows created by lagging
    bike_data = bike_data.dropna().reset_index(drop=True)
    
    return bike_data