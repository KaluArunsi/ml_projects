"""
Future predictions for next day hourly demand
"""
import numpy as np
import pandas as pd
from datetime import timedelta


def predict_next_day(bike_data, model, feature_cols, mae):
    """
    Predict hourly bike demand for the next day
    
    Uses historical hour-based averages for weather features
    and lagged demand values to generate predictions
    
    Returns DataFrame with predictions and confidence intervals
    """
    bike_data_copy = bike_data.copy()
    bike_data_copy['rbc_1'] = bike_data_copy['rented_bike_count'].shift(1)
    
    # Generate features for next 24 hours
    hour_list, date_list = [], []
    temp, hum, wind, vis, dew, solar, rain, snow, rbc_1 = [], [], [], [], [], [], [], [], []
    
    next_date = bike_data_copy['date'].iloc[-1] + timedelta(1)
    
    for i in range(24):
        hour_list.append(i)
        date_list.append(next_date)
        
        # Use historical hour-based averages
        agg_data = bike_data_copy[bike_data_copy['hour'] == i]
        
        temp.append(agg_data['temperature(°c)'].mean())
        hum.append(agg_data['humidity(%)'].mean())
        wind.append(agg_data['wind_speed_(m/s)'].mean())
        vis.append(agg_data['visibility_(10m)'].mean())
        dew.append(agg_data['dew_point_temperature(°c)'].mean())
        solar.append(agg_data['solar_radiation_(mj/m2)'].mean())
        rain.append(agg_data['rainfall(mm)'].mean())
        snow.append(agg_data['snowfall_(cm)'].mean())
        rbc_1.append(agg_data['rbc_1'].mean())
    
    # Create feature DataFrame
    feats = pd.DataFrame({
        'hour': hour_list, 'date': date_list,
        'temperature(°c)': temp, 'humidity(%)': hum,
        'wind_speed_(m/s)': wind, 'visibility_(10m)': vis,
        'dew_point_temperature(°c)': dew, 'solar_radiation_(mj/m2)': solar,
        'rainfall(mm)': rain, 'snowfall_(cm)': snow,
        'rbc_1': rbc_1
    })
    
    bike_data_copy = pd.concat([bike_data_copy, feats], ignore_index=True)
    
    # Engineer lag and rolling features
    bike_data_copy['rbc_2'] = bike_data_copy['rbc_1'].shift(1)
    bike_data_copy['rbc_3'] = bike_data_copy['rbc_1'].shift(2)
    bike_data_copy['rbc_rolling_mean'] = bike_data_copy['rbc_1'].shift(1).rolling(window=7).mean()
    bike_data_copy['rbc_rolling_std'] = bike_data_copy['rbc_1'].shift(1).rolling(window=7).std()
    bike_data_copy['rbc_trend'] = bike_data_copy['rbc_1'].rolling(3).apply(
        lambda x: x.iloc[-1] - x.iloc[0]
    )
    
    # Engineer cyclical features
    day_in_year = bike_data_copy['date'].dt.is_leap_year.map({True: 366, False: 365})
    dt_theta = 2 * np.pi * (bike_data_copy['date'].dt.day_of_year - 1) / day_in_year
    hr_theta = 2 * np.pi * bike_data_copy['hour'] / 24
    
    bike_data_copy['date_sin'] = np.sin(dt_theta)
    bike_data_copy['date_cos'] = np.cos(dt_theta)
    bike_data_copy['hour_sin'] = np.sin(hr_theta)
    bike_data_copy['hour_cos'] = np.cos(hr_theta)
    bike_data_copy['temp_hour'] = bike_data_copy['temperature(°c)'] / 24
    
    # Extract prediction features
    pred_data = bike_data_copy[feature_cols].iloc[-24:]
    
    # Make predictions
    predictions = model.predict(pred_data, verbose=0).flatten().round(0)
    hours = bike_data_copy['hour'].iloc[-24:].values
    
    # Create results with confidence intervals
    results = pd.DataFrame({
        'hour': hours,
        'predictions': predictions,
        'lower_limit': (predictions - mae).round(0),
        'upper_limit': (predictions + mae).round(0)
    })
    
    return results