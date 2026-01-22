"""
Seoul Bike Demand Forecasting
Main entry point for training and prediction
"""
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

from src.data_loader import load_and_prepare_data
from src.feature_engineering import engineer_features
from src.model import build_model, train_model
from src.evaluation import evaluate_model, plot_training_history
from src.prediction import predict_next_day
from src.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Seoul Bike Demand Forecasting')
    parser.add_argument('--mode', choices=['train', 'predict', 'both'], default='both',
                        help='Operation mode: train, predict, or both')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='Training batch size')
    parser.add_argument('--model-path', type=str, default='models/bike_model.keras',
                        help='Path to save/load model')
    parser.add_argument('--data-url', type=str,
                        default='https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip',
                        help='Data source URL')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)
    
    verbose = not args.quiet
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("Loading and preparing data...")
    
    bike_data = load_and_prepare_data(args.data_url)
    bike_data = engineer_features(bike_data)
    
    # Prepare features and split
    feature_cols = [
        'hour', 'temperature(°c)', 'humidity(%)', 'wind_speed_(m/s)', 
        'visibility_(10m)', 'dew_point_temperature(°c)', 'solar_radiation_(mj/m2)',
        'rainfall(mm)', 'snowfall_(cm)', 'date_sin', 'date_cos', 'hour_sin',
        'hour_cos', 'temp_hour', 'rbc_1', 'rbc_2', 'rbc_3', 
        'rbc_rolling_mean', 'rbc_rolling_std', 'rbc_trend'
    ]
    target = 'rented_bike_count'
    
    bike_data_num = bike_data.select_dtypes(include=np.number)
    split = int(len(bike_data_num) * 0.7)
    
    feature_train = bike_data_num[feature_cols].iloc[:split]
    feature_test = bike_data_num[feature_cols].iloc[split:]
    label_train = bike_data_num[target].iloc[:split]
    label_test = bike_data_num[target].iloc[split:]
    
    # Training
    if args.mode in ['train', 'both']:
        if verbose:
            print(f"\nTraining model for {args.epochs} epochs...")
        
        model = build_model(len(feature_cols))
        history = train_model(
            model, feature_train, label_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=verbose
        )
        
        model.save(model_path)
        if verbose:
            print(f"Model saved to {model_path}")
        
        # Evaluation
        mae = evaluate_model(model, feature_test, label_test, verbose=verbose)
        
        if not args.no_plots:
            plot_training_history(history)
    
    # Prediction
    if args.mode in ['predict', 'both']:
        if args.mode == 'predict':
            from keras.models import load_model
            model = load_model(model_path)
            mae = evaluate_model(model, feature_test, label_test, verbose=False)
        
        if verbose:
            print("\nGenerating next day predictions...")
        
        predictions = predict_next_day(bike_data, model, feature_cols, mae)
        
        print(f"\n{'='*5} Next Day Hourly Bike Demand Predictions {'='*5}")
        print(predictions.to_string(index=False))
        
        # Save predictions
        predictions.to_csv('predictions.csv', index=False)
        if verbose:
            print("\nPredictions saved to predictions.csv")


if __name__ == '__main__':
    main()