# Seoul Bike Demand Forecasting

A practical, production-oriented machine learning project for hourly bike demand prediction.

## Project Philosophy

This project prioritizes **speed, interpretability, and deployability** over exhaustive experimentation. It reflects a realistic business use case where operational relevance matters more than academic novelty.

## Features

- Cyclical time encoding (hour of day, day of year)
- Lag-based demand features for short-term momentum
- Rolling statistics for smoothing and volatility capture
- Neural network architecture optimized for stability
- Next-day hourly predictions with confidence intervals

## Project Structure

```
.
├── main.py                    # Entry point
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Data loading and preprocessing
│   ├── feature_engineering.py # Feature creation
│   ├── model.py               # Model architecture and training
│   ├── evaluation.py          # Evaluation and visualization
│   ├── prediction.py          # Future predictions
│   └── utils.py               # Utility functions
├── models/                    # Saved models (created automatically)
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train and Predict (default)
```bash
python main.py
```

### Train Only
```bash
python main.py --mode train
```

### Predict Only (requires trained model)
```bash
python main.py --mode predict
```

### Command Line Options

- `--mode {train,predict,both}` - Operation mode (default: both)
- `--quiet` - Suppress verbose output
- `--no-plots` - Skip generating plots
- `--epochs N` - Number of training epochs (default: 500)
- `--batch-size N` - Training batch size (default: 1024)
- `--model-path PATH` - Path to save/load model (default: models/bike_model.keras)
- `--data-url URL` - Data source URL

### Examples

```bash
# Quick run without plots
python main.py --quiet --no-plots

# Custom training configuration
python main.py --epochs 300 --batch-size 512

# Just make predictions
python main.py --mode predict --quiet
```

## Outputs

- **Model**: Saved to `models/bike_model.keras`
- **Predictions**: Saved to `predictions.csv` with confidence intervals
- **Training Plot**: Saved to `training_history.png` (unless `--no-plots`)

## Model Performance

The model achieves high R² scores driven by strong temporal structure in Seoul bike usage patterns. Performance is highly dependent on lag features capturing short-term demand persistence.

## Business Context

This forecasting system is designed for:
- Operational bike redistribution planning
- Staffing and maintenance scheduling
- Real-time demand response

The MAE-based confidence intervals provide practical bounds for decision-making under uncertainty.