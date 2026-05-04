bike-demand-forecasting/
├── main.py                    # Entry point with argparse
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Data loading & preprocessing
│   ├── feature_engineering.py # Cyclical & lag features
│   ├── model.py               # NN architecture & training
│   ├── evaluation.py          # Metrics & visualization
│   ├── prediction.py          # Next-day predictions
│   └── utils.py               # Seed setting
├── requirements.txt
└── README.md