# Hotel Booking Cancellation Risk Assessment

A two-model machine learning system that predicts hotel booking cancellations and assigns every booking a severity tier so the revenue team knows exactly how urgently to act.

## Overview

| Model | Type | Output |
|-------|------|--------|
| Logistic Regression | Binary classifier | Will this booking cancel? (yes/no + probability) |
| XGBoost | Multiclass classifier | How severe is the cancellation risk? (Tier 0-3) |

**Severity tiers:**

| Tier | Label | Response |
|------|-------|----------|
| 0 | Very Unlikely | No action |
| 1 | Unlikely | Monitor, re-confirm closer to date |
| 2 | Likely | Proactive offer: upgrade, discount, freebie |
| 3 | Most Likely | Immediate retention call |

## Project Structure

```
hotel-cancellation-determination/
├── main.py                    # Entry point
├── src/
│   ├── __init__.py
│   ├── config.py              Constants (severity bins, column lists)
│   ├── data_loader.py         Load hotel_bookings.csv from project root
│   ├── preprocessing.py       Cleaning, IQR capping, encoding
│   ├── outliers.py            Z-score/MAD and Mahalanobis outlier removal
│   ├── models.py              LR + XGBoost training, severity label creation
│   └── evaluation.py          All plotting and evaluation functions
├── output/                    Saved images (created automatically)
├── hotel-cancellation.ipynb   Full report notebook
├── hotel_bookings.csv         Raw dataset (included)
├── requirements.txt
└── MODEL_OUTPUTS.md           Visual flow guide + output image reference
```

## Installation

```bash
git clone <repo-url>
cd hotel-cancellation-determination

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

### Full run with images (default)
```bash
python main.py
```
Runs the complete pipeline and saves 9 images to `output/`.

### Print results only, no images
```bash
python main.py --no-images
```

### Custom output directory
```bash
python main.py --output-dir results/
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|kagglehub[pandas-datasets]
| `--no-images` | off | Skip image generation, print results only |
| `--output-dir DIR` | `output/` | Directory to save generated images |

## Outputs

**Console** (always printed):
- Dataset shape and preprocessing summary
- Rows removed during outlier cleaning
- Logistic Regression: ROC-AUC, F1, classification report
- Severity tier counts with recommended response per tier
- XGBoost: Macro ROC-AUC, Macro F1, per-tier classification report

**Images** (saved to `output/` unless `--no-images`):

| File | Content |
|------|---------|
| `01_class_distribution.png` | Cancelled vs not cancelled counts and split |
| `02_feature_distributions.png` | Histograms for 10 key numerical features |
| `03_lr_evaluation.png` | Confusion matrix, ROC curve, PR curve, per-class metrics |
| `04_lr_learning_curve.png` | Train vs CV F1 across training set sizes |
| `05_severity_distribution.png` | P(cancel) histogram with tier boundaries + bookings per tier |
| `06_xgb_evaluation.png` | 4x4 confusion matrix, per-tier metrics, top feature importances |
| `07_xgb_confidence.png` | Confidence score distributions per severity tier |
| `08_xgb_pr_curves.png` | One-vs-rest PR curves for all 4 tiers + average precision scores |
| `09_xgb_learning_curve.png` | XGBoost train vs CV F1 macro across training set sizes |

## Notebook

For a full narrative walkthrough of the project:
```bash
jupyter lab hotel-cancellation.ipynb
```
Run all cells top to bottom. The notebook covers EDA, preprocessing, outlier analysis, both models, and a results summary.

## Pipeline

```
Raw data (119,390 bookings)
    |
    v
Data cleaning (fill NAs, drop zero-ADR rows)
    |
    v
IQR outlier capping (lead_time, adr)
    |
    v
Feature encoding (frequency + one-hot)
    |
    v
Univariate outlier removal (Z-score AND Modified Z-score)
    |
    v
Multivariate outlier removal (Mahalanobis distance, p=0.999)
    |
    v
Train/test split (80/20 stratified) + StandardScaler
    |
    +---> Logistic Regression (binary: cancel/no cancel)
    |         |
    |         v
    |     ROC-AUC, F1, PR curve, Learning curve
    |         |
    |         +-> predict_proba -> P(cancel) per booking
    |                                  |
    |                    create_severity_labels() (5-fold OOF)
    |                                  |
    +---> XGBoost (multiclass: severity tier 0-3)
              |
              v
          Macro ROC-AUC, per-tier F1, feature importance,
          PR curves, confidence distributions, learning curve
```

## Dataset

`hotel_bookings.csv` is included in the repository. Original source: [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) by Jesse Mostipak, sourced from the paper *Hotel booking demand datasets* (Antonio, Almeida, Nunes, 2019).

- 119,390 bookings across two hotel types
- 32 features per booking
- Target: `is_canceled` (binary)

## Requirements

See [requirements.txt](requirements.txt). Python 3.9+.
