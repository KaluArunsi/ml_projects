# Credit Worthiness Assessment

A probabilistic loan default classifier that estimates the likelihood a borrower will default on a personal loan, built on LendingClub application data. Every prediction is a complementary pair: `P(cannot_pay_back) + P(can_pay_back) = 1.0`.

## Overview

| Item | Detail |
|---|---|
| Selected model | Logistic Regression |
| Operating threshold T | 0.3588 |
| Recall (default class) | 0.8827 |
| Precision (default class) | 0.2001 |
| ROC-AUC | 0.6758 |
| AUPRG | 0.5677 |
| Missed defaulters (FN) | 36 of 307 |
| Train/test ROC-AUC gap | 0.007 |

**OCC loan classification tiers:**

| P(cannot pay back) | OCC Tier | Underwriting Action |
|---|---|---|
| < 0.10 | Pass | Approve at standard terms |
| 0.10 – 0.25 | Special Mention | Approve with credit monitoring |
| 0.25 – 0.50 | Substandard | Higher rate or reduced credit limit |
| > 0.50 | Doubtful / Loss | Reject or require full collateral |

## Project Structure

```
credit-worthiness/
├── main.py                     Entry point — runs the full pipeline
├── credit_worthiness.ipynb     Full analysis and report notebook (showcase)
├── requirements.txt
├── data/
│   └── loan_data.csv           Raw LendingClub dataset (included)
├── src/
│   ├── config.py               Constants: model params, OCC tiers, column lists
│   ├── data_loader.py          CSV loader
│   ├── outliers.py             LOF and Mahalanobis flagging
│   ├── preprocessing.py        Feature engineering, IQR cap, K-Means, encode, split, SMOTE-ENN
│   ├── models.py               LR (selected), RF, XGBoost, LightGBM, PRG curve, threshold finder
│   └── evaluation.py           All 25 plot functions
├── output/                     Generated images (25 plots)
└── MODEL_OUTPUTS.md            Visual flow guide — all 25 output images
```

## Installation

```bash
git clone <repo-url>
cd credit-worthiness

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

**Terminal pipeline** (primary):

```bash
python main.py                        # full run — prints results and saves 25 images to output/
python main.py --no-images            # console output only, no image files written
python main.py --output-dir results/  # save images to a custom directory
```

**Notebook** (showcase and analysis walkthrough):

```bash
jupyter lab credit_worthiness.ipynb
```

Run all cells top to bottom. The notebook covers data loading, feature engineering, EDA, preprocessing, SMOTE-ENN resampling, four model comparisons with Optuna tuning, vintage analysis across OCC FICO risk tiers, model selection, and the repayment probability formula.

## Outputs

**Console** (always printed):
- Dataset shape and class distribution
- DSCR and regulatory flag summary
- LOF and Mahalanobis flagged counts
- Cluster default rates per segment
- SMOTE-ENN before/after class counts
- Per-model: classification report, ROC-AUC, AUPRG, operating threshold
- Vintage analysis: recall per OCC FICO tier cohort per model
- Model selection table ranked by recall at threshold
- Sample borrower assessment table with OCC tier and underwriting action

**Images** (saved to `output/`):

| File | Content |
|---|---|
| `01_class_distribution.png` | Class balance — 84% repaid, 16% defaulted |
| `02_univariate_numeric.png` | Histograms + KDE for all numeric features |
| `03_categorical_analysis.png` | Purpose counts, default rate by purpose, OCC tier distribution |
| `04_correlation_heatmap.png` | Full feature correlation matrix |
| `05_boxplots_by_target.png` | Feature distributions split by default outcome |
| `06_violin_capacity.png` | Violin plots for DSCR, DTI, and capacity features by target |
| `07_purpose_stacked.png` | Default rate stacked bar by loan purpose |
| `08_pairplot.png` | Pair plot of top 5 discriminating features |
| `09_lof_outliers.png` | LOF outlier scores in PCA 2D space |
| `10_mahalanobis.png` | Mahalanobis distance distribution + chi² threshold |
| `11_cluster_selection.png` | Elbow + silhouette score for k selection |
| `12_cluster_pca.png` | K-Means clusters and target overlay in PCA space |
| `13_cluster_profiles.png` | Cluster mean feature values and default rate per cluster |
| `14_smoteenn.png` | Class distribution before and after SMOTE-ENN |
| `15_lr_eval.png` | LR: confusion matrix, ROC, PR, PRG curves |
| `16_lr_coefficients.png` | LR: top 20 coefficients (log-odds scale) |
| `17_rf_eval.png` | RF: confusion matrix, ROC, PR, PRG curves |
| `18_rf_importance.png` | RF: feature importances (Gini) |
| `19_xgb_eval.png` | XGBoost: confusion matrix, ROC, PR, PRG curves |
| `20_xgb_importance.png` | XGBoost: feature importances (gain) |
| `21_lgbm_eval.png` | LightGBM: confusion matrix, ROC, PR, PRG curves |
| `22_lgbm_importance.png` | LightGBM: feature importances (gain) |
| `23_vintage_analysis.png` | Recall and AUPRG across OCC FICO tier cohorts per model |
| `24_model_selection.png` | PRG, ROC, and recall bar comparison across all four models |
| `25_repayment_formula.png` | P(cannot pay) distribution, OCC tier breakdown, threshold sweep |

## Pipeline

```
Raw data (9,578 loans, 14 features)
        |
        v
Feature Engineering
    monthly_income, payment_to_income_ratio, dscr_proxy,
    debt_burden_annual, dti_flag, high_utilization_flag,
    delinquency_flag, public_record_flag, occ_tier (OCC/FICO)
        |
        v
EDA
    Univariate → Multivariate → LOF + Mahalanobis → K-Means (k=3)
        |
        v
Preprocessing
    One-hot encode purpose (drop_first)
    Drop: occ_tier, log.annual.inc, lof_flag, mahal_flag
    IQR cap: installment, days.with.cr.line, revol.bal,
             inq.last.6mths, monthly_income
        |
        v
Train/test split (80/20 stratified) → StandardScaler (fit on train)
        |
        +---> SMOTE-ENN (training set only)
        |           |
        |           v
        |     Logistic Regression  class_weight='balanced'
        |     Optuna 50 trials · 5-fold CV on resampled train
        |
        +---> Random Forest        class_weight='balanced_subsample'
        |     XGBoost              scale_pos_weight = neg/pos
        |     LightGBM             is_unbalance=True
        |     Optuna 50 trials · 5-fold CV on original train
        |
        v
Vintage Analysis — recall + AUPRG per OCC FICO tier cohort
        |
        v
Model Selection
    Highest T where recall(class 1) >= 0.88 on test set
        |
        v
SELECTED: Logistic Regression
    T = 0.3588 · recall = 0.8827 · 36 missed defaulters
        |
        v
P(cannot_pay_back) + P(can_pay_back) = 1.0
OCC tier assigned per borrower from raw probability
```

## Dataset

`data/loan_data.csv` is included in the repository. Source: LendingClub loan data (public).

- 9,578 personal loans
- 14 features per application
- Target: `not.fully.paid` → `cannot_pay_back` (1 = defaulted, 0 = repaid)
- Class split: 84% repaid / 16% defaulted

## Requirements

See [requirements.txt](requirements.txt). Python 3.9+.
