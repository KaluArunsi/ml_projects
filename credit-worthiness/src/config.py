RANDOM_STATE = 42
TARGET_RECALL = 0.88

# Selected model: Logistic Regression
LR_C = 0.001022387163625499
LR_SOLVER = "liblinear"
LR_MAX_ITER = 1000
BEST_THRESHOLD = 0.3588

# Best Optuna params for tree models
RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 3,
    "min_samples_split": 15,
    "min_samples_leaf": 12,
    "max_features": "log2",
}
XGB_PARAMS = {
    "n_estimators": 350,
    "max_depth": 3,
    "learning_rate": 0.013722315372926544,
    "subsample": 0.6128667295254544,
    "colsample_bytree": 0.9275133114228716,
    "min_child_weight": 16,
    "gamma": 0.7143146560862477,
    "reg_alpha": 0.11207660676293843,
    "reg_lambda": 4.206759252961867,
}
LGBM_PARAMS = {
    "n_estimators": 250,
    "learning_rate": 0.016754639724525955,
    "num_leaves": 100,
    "max_depth": 6,
    "min_child_samples": 199,
    "subsample": 0.6961682039068637,
    "colsample_bytree": 0.9477392416585152,
    "reg_alpha": 0.9667890326770116,
    "reg_lambda": 0.66157528995131,
}

# OCC FICO tier mapping
FICO_BINS = [300, 620, 660, 720, 851]
FICO_LABELS = ["Doubtful/Loss", "Substandard", "Special Mention", "Pass"]

# OCC probability tier boundaries (P(cannot pay back))
OCC_PROB_BINS = [0.0, 0.10, 0.25, 0.50, 1.01]
OCC_PROB_LABELS = ["Pass", "Special Mention", "Substandard", "Doubtful/Loss"]
OCC_PROB_COLORS = ["#27ae60", "#f39c12", "#e67e22", "#e74c3c"]
OCC_PROB_ACTIONS = [
    "Approve at standard terms",
    "Approve with credit monitoring",
    "Higher rate or reduced credit limit",
    "Reject or require full collateral",
]

# IQR capping columns
IQR_CAP_COLS = ["installment", "days.with.cr.line", "revol.bal", "inq.last.6mths", "monthly_income"]

# Features used for K-Means clustering
CLUSTER_FEATURES = [
    "fico", "int.rate", "dti", "monthly_income",
    "payment_to_income_ratio", "revol.util", "days.with.cr.line", "inq.last.6mths",
]

# Features used for LOF and Mahalanobis outlier detection
OUTLIER_COLS = [
    "fico", "int.rate", "installment", "dti", "days.with.cr.line",
    "revol.bal", "revol.util", "inq.last.6mths", "monthly_income",
    "payment_to_income_ratio", "dscr_proxy",
]

# Columns dropped before modeling
COLS_TO_DROP = ["occ_tier", "log.annual.inc", "lof_flag", "mahal_flag"]

N_CLUSTERS = 3
