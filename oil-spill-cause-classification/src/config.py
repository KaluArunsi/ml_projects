"""
Oil Spill Cause Classification — Central Configuration
======================================================
All tunable parameters for data loading, preprocessing, three modeling tracks
(TF-IDF baseline, DistilBERT, Phi-3.5-mini MLX LoRA), evaluation, and reporting.

Data sources (CC BY 4.0 — figshare companion to the Nature Scientific Data
paper "An enhanced global oil spill dataset from 1967 to 2023 based on
text-form incident information"; underlying source: NOAA IncidentNews).
"""

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Primary data sources
# ---------------------------------------------------------------------------
INCIDENTS_FILENAME = "incidents_20231207.csv"
POSTS_FILENAME = "posts_raw_all.xlsx"

# figshare direct download URLs (article 26130892, v2).
INCIDENTS_URL = "https://ndownloader.figshare.com/files/47325709"
POSTS_URL = "https://ndownloader.figshare.com/files/47325727"

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
DATA_EXTERNAL_DIR = "data/external"
LABEL_MAPS_DIR = "data/label_maps"
MODELS_DIR = "models"
OUTPUT_DIR = "output"

# ---------------------------------------------------------------------------
# Label configuration
# ---------------------------------------------------------------------------
# Cause tags are pipe-separated multi-labels (e.g. "Coral|Grounding"). We keep
# every cause seen at least MIN_LABEL_COUNT times across the labeled corpus;
# rarer causes collapse into the residual "Other" label.
LABEL_COL = "tags"
MIN_LABEL_COUNT = 10
OTHER_LABEL = "Other"

# Master label taxonomy (order matches the binary matrix columns).
# Historic Wreck (count=5) folds into Other per MIN_LABEL_COUNT.
LABEL_NAMES = [
    "Grounding", "Collision", "Pipeline", "Mystery Substance",
    "Hurricane", "Derelict", "Wellhead", "Adrift",
    "Search + Rescue", "Railcar", "Coral", "Marine Debris",
    "Marine Mammal", "Tsunami", "Other",
]
NUM_LABELS = len(LABEL_NAMES)

# ---------------------------------------------------------------------------
# Text assembly
# ---------------------------------------------------------------------------
# Each incident's document = its own description plus the concatenated content
# of every post that belongs to it (joined on NOAA incident id). Posts inherit
# their incident's label, so they are concatenated INTO the incident document
# rather than treated as independent rows — this expands text per label without
# leaking the label across a train/test split.
INCIDENT_ID_COL = "id"
POST_INCIDENT_ID_COL = "noaa id"
POST_CONTENT_COL = "post content"
POST_TITLE_COL = "post title"
DESCRIPTION_COL = "description"
MAX_POSTS_PER_INCIDENT = 12          # cap runaway incidents (max observed: 380)
MAX_DOCUMENT_WORDS = 1500            # word-level cap for extreme outliers

# ---------------------------------------------------------------------------
# Commodity normalization
# ---------------------------------------------------------------------------
# 1,555 unique raw commodity values → ~20 canonical categories.
# Keys are canonical names; values are lists of raw variants (lowercased).
COMMODITY_CATEGORIES = {
    "crude oil": [
        "crude oil", "crude", "oil, crude", "crude petroleum",
        "crude, oil", "petroleum crude", "light crude", "heavy crude",
    ],
    "diesel": [
        "diesel", "diesel fuel", "diesel oil", "marine diesel",
        "#2 diesel", "#2 fuel oil", "diesel / #2 fuel oil",
        "no. 2 fuel oil", "number 2 fuel oil",
    ],
    "gasoline": [
        "gasoline", "gas", "unleaded gasoline", "petrol",
        "automotive gasoline", "unleaded gas",
    ],
    "heavy fuel oil": [
        "#6 fuel oil", "#6 oil", "bunker c", "bunker fuel",
        "ifo 380", "ifo 180", "heavy fuel oil", "hfo",
        "no. 6 fuel oil", "number 6 fuel oil", "fuel oil",
        "residual fuel oil", "intermediate fuel oil",
    ],
    "jet fuel": [
        "jet fuel", "jet a", "jet a-1", "jp-4", "jp-5", "jp-8",
        "aviation fuel", "kerosene", "aviation kerosene",
    ],
    "hydraulic oil": [
        "hydraulic oil", "hydraulic fluid",
    ],
    "lube oil": [
        "lube oil", "lubricating oil", "lubricant", "motor oil",
        "engine oil", "gear oil",
    ],
    "oil": [
        "oil", "unknown oil", "oily water", "oily waste",
        "waste oil", "sludge", "slop oil", "oily bilge",
    ],
    "other petroleum": [
        "naphtha", "condensate", "natural gas condensate",
        "mineral spirits", "mineral oil", "white spirit",
        "transformer oil", "xylene", "toluene", "benzene",
        "ethylbenzene", "styrene",
    ],
    "vegetable oil": [
        "vegetable oil", "palm oil", "soybean oil", "canola oil",
        "cooking oil", "corn oil", "olive oil",
    ],
    "chemical": [
        "chemical", "sulfuric acid", "caustic soda", "sodium hydroxide",
        "ammonia", "chlorine", "phosphoric acid", "hydrochloric acid",
        "methanol", "ethanol", "phenol", "acrylonitrile",
    ],
    "unknown": [
        "unknown", "n/a", "not specified", "tbd", "?", "-",
    ],
}
UNKNOWN_COMMODITY_LABEL = "Other"
COMMODITY_SYNONYMS_PATH = "data/label_maps/commodity_synonyms.yaml"

# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
TEST_SIZE = 0.20
VAL_SIZE = 0.10  # fraction of the *full* labeled set held out for validation

# ---------------------------------------------------------------------------
# Sequence lengths
# ---------------------------------------------------------------------------
# DistilBERT: 512 captures description + first 1–2 posts.
# Phi-3.5-mini: 1024 leverages the model's longer context window.
DISTILBERT_MAX_SEQ_LENGTH = 512
LLM_MAX_SEQ_LENGTH = 1024

# ---------------------------------------------------------------------------
# TF-IDF baseline
# ---------------------------------------------------------------------------
TFIDF_MAX_FEATURES = 20000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
BASELINE_C = 1.0

# ---------------------------------------------------------------------------
# DistilBERT fine-tuning
# ---------------------------------------------------------------------------
HF_MODEL_NAME = "distilbert-base-uncased"
BERT_MAX_SEQ_LENGTH = 512             # alias for DISTILBERT_MAX_SEQ_LENGTH
MAX_SEQ_LENGTH = 512                  # kept for backward compatibility
BERT_EPOCHS = 8                        # more epochs for better convergence
BERT_BATCH_SIZE = 8
BERT_LR = 3e-5                          # slightly higher LR
BERT_WEIGHT_DECAY = 0.01
BERT_WARMUP_RATIO = 0.1
BERT_DROPOUT = 0.3                      # higher dropout to combat overfitting
BERT_GRADIENT_ACCUM_STEPS = 2           # effective batch = 16
BERT_EARLY_STOPPING_PATIENCE = 4        # more patience before stopping
BERT_GRADIENT_CLIP_NORM = 1.0

# ---------------------------------------------------------------------------
# Focal Loss (shared across DistilBERT and MLX tracks)
# ---------------------------------------------------------------------------
# Focal Loss down-weights easy examples and provides per-class alpha weighting
# to address severe class imbalance (Grounding=287 vs Tsunami=10).
FOCAL_LOSS_GAMMA = 2.0
FOCAL_LOSS_ALPHA = None               # auto-computed from inverse class frequency

# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------
# Per-label grid search over [THRESHOLD_MIN, THRESHOLD_MAX] with N steps.
# Optimizes for THRESHOLD_METRIC ("f1" or "f2").
THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEPS = 50
THRESHOLD_METRIC = "f1"

# ---------------------------------------------------------------------------
# Phi-3.5-mini MLX LoRA fine-tuning
# ---------------------------------------------------------------------------
# Uses Apple's MLX framework for native Metal GPU acceleration on Apple Silicon.
# No CUDA, no Ollama fallback — pure MLX.
MLX_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MLX_FALLBACK_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # upgrade path — still fits 16GB
MLX_QUANTIZATION = "q4"               # 4-bit quantization for memory efficiency
MLX_LORA_RANK = 12                    # higher rank for better expressiveness
MLX_LORA_ALPHA = 24
MLX_LORA_DROPOUT = 0.15
MLX_BATCH_SIZE = 4
MLX_GRADIENT_ACCUM_STEPS = 4          # effective batch = 16
MLX_LEARNING_RATE = 3e-4               # higher LR for faster convergence
MLX_NUM_EPOCHS = 8                     # more epochs
MLX_WARMUP_STEPS = 30
MLX_WEIGHT_DECAY = 0.01
MLX_ADAPTER_PATH = "models/phi/adapters.safetensors"
MLX_HEAD_PATH = "models/phi/head.safetensors"
MLX_NUM_LORA_LAYERS = 12              # apply LoRA to half the transformer layers
MLX_FOCAL_GAMMA = 2.5                  # stronger focus on hard examples

# ---------------------------------------------------------------------------
# External datasets
# ---------------------------------------------------------------------------
EXTERNAL_DATASETS = {
    "purdue_usmart": {
        "figshare_id": 26031487,
        "description": "Purdue uSMART enhanced oil spill dataset — same NOAA base with NLP-extracted release volumes",
        "label_column": "cause_tags",
        "text_column": "description",
    },
    "phmsa": {
        "url": "https://www.phmsa.dot.gov/data-and-statistics/pipeline/source-data",
        "description": "PHMSA pipeline incident data — rich cause taxonomy (corrosion, excavation damage, etc.)",
        "cause_column": "CAUSE_CATEGORY",
        "description_column": "DESCRIPTION",
    },
    "kaggle_oil_spill": {
        "dataset_id": "anoopjohny/oil-spill-incidents-dataset",
        "description": "Kaggle oil spill incidents with contributing factors column",
        "cause_column": "contributing_factors",
        "text_column": "description",
    },
}

# Taxonomy mapping from external dataset cause labels → our unified labels.
# Values are lists to support multi-label mappings (e.g., natural disaster → Hurricane + Tsunami).
CAUSE_MAPPING = {
    "phmsa": {
        "corrosion_internal": ["Pipeline"],
        "corrosion_external": ["Pipeline"],
        "excavation_damage": ["Pipeline"],
        "material_failure": ["Pipeline"],
        "equipment_failure": ["Pipeline"],
        "operator_error": ["Pipeline"],
        "natural_force": ["Hurricane"],
        "other": ["Other"],
        "unknown": ["Other"],
    },
    "kaggle": {
        "collision": ["Collision"],
        "grounding": ["Grounding"],
        "pipeline leak": ["Pipeline"],
        "pipeline rupture": ["Pipeline"],
        "hull failure": ["Derelict"],
        "structural failure": ["Derelict"],
        "equipment failure": ["Pipeline"],
        "natural disaster": ["Hurricane", "Tsunami"],
        "human error": ["Other"],
        "unknown": ["Mystery Substance"],
        "other": ["Other"],
    },
}

# ---------------------------------------------------------------------------
# Semi-supervised learning (self-training)
# ---------------------------------------------------------------------------
PSEUDO_LABEL_CONFIDENCE_THRESHOLD = 0.90   # minimum probability to accept pseudo-label
PSEUDO_LABEL_MARGIN_THRESHOLD = 0.30       # minimum gap between top-1 and top-2 prob
MAX_PSEUDO_LABELED_PER_CLASS = 200         # cap to prevent class explosion
SELF_TRAINING_MAX_ITERATIONS = 3

# ---------------------------------------------------------------------------
# Evaluation & reporting
# ---------------------------------------------------------------------------
REPORT_FORMATS = ["html", "json", "csv"]
PLOT_DPI = 150
PLOT_FIGSIZE = (10, 6)
PLOT_STYLE = "seaborn-v0_8-whitegrid"

# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------
STREAMLIT_TITLE = "Oil Spill Cause Classification"
STREAMLIT_PORT = 8501
STREAMLIT_CACHE_TTL = 3600               # 1 hour model cache
