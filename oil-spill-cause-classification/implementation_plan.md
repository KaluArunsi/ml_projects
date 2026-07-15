# Oil Spill Cause Classification — Full System Plan

## Context

Build an end-to-end multi-label NLP system that classifies the cause of oil spills from incident descriptions and social media posts. Users upload incident data via a **Streamlit web app**; the system ingests, learns from labeled data, and produces a comprehensive report with accurate cause labels, commodity type, and threat classification. **Three modeling tracks** for rigorous comparison: TF-IDF baseline, DistilBERT fine-tuning, and a small LLM fine-tuned via **MLX on Apple Silicon**.

**Hardware:** 16GB MacBook M1 Pro (2020). All LLM work uses Apple's MLX framework, not CUDA.
**UI:** Streamlit web app (primary), CLI for scripting.
**Existing:** `src/config.py` with well-documented parameters; 4,473 incidents (19% labeled); 25,733 posts.

### Key Data Facts
- **849 labeled incidents** (19%), 3,624 unlabeled (81%)
- 15 unique cause tags; 14 survive `MIN_LABEL_COUNT=10` (Historic Wreck → "Other")
- Severe class imbalance: Grounding (287) → Tsunami (10)
- Only 35/849 incidents have 2+ labels (96% single-label)
- 1,982 unlabeled incidents have posts → semi-supervised potential
- Commodity: 1,555 unique raw values → needs normalization to ~20 canonical types

---

## Phase 1: Project Scaffold & Data Pipeline

### 1.1 Project Structure
```
oil-spill-cause-classification/
├── src/
│   ├── __init__.py
│   ├── config.py                # Extended config (all tracks + new datasets)
│   ├── data_loader.py           # Load all data sources (primary + external)
│   ├── preprocessing.py         # Clean, normalize, assemble documents
│   ├── label_utils.py           # Taxonomy mapping, multi-label binarization
│   ├── features.py              # TF-IDF vectorization
│   ├── models/
│   │   ├── __init__.py
│   │   ├── _base.py             # Abstract BaseClassifier interface
│   │   ├── baseline.py          # TF-IDF + LogisticRegression
│   │   ├── distilbert.py        # DistilBERT with multi-label head + Focal Loss
│   │   └── llm_mlx.py           # Small LLM via MLX LoRA + classification head
│   ├── ensemble.py              # Weighted voting ensemble
│   ├── evaluate.py              # Metrics, threshold tuning, comparison plots
│   ├── predict.py               # Inference pipeline for user data
│   ├── report.py                # HTML/MD report generator
│   └── semi_supervised.py       # Self-training with pseudo-labeling
├── app/
│   └── streamlit_app.py         # Streamlit web UI
├── data/
│   ├── raw/                     # Original files (incidents CSV, posts XLSX)
│   ├── external/                # Downloaded external datasets (gitignored)
│   ├── processed/               # Cleaned parquet files
│   └── label_maps/              # Taxonomy mapping YAML files
├── models/                      # Saved model artifacts
├── output/
│   ├── plots/                   # Evaluation figures
│   ├── logs/                    # Training logs
│   └── reports/                 # Generated reports
├── notebooks/
│   └── exploration.ipynb        # EDA + cross-track comparison
├── tests/
│   ├── test_preprocessing.py
│   ├── test_label_utils.py
│   ├── test_models.py
│   └── test_predict.py
├── main.py                      # CLI orchestrator
├── requirements.txt
└── README.md
```

### 1.2 Config Extension (`src/config.py`)

Preserve all existing constants. Add these sections:

```python
# ---- External datasets ------------------------------------------------
PURDUE_FIGSHARE_ID = 26031487
PHMSA_URL = "https://www.phmsa.dot.gov/..."  # Pipeline incident data
KAGGLE_DATASET = "anoopjohny/oil-spill-incidents-dataset"
COMMODITY_SYNONYMS_PATH = "data/label_maps/commodity_synonyms.yaml"

# ---- Sequence length fix ----------------------------------------------
DISTILBERT_MAX_SEQ_LENGTH = 512   # Up from 256 — captures description + 1-2 posts
LLM_MAX_SEQ_LENGTH = 1024         # MLX can handle longer contexts on M1

# ---- Loss functions ---------------------------------------------------
FOCAL_LOSS_GAMMA = 2.0
FOCAL_LOSS_ALPHA = None           # Auto-computed from inverse class frequency

# ---- Threshold tuning -------------------------------------------------
THRESHOLD_METRIC = "f1"           # Optimize per-label F1 on validation

# ---- Semi-supervised --------------------------------------------------
PSEUDO_LABEL_CONFIDENCE = 0.90
MAX_PSEUDO_LABELED_PER_CLASS = 200

# ---- MLX LLM ----------------------------------------------------------
LLM_MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
LLM_FALLBACK = "google/gemma-2-2b-it"   # If Phi-3.5 OOMs on M1
LLM_LORA_RANK = 16
LLM_LORA_ALPHA = 32
LLM_BATCH_SIZE = 2
LLM_GRAD_ACCUM = 4                # Effective batch = 8
LLM_EPOCHS = 4
LLM_LR = 2e-4
LLM_USE_4BIT = True
```

### 1.3 Data Loading (`data_loader.py`)
```python
def load_incidents(path=INCIDENTS_FILENAME) -> pd.DataFrame
def load_posts(path=POSTS_FILENAME) -> pd.DataFrame
def load_purdue_dataset(cache_dir="data/external") -> Optional[pd.DataFrame]
def load_phmsa_data(cache_dir="data/external") -> Optional[pd.DataFrame]
def load_kaggle_dataset(cache_dir="data/external") -> Optional[pd.DataFrame]
def load_all_data(use_external=True) -> dict[str, pd.DataFrame]
```
Each external loader handles download-on-missing with graceful fallback (logs warning, returns None).

### 1.4 Preprocessing (`preprocessing.py`)
```python
def normalize_commodity(df, synonym_path) -> pd.DataFrame
    # 1,555 raw values → ~20 canonical types via YAML synonym map

def assemble_documents(incidents, posts, max_posts=12) -> pd.DataFrame
    # description + [SEP] + post_1 + [SEP] + ... + post_n
    # Sorted by post date descending. Null posts fall back to post title.
    # Returns: [incident_id, text, labels, commodity, release_volume, threat, year]

def parse_tags(tags_series) -> tuple[pd.DataFrame, list[str]]
    # Pipe-split → MultiLabelBinarizer → binary matrix + label names

def filter_rare_labels(label_matrix, min_count=10) -> pd.DataFrame
    # Labels with count < min_count → "Other" column

def stratified_split(df, test_size=0.20, val_size=0.10) -> tuple
    # Stratify by label combination presence. Uses iterative_train_test_split.
    # ~595 train / ~85 val / ~169 test
```

### 1.5 Label Utils (`label_utils.py`)
```python
class TaxonomyMapper:
    """Map external dataset cause labels → unified taxonomy via YAML."""
    def __init__(self, mapping_dir="data/label_maps")
    def map_labels(self, source: str, external_labels) -> pd.DataFrame
    def normalize_commodity(self, raw) -> pd.Series

# YAML mapping example (phmsa_to_unified.yaml):
#   corrosion:          {label: Pipeline, weight: 1.0}
#   excavation_damage:  {label: Pipeline, weight: 0.8}
#   equipment_failure:  {label: Other,    weight: 0.6}
```

### 1.6 Additional Dataset Integration Strategy
| Dataset | Priority | Cause Labels? | Harmonization |
|---|---|---|---|
| Purdue uSMART (figshare 26031487) | High | Same NOAA tags | Direct merge — enriches release volume data |
| PHMSA Pipeline Incidents | High | Yes (corrosion, excavation, etc.) | Map taxonomy → unified labels via TaxonomyMapper |
| Kaggle Oil Spill Incidents | Medium | "Contributing factors" | Text fuzzy-match to unified labels |
| BOEM/BSEE Offshore | Medium | Cause codes | Map codes → unified; adds Gulf coverage |

---

## Phase 2: Modeling Tracks

### 2.1 Abstract Base (`models/_base.py`)
```python
class BaseClassifier(ABC):
    @abstractmethod
    def fit(self, X: pd.Series, y: pd.DataFrame, X_val=None, y_val=None) -> None: ...
    @abstractmethod
    def predict_proba(self, X: pd.Series) -> np.ndarray: ...  # (n, n_labels)
    @abstractmethod
    def predict(self, X, thresholds=None) -> np.ndarray: ...
    @abstractmethod
    def save(self, path: str) -> None: ...
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseClassifier": ...
```

### 2.2 Track 1: TF-IDF + Logistic Regression (`models/baseline.py`)
- `TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=2, sublinear_tf=True)`
- `OneVsRestClassifier(LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000))`
- Per-label threshold tuning via grid search on validation set
- **Feature importance**: top 20 tokens per cause label
- **Training**: ~30 seconds. Expected micro-F1: ~0.50-0.60

### 2.3 Track 2: DistilBERT (`models/distilbert.py`)
```
Architecture:
  DistilBertModel → [CLS] pooling → Dropout(0.2) → Linear(768, 14) → Sigmoid

Key changes from existing config:
  - MAX_SEQ_LENGTH: 256 → 512
  - Loss: BCEWithLogitsLoss → FocalLoss(gamma=2.0, alpha=per-class weights)
  - Device: cpu → mps (Apple Silicon GPU)
  - Mixed precision via torch.amp.autocast(device_type="mps")

Training:
  - FocalLoss with auto-computed alpha from inverse class frequency
  - AdamW(lr=2e-5, weight_decay=0.01), linear warmup 10%, linear decay
  - Batch size 8, 6 epochs, early stopping on val micro-F1 (patience=3)
  - Expected macro-F1: ~0.55-0.65
```

### 2.4 Track 3: Small LLM via MLX (`models/llm_mlx.py`)

**Why MLX on M1 Pro:**
- Apple's native array framework — uses Metal GPU + unified memory
- `mlx-lm` supports LoRA fine-tuning with 4-bit quantization
- 16GB unified memory can handle 4-bit 3.8B model + LoRA adapters
- No Docker/CUDA dependency — pure Python on macOS

**Architecture:**
```
Phi-3.5-mini (4-bit, frozen) → last hidden state → mean pool →
  LayerNorm → Dropout(0.1) → Linear(3072, 14) → Sigmoid
Only LoRA adapters + classification head are trainable.
```

**Training pipeline:**
1. Convert HF model → MLX format via `mlx-lm convert`
2. Load 4-bit quantized model with LoRA adapters (rank=16, alpha=32)
3. Forward: extract hidden states → mean pool → classification head → sigmoid
4. Focal Loss with per-class alpha (gamma=2.0)
5. Train 3-4 epochs, batch 2 × grad_accum 4 = effective 8, lr 2e-4
6. Per-label threshold tuning on validation set
7. Save LoRA adapter weights only (~10-30MB .safetensors)

**Fallback**: If MLX fine-tuning is unstable, use Ollama + GGUF model for prompt-based classification with constrained JSON output. Lower accuracy but more battle-tested.

**Expected**: Macro-F1 ~0.60-0.70. Training time: ~1-2 hours on M1 Pro.

### 2.5 Ensemble (`ensemble.py`)
Weighted average of probabilities from all three tracks. Weights learned by optimizing macro-F1 on the validation set.

### 2.6 Shared Evaluation (`evaluate.py`)
```python
def compute_metrics(y_true, y_pred, y_prob, label_names) -> dict
    # Per-label: F1, Precision, Recall, ROC-AUC, PR-AUC
    # Aggregated: Micro-F1 (primary), Macro-F1, Hamming Loss, Exact Match

def find_optimal_thresholds(y_true, y_prob, metric="f1") -> np.ndarray
    # Grid search [0.05, 0.95] step 0.05 per label

def plot_per_label_f1(metrics, label_names, output_path)        # Bar chart
def plot_threshold_curves(y_true, y_prob, label_names, output)   # F1 vs threshold
def plot_model_comparison(all_metrics, output_path)              # Radar + bar
def plot_confusion_heatmap(y_true, y_pred, label_names, output)  # Label co-occurrence
```

---

## Phase 3: Streamlit Web App (`app/streamlit_app.py`)

### Pages/Tabs:
1. **📤 Data Upload** — CSV/Excel upload. Preview table, auto-detect columns, summary stats (N incidents, date range, missing values).
2. **⚙️ Model Config** — Choose track(s): TF-IDF, DistilBERT, LLM, or Ensemble. Configure confidence threshold. Option to include commodity/threat prediction.
3. **🏷️ Classification** — Run inference. Progress bar. Results table: incident ID, description snippet, predicted causes (with confidence %), commodity, threat.
4. **📊 Report** — Auto-generated interactive report:
   - Cause distribution bar chart (Plotly)
   - Per-incident detail with expandable rows
   - Confidence histogram
   - Model comparison (if multiple tracks)
   - Export buttons: CSV, HTML, JSON

### Architecture:
- `@st.cache_resource` for model loading
- Session state for uploaded data + results
- Plotly for interactive charts
- Background inference for batch processing

---

## Phase 4: Report Generation (`report.py`)

HTML report with embedded Plotly charts:
1. Executive summary (N incidents, top causes, model, aggregate confidence)
2. Cause distribution bar chart
3. Per-incident predictions table (sortable, filterable)
4. Model comparison section (if multiple tracks)
5. Confidence analysis (histogram, low-confidence cases highlighted)
6. Commodity breakdown
7. Export: standalone HTML, CSV predictions table, JSON full results

---

## Phase 5: Packaging (Optional — Apple Container)

- `Containerfile` with Python 3.11, MLX, transformers, streamlit
- `fruitbox` for Compose-style orchestration
- One-command: `apple-container run oil-spill-classifier`

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Unlabeled data (81%) | Labeled-only training initially; semi-supervised in Phase 6 | Noisy pseudo-labels degrade quality. Keep unlabeled for later. |
| MAX_SEQ_LENGTH | 512 (DistilBERT), 1024 (LLM) | Captures description + 1-2 posts. Full docs need 2K+ — impractical. |
| LLM approach | Classification head (not generative) | Single forward pass, calibrated scores, native multi-label. Generative is slower. |
| LLM framework | MLX (Apple native) | M1 Pro has no CUDA. MLX uses Metal GPU + unified memory. |
| Loss function | Focal Loss (gamma=2.0) | Addresses Grounding (287) vs Tsunami (10) imbalance. |
| Multi-label strategy | Binary relevance (independent sigmoids) | 96% of incidents are single-label. Complex chain/transformer methods overkill. |
| Threshold tuning | Per-label grid search on validation | Single biggest impact technique — can improve macro-F1 by 10-20 points. |
| External datasets | Map to unified taxonomy via YAML | Different cause schemas. Don't blindly concatenate labels. |
| UI | Streamlit primary, CLI secondary | Fastest path to interactive web app. User's preference. |

---

## Verification Plan

1. **Data pipeline**: Run preprocessing → verify 849 labeled docs, 14-column label matrix, no leakage
2. **TF-IDF baseline**: Train → micro-F1 > 0.50 (must significantly beat random)
3. **DistilBERT**: Fine-tune → micro-F1 improves over TF-IDF by ≥5 points
4. **LLM/MLX**: Fine-tune LoRA → micro-F1 within ±3 points of DistilBERT (comparable)
5. **Threshold tuning**: Grid search → macro-F1 improves ≥3 points vs. fixed 0.5
6. **Ensemble**: Weighted average beats best individual model
7. **Streamlit app**: Upload CSV → predictions render, report downloads
8. **Cross-track**: `evaluate.py` outputs comparison table — clear winner identified

---

## Implementation Order

1. **Extended config** — add all new constants to `config.py`
2. **Data pipeline** — `data_loader.py` + `preprocessing.py` + `label_utils.py` (foundation)
3. **TF-IDF baseline** — quick win, establishes metric baseline
4. **DistilBERT** — primary deep learning track (MPS-optimized)
5. **MLX LLM** — experimental track, most complex (conversion + LoRA + head)
6. **Evaluation** — shared metrics, threshold tuning, comparison plots
7. **Ensemble** — weighted voting
8. **Streamlit app** — user-facing interface
9. **Report generation** — HTML reports
10. **External datasets** — PHMSA, Purdue, Kaggle integration
11. **Semi-supervised** — self-training on 1,982 unlabeled incidents with posts
12. **Apple Container** — optional packaging
