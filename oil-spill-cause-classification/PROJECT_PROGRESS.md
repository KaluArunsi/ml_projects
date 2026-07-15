# Oil Spill Cause Classification — Project Progress

**Last Updated:** 2026-07-15

---

## Overview

End-to-end multi-label NLP system that classifies the **cause of oil spills** from incident descriptions and social media posts. Built with three modeling tracks for rigorous comparison: TF-IDF baseline, DistilBERT fine-tuning, and a small LLM fine-tuned via **MLX on Apple Silicon**.

**Hardware target:** 16GB MacBook M1 Pro (2020). All LLM work uses Apple's MLX framework (Metal GPU + unified memory), not CUDA.
**UI:** Streamlit web app (primary), CLI (`main.py`) for scripting.

---

## What Has Been Built

### 1. Project Scaffold

```
oil-spill-cause-classification/
├── src/
│   ├── __init__.py                # Package init
│   ├── config.py                  # Central configuration (all tracks)
│   ├── data_loader.py             # Load NOAA + external datasets
│   ├── preprocessing.py           # Clean, normalize, document assembly, split
│   ├── label_utils.py             # Taxonomy mapping for external datasets
│   ├── evaluate.py                # Metrics, threshold tuning, visualization
│   ├── ensemble.py                # Weighted voting ensemble
│   ├── predict.py                 # Unified inference pipeline
│   ├── report.py                  # HTML/JSON report generation
│   └── models/
│       ├── __init__.py
│       ├── _base.py               # Abstract BaseClassifier interface
│       ├── baseline.py            # TF-IDF + Logistic Regression
│       ├── distilbert_model.py    # DistilBERT + Focal Loss + MPS
│       └── llm_mlx.py             # Qwen2.5 MLX LoRA classifier
├── app/
│   └── streamlit_app.py           # 5-page Streamlit web UI
├── data/
│   ├── raw/                       # NOAA incidents CSV + posts XLSX
│   ├── external/                  # Purdue uSMART dataset
│   └── processed/                 # Cleaned parquet, label matrices, splits
├── models/
│   ├── tfidf/                     # Trained TF-IDF baseline
│   ├── distilbert/                # Trained DistilBERT (safetensors)
│   ├── distilbert_checkpoints/    # HF Trainer checkpoints
│   └── phi/                       # MLX LoRA adapters + config
├── output/
│   ├── reports/                   # Generated HTML/JSON reports
│   └── plots/                     # Evaluation charts
├── main.py                        # CLI orchestrator (train/eval/predict/report/serve)
├── requirements.txt               # All dependencies
├── implementation_plan.md         # Original design document
└── PROJECT_PROGRESS.md            # This file
```

### 2. Central Configuration (`src/config.py`)

All tunable parameters consolidated in one place:
- **Data paths** — raw, processed, external, label maps, models, output
- **Label taxonomy** — 15 cause labels (Grounding, Collision, Pipeline, Mystery Substance, Hurricane, Derelict, Wellhead, Adrift, Search + Rescue, Railcar, Coral, Marine Debris, Marine Mammal, Tsunami, Other)
- **Text assembly** — max 12 posts per incident, 1500 word cap, [SEP] delimiter
- **Commodity normalization** — 1555 raw values mapped to ~12 canonical categories
- **Train/val/test split** — 70/10/20 with iterative multi-label stratification
- **Sequence lengths** — 512 (DistilBERT), 1024 (LLM)
- **TF-IDF** — 20K features, (1,2)-grams, min_df=2, sublinear TF
- **DistilBERT** — 8 epochs, batch 8, LR 3e-5, Focal Loss (γ=2.0), MPS device
- **MLX LoRA** — Qwen2.5-0.5B-Instruct, 4-bit, rank 12, batch 4, grad_accum 4
- **External datasets** — Purdue uSMART (figshare), PHMSA, Kaggle — with taxonomy mappings
- **Semi-supervised** — self-training config (pseudo-label confidence ≥ 0.90)
- **Evaluation** — per-label threshold tuning, report formats (HTML/JSON/CSV)

### 3. Data Pipeline (Complete)

| Step | Description | Status |
|------|-------------|--------|
| Data loading | Load 4,473 incidents + 25,733 posts from NOAA/figshare | ✅ |
| Commodity normalization | 1,555 raw values → ~12 canonical categories | ✅ |
| Label parsing | Pipe-split tags → MultiLabelBinarizer (14 active + "Other") | ✅ |
| Document assembly | Description + [SEP] + sorted posts, word-capped at 1500 | ✅ |
| Feature engineering | Log₁₀ release volume, temporal features (year/month/season) | ✅ |
| Stratified split | Iterative multi-label stratification (595 train / 85 val / 169 test) | ✅ |
| Label weights | Inverse-frequency weights capped at 100× for focal loss | ✅ |
| Artifact persistence | Parquet + .npy saved to `data/processed/` | ✅ |

**Key data facts:**
- 849 labeled incidents (19%), 3,624 unlabeled (81%)
- 15 unique cause tags; 14 survive MIN_LABEL_COUNT=10 (Historic Wreck → "Other")
- Severe class imbalance: Grounding (287) → Tsunami (10)
- 96% single-label, only 35/849 incidents have 2+ labels
- 1,982 unlabeled incidents have posts → semi-supervised potential

### 4. Model Track 1: TF-IDF Baseline ✅

**Architecture:** TfidfVectorizer → OneVsRestClassifier(LogisticRegression)

| Metric | Value |
|--------|-------|
| Micro-F1 | 0.612 |
| Macro-F1 | 0.593 |
| Weighted-F1 | 0.718 |
| Hamming Loss | 0.064 |
| Subset Accuracy | 0.378 |

**Top-performing labels:** Railcar (1.0), Marine Mammal (1.0), Tsunami (1.0), Hurricane (0.889)
**Struggling labels:** Wellhead (0.0), Marine Debris (0.044), Adrift (0.167), Derelict (0.238)

Trained in ~30 seconds. Per-label threshold tuning improves macro-F1 by ~10 points vs. fixed 0.5 threshold.

### 5. Model Track 2: DistilBERT ✅

**Architecture:** DistilBertModel → [CLS] pooling → Dropout(0.3) → Linear(768, 14) → Sigmoid
**Training:** Focal Loss (γ=2.0, per-class α auto-computed), AdamW (LR=3e-5), MPS device, batch 8 × grad_accum 2 = effective 16, early stopping (patience=4)

Model saved as safetensors. HF Trainer checkpoints preserved. Supports weighted random sampling for class imbalance.

### 6. Model Track 3: MLX LLM LoRA ✅

**Architecture:** Qwen2.5-0.5B-Instruct (4-bit, frozen) → mean pool over hidden states → LayerNorm → Dropout → Linear(896, 14) → Sigmoid

Only LoRA adapters (rank=12, α=24, 12 layers) + classification head are trainable.
Default model: Qwen2.5-0.5B-Instruct (~1GB, ~500MB in 4-bit). Fits comfortably on M1 Pro 16GB.

**Training:** AdamW w/ linear warmup + linear decay, batch 4 × grad_accum 4 = effective 16, up to 8 epochs, Focal Loss (γ=2.5).

### 7. Ensemble ✅

Weighted voting ensemble that combines probabilities from all three tracks. Weights learned via grid search on validation macro-F1. Per-label threshold tuning applied to ensemble output.

### 8. Evaluation Framework ✅

- **Per-label metrics:** Precision, Recall, F1, ROC-AUC, Support
- **Aggregated:** Micro-F1 (primary), Macro-F1, Weighted-F1, Hamming Loss, Subset Accuracy
- **Threshold tuning:** Per-label grid search [0.05, 0.95] × 50 steps optimizing F1
- **Visualizations:** Per-label F1 bar charts, model comparison charts
- **Reports:** Self-contained HTML with embedded matplotlib charts + structured JSON

### 9. Prediction Pipeline ✅

`Predictor` class providing:
- **Single prediction:** Structured dict or raw text → predicted causes with confidence scores
- **Batch prediction:** CSV DataFrame → labeled DataFrame with per-cause probabilities
- **Multi-model:** Load all available models, auto-select best, or ensemble average
- **Commodity classification:** Synonym-based canonical category lookup

### 10. Streamlit Web App ✅

5-page multi-page app:
1. **🏠 Home** — project overview, model status indicators
2. **📊 Data Explorer** — upload CSV/Excel, explore built-in NOAA dataset, label distribution charts
3. **⚙️ Training** — configure and train all three model tracks with adjustable hyperparameters
4. **🏷️ Prediction** — single/batch inference with Plotly confidence bars
5. **📄 Reports** — view existing reports, generate new ones

### 11. CLI (`main.py`) ✅

Five commands:
- `python main.py train --model all` — train one or all models
- `python main.py evaluate --model baseline` — evaluate and generate reports
- `python main.py predict incident.json` — single or batch inference
- `python main.py report --format html` — generate standalone report
- `python main.py serve` — launch Streamlit app

---

## External Dataset Integration (Partial)

| Dataset | Priority | Status |
|---------|----------|--------|
| Purdue uSMART (figshare 26031487) | High | ✅ Loaded & cached in `data/external/purdue_usmart/` |
| PHMSA Pipeline Incidents | High | ⬜ Loader written, data not yet downloaded |
| Kaggle Oil Spill Incidents | Medium | ⬜ Loader written, requires kagglehub |

Taxonomy mapping from external cause labels → unified labels is implemented in `label_utils.py` and `config.py` (CAUSE_MAPPING for PHMSA and Kaggle).

---

## What Remains (Not Yet Built)

| Feature | Priority | Status |
|---------|----------|--------|
| Semi-supervised learning (self-training on 1,982 unlabeled incidents) | High | ⬜ Config exists, implementation not started |
| PHMSA data download & integration | High | ⬜ Loader ready, data needs manual download |
| Kaggle dataset integration | Medium | ⬜ Loader ready, requires kagglehub + API key |
| Apple Container packaging | Low | ⬜ Optional — Containerfile + fruitbox |
| Unit tests (`tests/`) | Medium | ⬜ Directory not yet created |
| LLM `load()` full restoration (load model weights, not just metadata) | Medium | ⚠️ Metadata loads, model needs re-fit |
| Data augmentation for rare labels | Medium | ⬜ Not started |
| MLX model checkpointing during training | Low | ⬜ Not implemented |

---

## Known Issues

1. **DistilBERT `load()`**: Attempts `torch.load("pytorch_model.bin")` but model saved as safetensors — custom model reconstruction fragile. Fixed in latest cleanup.
2. **MLX LLM `load()`**: Only restores metadata; model weights are not reloadable without re-fitting. The `adapters.npz` is saved but not consumed on load.
3. **Ensemble weight learning**: Grid search hardcoded for exactly 3 models — doesn't generalize to arbitrary N.
4. **Rare-label performance**: Labels with <5 test examples (Wellhead, Adrift, Marine Debris, Coral, Marine Mammal, Tsunami, Other) show unstable F1 due to 1–4 sample test sets.
5. **MLX training stability**: No mid-epoch checkpointing — if training crashes at epoch 6/8, all progress lost.

---

## Performance Summary (TF-IDF Baseline on Test Set)

**169 test incidents, 15 labels**

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Grounding | 0.825 | 0.825 | 0.825 | 57 |
| Collision | 0.550 | 0.786 | 0.647 | 28 |
| Pipeline | 0.760 | 0.704 | 0.731 | 27 |
| Mystery Substance | 0.850 | 0.680 | 0.756 | 25 |
| Hurricane | 1.000 | 0.800 | 0.889 | 10 |
| Derelict | 0.139 | 0.833 | 0.238 | 6 |
| Wellhead | 0.000 | 0.000 | 0.000 | 4 |
| Adrift | 0.111 | 0.333 | 0.167 | 3 |
| Search + Rescue | 1.000 | 0.667 | 0.800 | 3 |
| Railcar | 1.000 | 1.000 | 1.000 | 3 |
| Coral | 0.667 | 1.000 | 0.800 | 2 |
| Marine Debris | 0.023 | 0.500 | 0.044 | 2 |
| Marine Mammal | 1.000 | 1.000 | 1.000 | 2 |
| Tsunami | 1.000 | 1.000 | 1.000 | 2 |
| Other | 0.000 | 0.000 | 0.000 | 1 |

**Aggregate:** Micro-F1 0.612 | Macro-F1 0.593 | Weighted-F1 0.718 | Hamming Loss 0.064

The baseline achieves solid performance on frequent labels (Grounding, Collision, Pipeline) but degrades on rare labels where test support is only 1–6 examples. The DistilBERT and MLX LLM tracks are expected to improve on the rare-label tail through better semantic understanding.

---

## Dependencies

See `requirements.txt` for full list. Key packages:
- **Data:** numpy, pandas, scipy, pyyaml
- **ML:** scikit-learn, scikit-multilearn
- **DL:** torch, transformers, datasets, accelerate, evaluate, peft
- **MLX:** mlx, mlx-lm, safetensors
- **Viz:** matplotlib, seaborn, plotly
- **App:** streamlit
- **Utils:** tqdm, joblib, requests

---

## Reproducibility

- Random seed: 42 (set in `config.RANDOM_STATE`)
- Data: NOAA IncidentNews via figshare (CC BY 4.0)
- Hardware: Apple Silicon (MPS for PyTorch, MLX for LLM)
- All hyperparameters documented in `src/config.py`
