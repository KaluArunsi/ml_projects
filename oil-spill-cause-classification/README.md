# Oil Spill Cause Classification

Multi-label NLP system for classifying the cause of oil spills from incident descriptions and associated social media posts. The project compares three model tracks: a fast TF-IDF baseline, DistilBERT fine-tuning, and an MLX LoRA model intended for Apple Silicon experimentation.

## Status

Active build. The data pipeline, TF-IDF baseline, DistilBERT track, MLX track scaffolding, ensemble logic, CLI, and Streamlit app exist. Tests are not yet in place, and some model-loading paths need hardening before this should be treated as production-ready.

## What It Does

- Loads NOAA incident data and associated post data
- Normalizes incident text, commodity fields, and cause labels
- Builds multi-label train/validation/test splits
- Trains or evaluates TF-IDF, DistilBERT, and MLX model tracks
- Runs single or batch prediction
- Generates HTML/JSON reports
- Provides a Streamlit app for data exploration, training, prediction, and reporting

## Project Structure

```text
oil-spill-cause-classification/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   ├── external/
│   └── processed/
├── models/
│   ├── tfidf/
│   ├── distilbert/
│   └── phi/
├── output/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── ensemble.py
│   ├── evaluate.py
│   ├── label_utils.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── report.py
│   └── models/
├── main.py
├── requirements.txt
├── PROJECT_PROGRESS.md
└── implementation_plan.md
```

## Setup

```bash
cd oil-spill-cause-classification
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some model tracks require heavier dependencies and Apple Silicon assumptions. The TF-IDF baseline is the lowest-friction path.

## Usage

Train a model:

```bash
python main.py train --model baseline
```

Evaluate trained models:

```bash
python main.py evaluate --model baseline
```

Run prediction:

```bash
python main.py predict incident.json --model baseline
```

Launch the Streamlit app:

```bash
python main.py serve
```

## Current Limitations

- No automated test suite yet
- Some data sources require local files or external credentials
- DistilBERT and MLX model restoration paths need additional validation
- Rare-label metrics are unstable because several labels have very small test support
- Large datasets and model artifacts need a clearer long-term storage policy

See [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md) for detailed implementation notes and known issues.
