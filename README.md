# ml_projects

Machine learning and AI projects across structured risk modeling, forecasting, NLP, and local-first Streamlit apps. Each project is self-contained with its own dependencies and documentation.

## Project Index

| Project | Problem | Interface | Tests | Status |
| --- | --- | --- | --- | --- |
| [OpenBPO Drift](openbpo-drift/) | Local-first KPI drift monitoring for BPO/contact-center operations | Streamlit app | Yes | Deployed v1 |
| [Oil Spill Cause Classification](oil-spill-cause-classification/) | Multi-label NLP classification of oil spill causes | CLI + Streamlit app | Not yet | Active build |
| [Credit Worthiness Assessment](credit-worthiness/) | Probabilistic loan default risk assessment | CLI + notebook | Not yet | Portfolio project |
| [Hotel Cancellation Risk](hotel-cancellation-determination/) | Cancellation prediction and severity tiering | CLI + notebook | Not yet | Portfolio project |
| [Bike Demand Forecasting](bike-demand-prediction/) | Hourly bike demand prediction | CLI + notebook | Not yet | Early project |

## Highlights

**[OpenBPO Drift](openbpo-drift/)**
Local-first, open-source reference app for detecting KPI drift in BPO operations. Includes upload, schema mapping, validation, rolling drift detection, charts, exports, tests, screenshots, and a documented security posture.

**[Oil Spill Cause Classification](oil-spill-cause-classification/)**
Multi-label NLP system that classifies oil spill causes from incident descriptions and social media posts. Includes TF-IDF, DistilBERT, and MLX LoRA model tracks, plus a Streamlit UI and CLI.

**[Credit Worthiness Assessment](credit-worthiness/)**
Probabilistic LendingClub loan default classifier with complementary repayment/default probabilities, OCC-inspired risk tiers, model comparison, and visual model outputs.

**[Hotel Cancellation Risk](hotel-cancellation-determination/)**
Two-model cancellation risk system with severity tiers so revenue teams can prioritize bookings without spreadsheet triage.

**[Bike Demand Forecasting](bike-demand-prediction/)**
Hourly Seoul bike demand prediction using weather, seasonal, and calendar features.

## Repository Standards

New or refreshed projects should aim for:

- `README.md` with problem framing, setup, usage, outputs, and limitations
- `src/` for reusable code and `main.py` or documented app entry point
- `requirements.txt` or project-specific dependency file
- `tests/` for productionized projects
- `data/README.md` when datasets are included or expected externally
- `model_output_showcase/` only for curated presentation artifacts
- generated `output/`, local virtual environments, caches, and checkpoints excluded from git

## Tech

Python across the board. The stack varies by project but includes scikit-learn, PyTorch, Transformers, MLX on Apple Silicon, Streamlit, Plotly, pandas, NumPy, and the standard data science toolchain.
