"""
Oil Spill Cause Classification — Streamlit App
==============================================
Multi-page web app for training models, exploring data, running predictions,
and generating reports.

Usage:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="Oil Spill Cause Classification",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Streamlit entry point."""

    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🛢️ Oil Spill Classifier")
        st.markdown("---")

        st.markdown("### Navigation")
        page = st.radio(
            "Go to",
            ["🏠 Home", "📊 Data Explorer", "⚙️ Training", "🏷️ Prediction", "📄 Reports"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.caption("Models: TF-IDF · DistilBERT · Phi-3.5 MLX")
        st.caption(f"Project: {PROJECT_ROOT.name}")

    # ── Page Router ──────────────────────────────────────────────────
    if page == "🏠 Home":
        show_home()
    elif page == "📊 Data Explorer":
        show_data_explorer()
    elif page == "⚙️ Training":
        show_training()
    elif page == "🏷️ Prediction":
        show_prediction()
    elif page == "📄 Reports":
        show_reports()


# ═══════════════════════════════════════════════════════════════════════
# Home
# ═══════════════════════════════════════════════════════════════════════

def show_home():
    st.title("Oil Spill Cause Classification")
    st.markdown(
        """
        **Multi-label NLP system** for classifying the cause of oil spills
        from incident descriptions and social media posts.

        ---
        ### How It Works
        1. **Upload** incident data (CSV or Excel)
        2. **Train** one or more models on labeled data
        3. **Predict** causes for new incidents with confidence scores
        4. **Generate** comprehensive evaluation reports

        ### Three Modeling Tracks
        | Track | Model | Strength |
        |-------|-------|----------|
        | Baseline | TF-IDF + Logistic Regression | Fast, interpretable |
        | Deep Learning | DistilBERT (67M) | Good accuracy on MPS |
        | LLM | Phi-3.5-mini MLX LoRA (3.8B) | Best accuracy, Apple Silicon native |

        ### Dataset
        - **4,473** oil spill incidents (1967–2023) from NOAA IncidentNews
        - **25,733** associated social media posts
        - **15 cause labels** including Grounding, Collision, Pipeline, Hurricane, etc.
        - **849** labeled incidents for supervised training

        ---
        ### Quick Start
        Use the sidebar to navigate between pages.

        - **Data Explorer**: Upload your data and explore distributions
        - **Training**: Configure and train models
        - **Prediction**: Classify new oil spill incidents
        - **Reports**: View and download evaluation reports
        """
    )

    # Model status
    st.markdown("### System Status")

    col1, col2, col3 = st.columns(3)
    models_dir = PROJECT_ROOT / "models"

    with col1:
        baseline_path = models_dir / "tfidf" / "metadata.json"
        if baseline_path.exists():
            st.success("✅ TF-IDF Baseline")
        else:
            st.info("⬜ TF-IDF Baseline (not trained)")

    with col2:
        bert_path = models_dir / "distilbert" / "training_metadata.json"
        if bert_path.exists():
            st.success("✅ DistilBERT")
        else:
            st.info("⬜ DistilBERT (not trained)")

    with col3:
        phi_path = models_dir / "phi" / "metadata.json"
        if phi_path.exists():
            st.success("✅ Phi-3.5 MLX LLM")
        else:
            st.info("⬜ Phi-3.5 MLX LLM (not trained)")


# ═══════════════════════════════════════════════════════════════════════
# Data Explorer
# ═══════════════════════════════════════════════════════════════════════

def show_data_explorer():
    st.title("📊 Data Explorer")
    st.markdown("Upload oil spill incident data or explore the built-in NOAA dataset.")

    tab1, tab2 = st.tabs(["Upload Data", "Built-in Dataset"])

    with tab1:
        st.markdown("### Upload Incident Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=["csv", "xlsx"],
            help="File should contain at minimum a 'description' column with incident text.",
        )

        if uploaded_file is not None:
            import pandas as pd

            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"Loaded {len(df):,} incidents, {len(df.columns)} columns")

            with st.expander("Preview", expanded=True):
                st.dataframe(df.head(20), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Incidents", len(df))
                st.metric("Columns", len(df.columns))
            with col2:
                text_cols = [c for c in df.columns if df[c].dtype == "object"]
                st.metric("Text Columns", len(text_cols))
                if text_cols:
                    st.caption(f"Found: {', '.join(text_cols[:5])}")

            st.markdown("### Column Types")
            st.dataframe(
                pd.DataFrame({
                    "Column": df.columns,
                    "Type": df.dtypes.values,
                    "Non-Null": df.count().values,
                    "Null %": (df.isna().mean() * 100).round(1).values,
                }),
                use_container_width=True,
            )

    with tab2:
        st.markdown("### Built-in NOAA IncidentNews Dataset")
        if st.button("Load Dataset"):
            with st.spinner("Loading data..."):
                from src.data_loader import load_all_data
                data = load_all_data()
                incidents = data["incidents"]
                posts = data["posts"]

            st.success(f"Loaded {len(incidents):,} incidents and {len(posts):,} posts")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Incidents", f"{len(incidents):,}")
            with col2:
                labeled = incidents["tags"].notna().sum()
                st.metric("Labeled", f"{labeled:,} ({100*labeled/len(incidents):.0f}%)")
            with col3:
                st.metric("Total Posts", f"{len(posts):,}")
            with col4:
                n_causes = incidents["tags"].dropna().str.split("|").explode().nunique()
                st.metric("Unique Causes", int(n_causes))

            with st.expander("Label Distribution", expanded=True):
                import matplotlib.pyplot as plt
                from collections import Counter

                all_tags = incidents["tags"].dropna().str.split("|").explode()
                tag_counts = Counter(all_tags)

                fig, ax = plt.subplots(figsize=(10, 5))
                labels, counts = zip(*tag_counts.most_common(15))
                ax.barh(list(labels)[::-1], list(counts)[::-1], color="steelblue")
                ax.set_xlabel("Count")
                ax.set_title("Oil Spill Cause Distribution (Labeled Incidents)")
                st.pyplot(fig)


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════

def show_training():
    st.title("⚙️ Model Training")
    st.markdown("Configure and train classification models on labeled oil spill data.")

    # Model selection
    st.markdown("### Select Models to Train")
    col1, col2, col3 = st.columns(3)
    with col1:
        train_baseline = st.checkbox("TF-IDF Baseline", value=True)
    with col2:
        train_distilbert = st.checkbox("DistilBERT")
    with col3:
        train_mlx = st.checkbox("Phi-3.5 MLX LoRA", disabled=False)

    # Hyperparameters (in expanders)
    if train_baseline:
        with st.expander("TF-IDF Baseline Parameters", expanded=False):
            from src import config as cfg
            tfidf_features = st.number_input("Max Features", 1000, 50000, cfg.TFIDF_MAX_FEATURES, step=1000)
            tfidf_c = st.number_input("C (regularization)", 0.01, 10.0, cfg.BASELINE_C, step=0.1)

    if train_distilbert:
        with st.expander("DistilBERT Parameters", expanded=False):
            bert_epochs = st.slider("Epochs", 1, 10, 4)
            bert_batch = st.slider("Batch Size", 2, 16, 8, step=2)
            bert_lr = st.number_input("Learning Rate", 1e-6, 1e-4, 2e-5, format="%.1e")

    if train_mlx:
        with st.expander("MLX LLM Parameters", expanded=False):
            lora_rank = st.slider("LoRA Rank", 4, 64, 16, step=4)
            mlx_epochs = st.slider("Epochs", 1, 6, 3)
            mlx_batch = st.slider("Batch Size", 1, 4, 2)

    # Data options
    st.markdown("### Data Options")
    use_external = st.checkbox("Include external datasets (PHMSA, Purdue, Kaggle)", value=False)

    # Train button
    st.markdown("---")
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        st.info("Training pipeline starting...")

        with st.spinner("Loading and preprocessing data..."):
            from src.preprocessing import run_preprocessing_pipeline
            result = run_preprocessing_pipeline(save_artifacts=True)

        incidents = result["incidents"]
        label_matrix = result["label_matrix"]
        label_names = result["label_names"]
        splits = result["splits"]
        has_labels = result["has_labels"]
        labeled_df = incidents[has_labels].reset_index(drop=True)

        train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]
        X_train = labeled_df.loc[train_idx, "document"].tolist()
        y_train = label_matrix[train_idx]
        X_val = labeled_df.loc[val_idx, "document"].tolist()
        y_val = label_matrix[val_idx]

        st.success(f"Data ready: {len(X_train)} train, {len(X_val)} val samples")

        results_summary = {}

        if train_baseline:
            st.markdown("#### TF-IDF Baseline")
            progress = st.progress(0, "Training TF-IDF...")
            from src.models.baseline import BaselineClassifier

            bl = BaselineClassifier(max_features=tfidf_features, C=tfidf_c)
            bl.fit(X_train, y_train, X_val, y_val)
            bl.save(str(PROJECT_ROOT / "models" / "tfidf"))
            progress.progress(100, "Done!")
            st.success("✅ TF-IDF Baseline trained and saved")

        if train_distilbert:
            st.markdown("#### DistilBERT")
            progress = st.progress(0, "Training DistilBERT (may take 30+ min on MPS)...")
            from src.models.distilbert_model import DistilBertClassifier

            db = DistilBertClassifier(epochs=bert_epochs, batch_size=bert_batch, lr=bert_lr)
            db.fit(X_train, y_train, X_val, y_val)
            db.save(str(PROJECT_ROOT / "models" / "distilbert"))
            progress.progress(100, "Done!")
            st.success("✅ DistilBERT trained and saved")

        if train_mlx:
            st.markdown("#### Phi-3.5 MLX LoRA")
            progress = st.progress(0, "Training MLX LLM (this may take 2+ hours)...")
            try:
                from src.models.llm_mlx import MLXLLMClassifier
                mlx_clf = MLXLLMClassifier(
                    lora_rank=lora_rank,
                    epochs=mlx_epochs,
                    batch_size=mlx_batch,
                )
                mlx_clf.fit(X_train, y_train, X_val, y_val)
                mlx_clf.save(str(PROJECT_ROOT / "models" / "phi"))
                progress.progress(100, "Done!")
                st.success("✅ Phi-3.5 MLX LLM trained and saved")
            except ImportError as e:
                st.error(f"Cannot train MLX model: {e}")
                st.info("Install MLX: `pip install mlx mlx-lm`")

        st.balloons()
        st.success("🎉 All selected models trained successfully!")

    # Training history placeholder
    st.markdown("---")
    st.markdown("### Training History")
    st.info("Training runs will appear here after execution.")


# ═══════════════════════════════════════════════════════════════════════
# Prediction
# ═══════════════════════════════════════════════════════════════════════

def show_prediction():
    st.title("🏷️ Prediction")
    st.markdown("Classify oil spill causes for new incidents.")

    tab1, tab2 = st.tabs(["Single Incident", "Batch Prediction"])

    # ── Single prediction ────────────────────────────────────────────
    with tab1:
        st.markdown("### Enter Incident Details")

        description = st.text_area(
            "Incident Description *",
            height=120,
            placeholder="Describe the oil spill incident...",
        )

        posts_text = st.text_area(
            "Associated Posts (optional, one per line)",
            height=80,
            placeholder="Post 1 about the incident...\nPost 2 about the response...",
        )

        commodity = st.text_input("Commodity (optional)", placeholder="e.g., crude oil, diesel")

        col1, col2 = st.columns(2)
        with col1:
            model_choice = st.selectbox(
                "Model",
                ["auto (best available)", "baseline", "distilbert", "mlx_llm"],
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("🔍 Predict", type="primary", use_container_width=True)

        if predict_btn and description:
            with st.spinner("Running prediction..."):
                from src.predict import Predictor

                predictor = Predictor()
                loaded = predictor.load_all_models()

                if loaded == 0:
                    st.error("No trained models found. Train a model first in the Training tab.")
                else:
                    posts_list = (
                        [p.strip() for p in posts_text.split("\n") if p.strip()]
                        if posts_text else None
                    )
                    model_key = None if "auto" in model_choice else model_choice

                    result = predictor.predict_single(
                        description=description,
                        posts=posts_list,
                        commodity=commodity,
                        model_key=model_key or "all",
                    )

                    # Display results
                    st.markdown("### Results")
                    st.markdown(f"**Model used:** `{result['model_used']}`")

                    if result["predicted_causes"]:
                        st.success(f"Causes: **{', '.join(result['predicted_causes'])}**")
                    else:
                        st.warning("No specific cause identified.")

                    # Probability bars
                    if result["probabilities"]:
                        st.markdown("#### Confidence Scores")
                        import plotly.express as px

                        prob_df = (
                            pd.DataFrame(
                                result["probabilities"].items(),
                                columns=["Cause", "Probability"],
                            )
                            .sort_values("Probability", ascending=True)
                        )
                        fig = px.bar(
                            prob_df, x="Probability", y="Cause",
                            orientation="h", range_x=[0, 1],
                            title="Per-Cause Probability",
                            color="Probability",
                            color_continuous_scale="Blues",
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)

                    if result.get("commodity_category"):
                        st.caption(f"Commodity category: {result['commodity_category']}")

    # ── Batch prediction ─────────────────────────────────────────────
    with tab2:
        st.markdown("### Upload Batch File")
        batch_file = st.file_uploader(
            "CSV with incident descriptions",
            type=["csv"],
            key="batch_upload",
        )

        if batch_file is not None:
            import pandas as pd
            batch_df = pd.read_csv(batch_file)
            st.dataframe(batch_df.head(10), use_container_width=True)

            description_col = st.selectbox(
                "Select description column",
                batch_df.columns.tolist(),
            )

            if st.button("🔍 Run Batch Prediction", type="primary"):
                with st.spinner(f"Predicting {len(batch_df)} incidents..."):
                    from src.predict import Predictor

                    predictor = Predictor()
                    predictor.load_all_models()

                    results = predictor.predict_batch(
                        batch_df,
                        description_col=description_col,
                    )

                st.success(f"Predicted {len(results)} incidents")
                st.dataframe(results, use_container_width=True)

                # Download button
                csv = results.to_csv(index=False)
                st.download_button(
                    "📥 Download Results (CSV)",
                    csv,
                    "oil_spill_predictions.csv",
                    "text/csv",
                )


# ═══════════════════════════════════════════════════════════════════════
# Reports
# ═══════════════════════════════════════════════════════════════════════

def show_reports():
    st.title("📄 Reports")
    st.markdown("View and download evaluation reports.")

    # Check for existing reports
    reports_dir = PROJECT_ROOT / "output" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    existing = list(reports_dir.glob("*.html"))

    if existing:
        st.markdown("### Existing Reports")
        for report_path in sorted(existing, reverse=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"📄 `{report_path.name}`")
            with col2:
                st.caption(f"{report_path.stat().st_size / 1024:.0f} KB")

    st.markdown("---")
    st.markdown("### Generate New Report")

    if st.button("📊 Generate Report", type="primary"):
        st.info("Run training and evaluation first, then generate reports from the Evaluation tab.")
        st.markdown(
            """
            To generate a full report:
            1. Train at least one model in the **Training** tab
            2. Run evaluation from the CLI: `python -m src.evaluate`
            3. Reports will appear here
            """
        )


# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
