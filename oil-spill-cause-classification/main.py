#!/usr/bin/env python3
"""
Oil Spill Cause Classification — CLI Entry Point
=================================================
Usage:
    python main.py train --model all
    python main.py predict --input incident.json
    python main.py evaluate --model baseline
    python main.py report --format html
    python main.py serve
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-5s | %(name)s | %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Oil Spill Cause Classification — Multi-Label NLP System",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ─────────────────────────────────────────────────────
    train = sub.add_parser("train", help="Train one or all models")
    train.add_argument(
        "--model",
        choices=["baseline", "distilbert", "mlx_llm", "all"],
        default="all",
    )
    train.add_argument("--use-external-data", action="store_true")
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--output-dir", default="models")

    # ── evaluate ──────────────────────────────────────────────────
    eval_p = sub.add_parser("evaluate", help="Evaluate trained models")
    eval_p.add_argument(
        "--model",
        nargs="+",
        choices=["baseline", "distilbert", "mlx_llm", "all"],
        default=["all"],
    )
    eval_p.add_argument("--output-dir", default="output")

    # ── predict ───────────────────────────────────────────────────
    pred = sub.add_parser("predict", help="Run inference on new incidents")
    pred.add_argument("input", help="CSV file path or JSON string")
    pred.add_argument(
        "--model",
        choices=["baseline", "distilbert", "mlx_llm", "all"],
        default="all",
    )
    pred.add_argument("--output", default=None, help="Output CSV path")
    pred.add_argument("--description-col", default="description")

    # ── report ────────────────────────────────────────────────────
    report = sub.add_parser("report", help="Generate evaluation report")
    report.add_argument("--format", choices=["html", "json"], default="html")
    report.add_argument("--output", default=None)

    # ── serve ─────────────────────────────────────────────────────
    sub.add_parser("serve", help="Launch Streamlit web app")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "serve":
        cmd_serve(args)


# ═══════════════════════════════════════════════════════════════════════


def cmd_train(args):
    """Train model(s)."""
    from src.preprocessing import run_preprocessing_pipeline

    print("=" * 60)
    print("OIL SPILL CAUSE CLASSIFICATION — TRAINING")
    print("=" * 60)

    # Data
    print("\n[1/3] Loading and preprocessing data...")
    result = run_preprocessing_pipeline(save_artifacts=True)

    incidents = result["incidents"]
    label_matrix = result["label_matrix"]
    splits = result["splits"]
    has_labels = result["has_labels"]
    labeled_df = incidents[has_labels].reset_index(drop=True)

    train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]
    X_train = labeled_df.loc[train_idx, "document"].tolist()
    y_train = label_matrix[train_idx]
    X_val = labeled_df.loc[val_idx, "document"].tolist()
    y_val = label_matrix[val_idx]

    print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(test_idx)}")
    print(f"   Labels: {y_train.shape[1]} | Imbalance ratio: {y_train.sum(axis=0).max() / max(y_train.sum(axis=0).min(), 1):.1f}x")

    # Train
    models_to_train = (
        ["baseline", "distilbert", "mlx_llm"]
        if args.model == "all"
        else [args.model]
    )

    for model_name in models_to_train:
        print(f"\n[2/3] Training {model_name}...")

        if model_name == "baseline":
            from src.models.baseline import BaselineClassifier

            clf = BaselineClassifier()
            clf.fit(X_train, y_train, X_val, y_val)
            clf.save(str(Path(args.output_dir) / "tfidf"))
            print(f"   Baseline trained. Thresholds: {clf.thresholds_}")

        elif model_name == "distilbert":
            from src.models.distilbert_model import DistilBertClassifier

            epochs = args.epochs or 4
            clf = DistilBertClassifier(epochs=epochs)
            clf.fit(X_train, y_train, X_val, y_val)
            clf.save(str(Path(args.output_dir) / "distilbert"))
            print("   DistilBERT trained and saved.")

        elif model_name == "mlx_llm":
            try:
                from src.models.llm_mlx import MLXLLMClassifier

                epochs = args.epochs or 3
                clf = MLXLLMClassifier(epochs=epochs)
                clf.fit(X_train, y_train, X_val, y_val)
                clf.save(str(Path(args.output_dir) / "phi"))
                print("   MLX LLM trained and saved.")
            except ImportError as e:
                print(f"   Skipping MLX LLM: {e}")

    print("\n[3/3] Training complete!")
    print(f"   Models saved to: {Path(args.output_dir).resolve()}")


def cmd_evaluate(args):
    """Evaluate trained models."""
    from src.preprocessing import run_preprocessing_pipeline
    from src.evaluate import compute_metrics, format_metrics_summary
    from src.models.baseline import BaselineClassifier
    from src.models.distilbert_model import DistilBertClassifier

    print("=" * 60)
    print("OIL SPILL CAUSE CLASSIFICATION — EVALUATION")
    print("=" * 60)

    # Data
    print("\nLoading data...")
    result = run_preprocessing_pipeline(save_artifacts=False)
    incidents = result["incidents"]
    label_matrix = result["label_matrix"]
    label_names = result["label_names"]
    splits = result["splits"]
    has_labels = result["has_labels"]
    labeled_df = incidents[has_labels].reset_index(drop=True)
    test_idx = splits["test"]
    X_test = labeled_df.loc[test_idx, "document"].tolist()
    y_test = label_matrix[test_idx]

    models_dir = Path("models")

    all_metrics = {}

    # Evaluate each available model
    model_paths = {
        "baseline": models_dir / "tfidf",
        "distilbert": models_dir / "distilbert",
        "mlx_llm": models_dir / "phi",
    }

    for name, path in model_paths.items():
        if not path.exists():
            continue

        print(f"\nEvaluating {name}...")
        try:
            if name == "baseline":
                clf = BaselineClassifier.load(str(path))
            elif name == "distilbert":
                clf = DistilBertClassifier.load(str(path))
            else:
                continue

            y_prob = clf.predict_proba(X_test)
            y_pred = clf.predict(X_test)
            metrics = compute_metrics(y_test, y_pred, y_prob, label_names)
            all_metrics[name] = metrics
            print(format_metrics_summary(metrics))

        except Exception as e:
            print(f"  Failed: {e}")

    # Generate comparison report
    if all_metrics:
        from src.report import generate_html_report, generate_json_report

        html_path = generate_html_report(all_metrics)
        json_path = generate_json_report(all_metrics)
        print(f"\nReports generated:\n  HTML: {html_path}\n  JSON: {json_path}")


def cmd_predict(args):
    """Run inference."""
    from src.predict import Predictor
    import pandas as pd

    predictor = Predictor()
    loaded = predictor.load_all_models()

    if loaded == 0:
        print("No trained models found. Train first with: python main.py train")
        sys.exit(1)

    print(f"Loaded {loaded} model(s)")

    # Determine input type
    input_str = args.input

    if input_str.endswith(".csv"):
        # Batch prediction
        df = pd.read_csv(input_str)
        print(f"Predicting {len(df)} incidents from {input_str}...")
        results = predictor.predict_batch(df, description_col=args.description_col)

        output_path = args.output or input_str.replace(".csv", "_predictions.csv")
        results.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")

    elif input_str.endswith(".json"):
        # JSON input
        with open(input_str) as f:
            data = json.load(f)

        result = predictor.predict_single(
            description=data.get("description", ""),
            posts=data.get("posts"),
            commodity=data.get("commodity"),
        )

        print("\nPrediction Results:")
        print(f"  Causes: {result['predicted_causes']}")
        print(f"  Model: {result['model_used']}")
        print("\nProbabilities:")
        for cause, prob in sorted(
            result["probabilities"].items(), key=lambda x: -x[1]
        ):
            print(f"  {cause:<25s} {prob:.3f}")

    else:
        # Plain text
        result = predictor.predict_from_text(input_str)
        print(f"\nPredicted causes: {result['predicted_causes']}")
        print(f"Model: {result['model_used']}")


def cmd_report(args):
    """Generate a report from saved evaluation data."""
    from src.report import generate_html_report, generate_json_report

    report_dir = Path("output") / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Try to load saved metrics
    metrics = {}
    json_files = sorted(report_dir.glob("*.json"), reverse=True)
    if json_files:
        try:
            import json
            with open(json_files[0]) as f:
                saved = json.load(f)
            # Reconstruct metrics dict from saved format
            if "models" in saved:
                logger.info("Loading metrics from %s", json_files[0])
                # Pass raw saved data — report handles both formats
                metrics = saved.get("models", {})
        except Exception:
            logger.warning("Could not parse saved report, generating empty report")

    if args.format == "html":
        html_path = generate_html_report(
            {} if isinstance(metrics, dict) and not any(
                isinstance(v, dict) and "micro" in v for v in metrics.values()
            ) else {},
            output_path=str(report_dir / "report.html"),
        )
        print(f"HTML report saved to: {html_path}")
    elif args.format == "json":
        json_path = generate_json_report(
            {},
            output_path=str(report_dir / "report.json"),
        )
        print(f"JSON report saved to: {json_path}")


def cmd_serve(args):
    """Launch Streamlit."""
    import subprocess

    app_path = Path(__file__).parent / "app" / "streamlit_app.py"
    print(f"Launching Streamlit app: {app_path}")
    subprocess.run(["streamlit", "run", str(app_path)])


if __name__ == "__main__":
    main()
