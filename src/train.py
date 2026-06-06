import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import json
import logging
import mlflow
import mlflow.sklearn

from model import TicketClassifier

logging.basicConfig(level=logging.INFO)

# ── MLflow setup ─────────────────────────────────────────
mlflow.set_experiment("ticket-classifier")

# Load data
df = pd.read_csv("data/data.csv")
print(f"Loaded {len(df)} rows")
print(f"Categories: {df['category'].value_counts().to_dict()}")

df = df.dropna(subset=["text", "category"])
df = df[df.groupby("category")["category"].transform("count") >= 10]

print(f"Training on {len(df)} rows across {df['category'].nunique()} categories")

# ── Train with MLflow tracking ────────────────────────────
with mlflow.start_run():

    # Log parameters
    mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
    mlflow.log_param("classifier", "LogisticRegression")
    mlflow.log_param("train_size", int(len(df) * 0.8))
    mlflow.log_param("test_size", int(len(df) * 0.2))
    mlflow.log_param("num_categories", df["category"].nunique())
    mlflow.log_param("total_samples", len(df))
    mlflow.log_param("class_distribution",
                     str(df["category"].value_counts().to_dict()))

    # Train
    clf = TicketClassifier()
    metrics = clf.train(
        texts=df["text"].tolist(),
        labels=df["category"].tolist()
    )

    # Log metrics
    mlflow.log_metric("f1_weighted", metrics["f1_weighted"])
    mlflow.log_metric("train_size", metrics["train_size"])
    mlflow.log_metric("test_size", metrics["test_size"])

    # Log per-class F1 scores from the report
    report = metrics.get("report", {})
    for label, scores in report.items():
        if isinstance(scores, dict) and "f1-score" in scores:
            clean_label = label.replace(" ", "_")
            mlflow.log_metric(f"f1_{clean_label}", round(scores["f1-score"], 4))

    # Save model
    clf.save("models")
    mlflow.sklearn.log_model(clf.classifier, "classifier")

    # Save metrics JSON
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    mlflow.log_artifact("metrics.json")

    run_id = mlflow.active_run().info.run_id

print("\n✅ Training complete!")
print(f"F1 Score (weighted): {metrics['f1_weighted']}")
print(f"Train size: {metrics['train_size']}")
print(f"Test size:  {metrics['test_size']}")
print(f"\nMLflow run ID: {run_id}")
print("Run 'mlflow ui' to view experiment dashboard")
print("\nMetrics saved to metrics.json")
print("Model saved to models/")