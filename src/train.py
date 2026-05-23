import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import json
import logging
from model import TicketClassifier

logging.basicConfig(level=logging.INFO)

# Load data
df = pd.read_csv("data/data.csv")
print(f"Loaded {len(df)} rows")
print(f"Categories: {df['category'].value_counts().to_dict()}")

# Drop empty rows
df = df.dropna(subset=["text", "category"])

# Drop categories with fewer than 10 samples
df = df[df.groupby("category")["category"].transform("count") >= 10]

print(f"Training on {len(df)} rows across {df['category'].nunique()} categories")

# Train
clf = TicketClassifier()
metrics = clf.train(
    texts=df["text"].tolist(),
    labels=df["category"].tolist()
)

# Save model
clf.save("models")

# Save metrics for README and /model/info endpoint
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n✅ Training complete!")
print(f"F1 Score (weighted): {metrics['f1_weighted']}")
print(f"Train size: {metrics['train_size']}")
print(f"Test size:  {metrics['test_size']}")
print("\nMetrics saved to metrics.json")
print("Model saved to models/")