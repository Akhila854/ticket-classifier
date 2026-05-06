from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
from datetime import datetime
import os

app = FastAPI()

# Load model
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Request schema
class Request(BaseModel):
    text: str

# Ensure logs folder exists
os.makedirs("logs", exist_ok=True)

@app.get("/")
def home():
    return {"message": "Ticket Classifier API is running"}

@app.post("/predict")
def predict(req: Request):
    text_vec = vectorizer.transform([req.text])

    prediction = model.predict(text_vec)[0]
    confidence = float(model.predict_proba(text_vec).max())

    # Create log entry
    log_entry = {
        "text": req.text,
        "prediction": prediction,
        "confidence": confidence,
        "timestamp": datetime.utcnow().isoformat()
    }

    # Save to file
    with open("logs/predictions.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {
        "category": prediction,
        "confidence": confidence
    }