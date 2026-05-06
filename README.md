# 🚀 Smart Support Ticket Classifier

An end-to-end NLP-based machine learning system that classifies customer support queries and serves real-time predictions via a FastAPI API.

---

## 🔥 Features

* Real-time text classification API (FastAPI)
* TF-IDF + Logistic Regression model
* Prediction logging for monitoring (MLOps)
* Dockerized for deployment

---

## 🚀 Demo

### Request

POST `/predict`

```json
{
  "text": "I forgot my password"
}
```

### Response

```json
{
  "category": "Account",
  "confidence": 0.91
}
```

---

## 🏗️ Architecture

```
[Input Text] → [TF-IDF] → [ML Model] → [FastAPI] → [Prediction + Logging]
```

---

## ⚙️ Tech Stack

* Python
* Scikit-learn
* FastAPI
* Docker

---

## 🖥️ How to Run Locally

```bash
git clone https://github.com/Akhila854/ticket-classifier.git
cd ticket-classifier

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt

uvicorn src.main:app --reload --port 8001
```

Open in browser:

```
http://127.0.0.1:8001/docs
```

---

## 🐳 Run with Docker

```bash
docker build -t ticket-classifier .
docker run -p 8000:8000 ticket-classifier
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## 📊 Logging

Predictions are stored in:

```
logs/predictions.jsonl
```

This enables:

* Monitoring model performance
* Debugging predictions
* Collecting data for future retraining

---

## 📁 Project Structure

```
ticket-classifier/
├── src/
├── data/
├── models/
├── logs/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚠️ Note

Model files and logs are excluded from the repository and generated during runtime.

---

## 🎯 Key Highlights

* End-to-end ML system (not just a notebook)
* Real-time API with FastAPI
* Logging for production monitoring
* Dockerized for deployment
