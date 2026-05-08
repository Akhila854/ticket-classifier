# 🚀 Smart Support Ticket Classifier

An end-to-end NLP-based machine learning system that classifies customer support queries and serves real-time predictions via a FastAPI API.

---

## 🔥 Features

* Real-time text classification API using FastAPI
* Interactive Streamlit UI for predictions
* TF-IDF + Logistic Regression NLP model
* Prediction logging for monitoring and retraining (MLOps)
* Dockerized for deployment
* Modular project structure for scalability

---

## 🚀 Demo

### API Request

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
[Streamlit UI] → [FastAPI API] → [TF-IDF Vectorizer] → [ML Model] → [Prediction + Logging]
```

---

## ⚙️ Tech Stack

* Python
* Scikit-learn
* FastAPI
* Streamlit
* Docker
* Git

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
## 🎨Run Streamlit Frontend
```
streamlit run src/app.py

```
http://localhost:8501
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

* Monitoring prediction behavior
* Debugging incorrect predictions
* Collecting data for future retraining

---

## 📁 Project Structure

```
ticket-classifier/
├── src/
│   ├── main.py
│   ├── train.py
│   ├── test_model.py
│   └── app.py
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

* End-to-end NLP application
* Real-time prediction API
* Interactive frontend UI
* Logging for MLOps monitoring
* Dockerized for deployment
* Production-oriented project structure
