import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib
import os

st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="🎫",
    layout="centered"
)

# ── Load model (cached so it only loads once) ─────────────
@st.cache_resource
def load_model():
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    classifier = joblib.load("models/classifier.joblib")
    return encoder, classifier

encoder, classifier = load_model()

# ── UI ────────────────────────────────────────────────────
st.title("🎫 Smart Support Ticket Classifier")
st.markdown(
    "Classify IT support tickets using **LLM-powered embeddings** "
    "(sentence-transformers + Logistic Regression)."
)
st.markdown("---")

# ── Single prediction ─────────────────────────────────────
st.subheader("Single Prediction")
ticket_text = st.text_area(
    "Enter ticket description:",
    placeholder="e.g. My laptop screen went black and won't turn on...",
    height=120
)

if st.button("Classify Ticket", type="primary"):
    if not ticket_text.strip():
        st.warning("Please enter a ticket description.")
    else:
        with st.spinner("Classifying..."):
            embedding = encoder.encode([ticket_text])
            label = classifier.predict(embedding)[0]
            proba = classifier.predict_proba(embedding)[0]
            confidence = float(np.max(proba))

            col1, col2 = st.columns(2)
            col1.metric("Category", label)
            col2.metric("Confidence", f"{confidence*100:.1f}%")

            # Show all probabilities
            st.markdown("#### Confidence per category")
            classes = classifier.classes_
            proba_dict = dict(zip(classes, proba))
            for cat, prob in sorted(
                proba_dict.items(), key=lambda x: x[1], reverse=True
            ):
                st.progress(float(prob), text=f"{cat}: {prob*100:.1f}%")

st.markdown("---")

# ── Batch prediction ──────────────────────────────────────
st.subheader("Batch Prediction")
uploaded = st.file_uploader(
    "Upload a CSV with a 'text' column", type=["csv"]
)

if uploaded:
    import pandas as pd
    df = pd.read_csv(uploaded)
    if "text" not in df.columns:
        st.error("CSV must have a column named 'text'")
    else:
        if st.button("Run Batch Classification"):
            with st.spinner("Classifying all tickets..."):
                embeddings = encoder.encode(
                    df["text"].tolist(), show_progress_bar=False
                )
                labels = classifier.predict(embeddings)
                probas = classifier.predict_proba(embeddings)
                confidences = np.max(probas, axis=1)

                df["predicted_category"] = labels
                df["confidence"] = (confidences * 100).round(1)

                st.dataframe(df)
                st.download_button(
                    "Download Results",
                    df.to_csv(index=False),
                    "predictions.csv",
                    "text/csv"
                )

st.markdown("---")
st.markdown(
    "**Model:** sentence-transformers/all-MiniLM-L6-v2 + "
    "LogisticRegression | **F1:** 0.9924 | "
    "[GitHub](https://github.com/Akhila854/ticket-classifier) | "
    "[Hugging Face](https://huggingface.co/akhilaarekal/ticket-classifier)"
)