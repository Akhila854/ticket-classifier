import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import requests

st.set_page_config(page_title="Support Ticket Classifier", page_icon="🎫")

st.title("🎫 Support Ticket Classifier")
st.markdown("Classify IT support tickets using LLM-powered embeddings.")

API_URL = st.sidebar.text_input("API URL", value="http://localhost:8000")
API_KEY = st.sidebar.text_input("API Key", value="dev-secret-key-change-in-prod", type="password")

st.sidebar.markdown("---")

# Health check
try:
    health = requests.get(f"{API_URL}/health", timeout=3)
    if health.json().get("model_loaded"):
        st.sidebar.success("✅ Model loaded")
    else:
        st.sidebar.warning("⚠️ Model not loaded yet")
except Exception:
    st.sidebar.error("❌ API not reachable")

st.markdown("---")

# Single prediction
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
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": ticket_text},
                    headers={"X-API-Key": API_KEY},
                    timeout=10
                )
                if response.status_code == 200:
                    result = response.json()
                    col1, col2 = st.columns(2)
                    col1.metric("Category", result["label"])
                    col2.metric("Confidence", f"{result['confidence']*100:.1f}%")
                    st.caption(f"Request ID: {result['request_id']}")
                elif response.status_code == 401:
                    st.error("Invalid API key.")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Could not reach API: {e}")

st.markdown("---")

# Batch prediction
st.subheader("Batch Prediction")
uploaded = st.file_uploader("Upload a CSV with a 'text' column", type=["csv"])

if uploaded:
    import pandas as pd
    df = pd.read_csv(uploaded)
    if "text" not in df.columns:
        st.error("CSV must have a column named 'text'")
    else:
        if st.button("Run Batch Classification"):
            results = []
            progress = st.progress(0)
            for i, row in df.iterrows():
                try:
                    r = requests.post(
                        f"{API_URL}/predict",
                        json={"text": str(row["text"])},
                        headers={"X-API-Key": API_KEY},
                        timeout=10
                    )
                    if r.status_code == 200:
                        res = r.json()
                        results.append({
                            "text": row["text"],
                            "label": res["label"],
                            "confidence": res["confidence"]
                        })
                except Exception:
                    results.append({"text": row["text"], "label": "ERROR", "confidence": 0.0})
                progress.progress((i + 1) / len(df))

            result_df = pd.DataFrame(results)
            st.dataframe(result_df)
            st.download_button(
                "Download Results",
                result_df.to_csv(index=False),
                "predictions.csv",
                "text/csv"
            )