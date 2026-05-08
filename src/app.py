import streamlit as st
import requests

st.set_page_config(page_title="Ticket Classifier")

st.title("🚀 Smart Support Ticket Classifier")

st.write("Enter a customer support query below:")

text = st.text_area("Customer Query")

if st.button("Predict"):

    response = requests.post(
        "http://127.0.0.1:8001/predict",
        json={"text": text}
    )

    result = response.json()

    st.subheader("Prediction Result")

    st.write(result)