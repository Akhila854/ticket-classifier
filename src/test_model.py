import joblib

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

text = ["I cannot log into my account"]

text_vec = vectorizer.transform(text)

prediction = model.predict(text_vec)

print("Prediction:", prediction[0])