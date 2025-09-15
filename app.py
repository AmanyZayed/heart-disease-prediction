import streamlit as st
import joblib
import os
import pandas as pd



st.title("Heart Disease Risk Prediction")

MODEL_PATH = os.path.join("models", "final_model.pkl")

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found!\n\n👉 Please open `Heart_Disease_Final_Colab.ipynb` and run all cells. "
             "That will train the model and create `models/final_model.pkl`.")
    st.stop()

# Load trained bundle
bundle = joblib.load(MODEL_PATH)
scaler = bundle['scaler']
features = bundle['selected_features']
model = bundle['model']

st.write("Enter patient data below:")

inputs = {}
for f in features:
    inputs[f] = st.number_input(f, value=0.0)

df_in = pd.DataFrame([inputs])

try:
    df_in_scaled = scaler.transform(df_in)
except Exception:
    df_in_scaled = df_in.values

pred = model.predict(df_in_scaled)
prob = model.predict_proba(df_in_scaled)[:, 1] if hasattr(model, "predict_proba") else None

st.write("### Prediction:", "🟢 No disease" if pred[0] == 0 else "🔴 Disease detected")
if prob is not None:
    st.write(f"Probability: {prob[0]:.2f}")
