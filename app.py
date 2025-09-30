import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("liver_model.joblib")

st.title("🧪 Liver Disease Prediction Demo")
st.write("Enter patient details to predict likelihood of liver disease")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=40)
gender = st.selectbox("Gender", ["Male", "Female"])
tb = st.number_input("Total Bilirubin", min_value=0.0, value=1.0)
db = st.number_input("Direct Bilirubin", min_value=0.0, value=0.3)
alkphos = st.number_input("Alkaline Phosphatase", min_value=0.0, value=200.0)
alt = st.number_input("ALT (SGPT)", min_value=0.0, value=30.0)
ast = st.number_input("AST (SGOT)", min_value=0.0, value=35.0)
tp = st.number_input("Total Proteins", min_value=0.0, value=6.8)
albumin = st.number_input("Albumin", min_value=0.0, value=3.5)
ag_ratio = st.number_input("A/G Ratio", min_value=0.0, value=1.0)

# Prepare input for prediction
gender_enc = 1 if gender == "Male" else 0
features = np.array([[age, gender_enc, tb, db, alkphos, alt, ast, tp, albumin, ag_ratio]])

# Predict
if st.button("Predict"):
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]
    if pred == 1:
        st.error(f"⚠️ Likely Liver Disease (Probability: {prob:.2f})")
    else:
        st.success(f"✅ No Liver Disease Detected (Probability: {prob:.2f})")
