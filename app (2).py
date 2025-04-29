import streamlit as st
import joblib
import numpy as np
import base64

# Load model and features
model = joblib.load("lung_model.joblib")
features = joblib.load("features.joblib")

# Set background
def set_background(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    encoded = f"data:image/png;base64,{base64.b64encode(data).decode()}"
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            font-family: 'Segoe UI', sans-serif;
        }}
        .main-container {{
            background-color: rgba(255, 255, 255, 0.9);
            padding: 30px;
            border-radius: 15px;
            margin-top: 20px;
        }}
        </style>
    """, unsafe_allow_html=True)

set_background("static/background.png")  # Make sure this file exists

# Title
st.markdown("""
    <h1 style='text-align: center; color: white; font-size: 50px;'>
        🫁 Lung Cancer Classification App
    </h1>
    <h4 style='text-align: center; color: white; font-weight: normal;'>
        Get a fast and accurate prediction based on your symptoms
    </h4>
""", unsafe_allow_html=True)

# Main input container
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

st.subheader("📝 Personal & Medical Information")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 20, 100, 45)

st.subheader("🩺 Symptoms & Conditions")

smoking = st.selectbox("Do you smoke?", ["Yes", "No"])
anxiety = st.selectbox("Do you feel anxious?", ["Yes", "No"])
fatigue = st.selectbox("Do you feel fatigue?", ["Yes", "No"])
weight_loss = st.selectbox("Have you experienced weight loss?", ["Yes", "No"])
wheezing = st.selectbox("Do you experience wheezing?", ["Yes", "No"])
coughing = st.selectbox("Do you cough frequently?", ["Yes", "No"])
alcohol = st.selectbox("Do you consume alcohol?", ["Yes", "No"])
chronic_disease = st.selectbox("Do you have a chronic disease?", ["Yes", "No"])

st.markdown("</div>", unsafe_allow_html=True)

# Encode input
input_data = {
    "GENDER": 1 if gender == "Male" else 0,
    "AGE": age,
    "SMOKING": 1 if smoking == "Yes" else 0,
    "ANXIETY": 1 if anxiety == "Yes" else 0,
    "FATIGUE": 1 if fatigue == "Yes" else 0,
    "WEIGHTLOSS": 1 if weight_loss == "Yes" else 0,
    "WHEEZING": 1 if wheezing == "Yes" else 0,
    "COUGHING": 1 if coughing == "Yes" else 0,
    "ALCOHOLCONSUMING": 1 if alcohol == "Yes" else 0,
    "CHRONICDISEASE": 1 if chronic_disease == "Yes" else 0
}

input_list = [input_data.get(feat, 0) for feat in features]
input_array = np.array(input_list).reshape(1, -1)

# Predict
if st.button("🔍 Predict", key="predict_button"):
    result = model.predict(input_array)[0]
    if result == 0:
        st.success("✅ Result: Normal")
    elif result == 1:
        st.warning("⚠️ Result: Benign (Non-cancerous)")
    else:
        st.error("❌ Result: Malignant (Cancerous)")
