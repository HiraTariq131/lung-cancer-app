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
    encoded = base64.b64encode(data).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
    """, unsafe_allow_html=True)

# Use your uploaded background image here 👇
set_background("/mnt/data/8aa572a2-f861-4fd2-81de-2ecf6f001670.png")

# Stylish Title
st.markdown("""
    <h1 style='text-align: center; color: white; font-size: 50px;'>🫁 Lung Cancer Prediction</h1>
    <h4 style='text-align: center; color: white;'>Enter your health details to check the result</h4>
""", unsafe_allow_html=True)

# Input Section in light box
st.markdown("<div style='padding: 25px; background-color: rgba(255,255,255,0.90); border-radius: 15px;'>", unsafe_allow_html=True)

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 20, 100, 45)
smoking = st.selectbox("Do you smoke?", ["Yes", "No"])
anxiety = st.selectbox("Do you feel anxious?", ["Yes", "No"])
fatigue = st.selectbox("Do you feel fatigue?", ["Yes", "No"])
weight_loss = st.selectbox("Have you experienced weight loss?", ["Yes", "No"])
cough = st.selectbox("Do you have persistent cough?", ["Yes", "No"])
shortness_breath = st.selectbox("Do you feel shortness of breath?", ["Yes", "No"])
chest_pain = st.selectbox("Do you feel chest pain?", ["Yes", "No"])

st.markdown("</div>", unsafe_allow_html=True)

# Convert input to model format
input_data = {
    "GENDER": 1 if gender == "Male" else 0,
    "AGE": age,
    "SMOKING": 1 if smoking == "Yes" else 0,
    "ANXIETY": 1 if anxiety == "Yes" else 0,
    "FATIGUE": 1 if fatigue == "Yes" else 0,
    "WEIGHTLOSS": 1 if weight_loss == "Yes" else 0,
    "COUGH": 1 if cough == "Yes" else 0,
    "SHORTNESSOFBREATH": 1 if shortness_breath == "Yes" else 0,
    "CHESTPAIN": 1 if chest_pain == "Yes" else 0,
}

# Arrange features in order
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
