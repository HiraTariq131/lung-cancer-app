import streamlit as st
import joblib
import numpy as np

# Load the trained model and features
model = joblib.load("lung_model.joblib")
features = joblib.load("features.joblib")

# Set custom page background using inline CSS
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://cdn.pixabay.com/photo/2016/10/24/17/14/lungs-1761277_1280.jpg");
        background-size: cover;
        background-position: center;
        color: white;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        color: white;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        font-size: 16px;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🫁 Lung Cancer Prediction App</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Provide the following health details:</div>", unsafe_allow_html=True)

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 20, 100, 50)
smoking = st.selectbox("Do you smoke?", ["Yes", "No"])
anxiety = st.selectbox("Do you feel anxious?", ["Yes", "No"])
fatigue = st.selectbox("Do you feel fatigue?", ["Yes", "No"])
weight_loss = st.selectbox("Have you experienced weight loss?", ["Yes", "No"])

# Map inputs to numeric
input_data = {
    "GENDER": 1 if gender == "Male" else 0,
    "AGE": age,
    "SMOKING": 1 if smoking == "Yes" else 0,
    "ANXIETY": 1 if anxiety == "Yes" else 0,
    "FATIGUE": 1 if fatigue == "Yes" else 0,
    "WEIGHTLOSS": 1 if weight_loss == "Yes" else 0
}

# Extract feature values in correct order
input_list = [input_data.get(feat, 0) for feat in features]
input_array = np.array(input_list).reshape(1, -1)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_array)[0]
    if prediction == 0:
        st.success("✅ Result: Normal")
    elif prediction == 1:
        st.warning("⚠️ Result: Benign (Non-cancerous)")
    else:
        st.error("❌ Result: Malignant (Cancerous)")
