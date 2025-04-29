import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image

# Load model and features
model = joblib.load("lung_model.joblib")
features = joblib.load("features.joblib")

# Background Image and Styling
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://i.imgur.com/ZFzn7V3.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;
    }
    .css-1d391kg {
        background-color: rgba(0, 0, 0, 0.6);
        border-radius: 10px;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Introduction
st.title("🫁 Lung Cancer Prediction App")
st.markdown("Provide the following health details:")

# Dynamic input handling using st.selectbox or st.slider
def user_input():
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 20, 100, 50)
    smoking = st.selectbox("Do you smoke?", ["Yes", "No"])
    yellow_fingers = st.selectbox("Yellow fingers?", ["Yes", "No"])
    anxiety = st.selectbox("Do you feel anxious?", ["Yes", "No"])
    peer_pressure = st.selectbox("Peer pressure exposure?", ["Yes", "No"])
    chronic_disease = st.selectbox("Chronic disease?", ["Yes", "No"])
    fatigue = st.selectbox("Fatigue experience?", ["Yes", "No"])
    allergy = st.selectbox("Do you have allergies?", ["Yes", "No"])
    wheezing = st.selectbox("Wheezing symptoms?", ["Yes", "No"])
    alcohol = st.selectbox("Alcohol consumption?", ["Yes", "No"])
    coughing = st.selectbox("Do you cough regularly?", ["Yes", "No"])
    short_breath = st.selectbox("Shortness of breath?", ["Yes", "No"])
    swallowing_diff = st.selectbox("Swallowing difficulty?", ["Yes", "No"])
    chest_pain = st.selectbox("Chest pain?", ["Yes", "No"])

    data = [gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
            chronic_disease, fatigue, allergy, wheezing, alcohol, coughing,
            short_breath, swallowing_diff, chest_pain]

    # Encoding: Yes/No → 1/2, Male/Female → 1/2
    encoding = {"Male": 1, "Female": 2, "Yes": 1, "No": 2}
    encoded_data = [encoding.get(val, val) for val in data]
    return np.array(encoded_data).reshape(1, -1)

# Button for prediction
if st.button("Predict"):
    input_data = user_input()
    pred = model.predict(input_data)[0]
    result = "Likely Lung Cancer ❗" if pred == 1 else "No Lung Cancer ✅"
    
    # Display the prediction result with styling
    st.subheader("🩺 Prediction Result:")
    st.success(result)
