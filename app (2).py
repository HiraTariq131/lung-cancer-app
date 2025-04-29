
import streamlit as st
import numpy as np
import joblib
from PIL import Image

# Load model and features
model = joblib.load("lung_model.pkl")
features = joblib.load("features.joblib")

st.set_page_config(page_title="Lung Cancer Classifier", layout="centered")

st.markdown("""
    <style>
    .main {
        background-image: url('https://i.imgur.com/JzAJkG7.jpg');
        background-size: cover;
        padding: 2rem;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🫁 Lung Cancer Prediction App")
st.markdown("Provide the following health details:")

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

    encoding = {"Male": 1, "Female": 2, "Yes": 1, "No": 2}
    encoded_data = [encoding.get(val, val) for val in data]
    return np.array(encoded_data).reshape(1, -1)

if st.button("Predict"):
    input_data = user_input()
    pred = model.predict(input_data)[0]
    result = "Likely Lung Cancer ❗" if pred == 1 else "No Lung Cancer ✅"
    st.subheader("🩺 Prediction Result:")
    st.success(result)
