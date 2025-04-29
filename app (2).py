import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

st.set_page_config(layout="centered")

# Custom background and styling — remove black overlay
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background-image: url('https://raw.githubusercontent.com/yourusername/yourrepo/main/lung-bg.png'); /* Change to your image URL */
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }
        h1, h2, h3, h4, h5, h6, label, .stButton>button {
            color: white !important;
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px;
        }
        .stButton>button:hover {
            background-color: #ff1c1c;
        }
    </style>
""", unsafe_allow_html=True)

# Title & Header
st.markdown("<h1 style='text-align: center;'>🩺 Lung Cancer Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Classifying: Normal | Benign | Malignant</h3>", unsafe_allow_html=True)
st.markdown("<h4>🔍 Enter Patient Details Below</h4>", unsafe_allow_html=True)

# Input form
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 1, 100, 30)
smoking = st.selectbox("Smoking", ["Yes", "No"])
yellow_fingers = st.selectbox("Yellow Fingers", ["Yes", "No"])
anxiety = st.selectbox("Anxiety", ["Yes", "No"])
peer_pressure = st.selectbox("Peer Pressure", ["Yes", "No"])
chronic_disease = st.selectbox("Chronic Disease", ["Yes", "No"])
fatigue = st.selectbox("Fatigue", ["Yes", "No"])
allergy = st.selectbox("Allergy", ["Yes", "No"])

# Convert Yes/No and Gender to numeric
def to_binary(val): return 1 if val in ["Yes", "Male"] else 0

# Button Row
col1, col2, col3 = st.columns(3)
with col1:
    predict = st.button("🔍 Predict")
with col2:
    clear = st.button("🔄 Clear")
with col3:
    exit = st.button("❌ Exit")

if predict:
    input_data = np.array([
        to_binary(gender),
        age,
        to_binary(smoking),
        to_binary(yellow_fingers),
        to_binary(anxiety),
        to_binary(peer_pressure),
        to_binary(chronic_disease),
        to_binary(fatigue),
        to_binary(allergy)
    ]).reshape(1, -1)

    prediction = model.predict(input_data)[0]
    probability = np.max(model.predict_proba(input_data)) * 100

    if prediction == 0:
        st.success(f"✅ Result: **Negative Lung Cancer** ({probability:.2f}% confidence)")
        st.info("🟢 Recommendation: Maintain a healthy lifestyle. Regular screenings are still important.")
    else:
        st.error(f"⚠️ Result: **Positive Lung Cancer** ({probability:.2f}% confidence)")
        st.warning("📝 Recommendation: Immediate consultation with a specialist is advised.")
        st.markdown("### 🍎 Suggested Foods to Support Lung Health:")
        st.markdown("- Broccoli, spinach, kale (green leafy vegetables)")
        st.markdown("- Garlic and ginger (anti-inflammatory)")
        st.markdown("- Berries (antioxidants)")
        st.markdown("- Green tea (detoxifying)")
        st.markdown("- Omega-3 fish (like salmon, mackerel)")

if clear:
    st.experimental_rerun()

if exit:
    st.stop()
