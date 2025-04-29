import streamlit as st
import joblib
import base64
import numpy as np

# Load trained model and features
model = joblib.load("lung_model.joblib")
features = [
    'GENDER', 'AGE', 'SMOKING', 'YELLOW FINGERS', 'ANXIETY', 'PEER PRESSURE',
    'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL CONSUMING',
    'COUGHING', 'SHORTNESS BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN'
]

# Set background image
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        b64_img = base64.b64encode(image_file.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{b64_img}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: white;
        }}
        h1, h2, h3, h4, label, p, .stSelectbox label, .stSlider label {{
            color: white !important;
            font-size: 18px !important;
            font-weight: bold !important;
        }}
        .stButton > button {{
            background-color: #0077b6;
            color: white;
            font-size: 18px;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 24px;
        }}
        </style>
    """, unsafe_allow_html=True)

# Set background
set_background("lung image.jpg")

# Title
st.markdown("<h1 style='text-align: center;'>🫁 Lung Cancer Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>🔎 Predicting: Positive or Negative</h4><hr>", unsafe_allow_html=True)

# Yes/No fields
yes_no_features = [
    "SMOKING", "YELLOW FINGERS", "ANXIETY", "PEER PRESSURE", "CHRONIC DISEASE",
    "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL CONSUMING", "COUGHING",
    "SHORTNESS BREATH", "SWALLOWING DIFFICULTY", "CHEST PAIN"
]

inputs = []

# Form inputs
for feature in features:
    name = feature.strip().replace("_", " ").title()
    if feature == "GENDER":
        val = st.selectbox("Gender", ["Male", "Female"])
        inputs.append(1 if val == "Male" else 0)
    elif feature == "AGE":
        val = st.slider("Age", 10, 100, 30)
        inputs.append(val)
    elif feature in yes_no_features:
        val = st.selectbox(name, ["No", "Yes"])
        inputs.append(1 if val == "Yes" else 0)

# Buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🩺 Predict"):
        pred = model.predict([inputs])[0]
        proba = np.max(model.predict_proba([inputs])) * 100

        st.markdown("## 🧬 Prediction Result:")
        if pred == 1:
            st.error(f"🚨 **Positive Lung Cancer** ({proba:.2f}% confidence)")
            st.warning("📝 Recommendation: Consult a lung cancer specialist immediately.")
            st.markdown("**🍏 Healthy Food Recommendations:**")
            st.markdown("- Broccoli, Spinach, Kale")
            st.markdown("- Blueberries, Raspberries")
            st.markdown("- Garlic, Ginger, Turmeric")
            st.markdown("- Green Tea, Lemon Water")
            st.markdown("- Omega-3 rich Fish (Salmon, Tuna)")
        else:
            st.success(f"✅ **Negative Lung Cancer** ({proba:.2f}% confidence)")
            st.info("🟢 Health Tip: Keep a balanced diet and stay physically active.")

with col2:
    if st.button("🔄 Clear"):
        st.experimental_rerun()

with col3:
    if st.button("❌ Exit"):
        st.markdown("### Thank you for using the Lung Cancer Predictor.")
        st.stop()

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; font-weight: bold;'>Made with ❤️ by Hira Tariq | 2025</p>", unsafe_allow_html=True)
