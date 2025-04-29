import streamlit as st
import joblib
import base64
import numpy as np

# Load model and features
model = joblib.load("lung_model.joblib")
features = joblib.load("features.joblib")

# Background setup
def set_background(image_path):
    with open(image_path, "rb") as img:
        b64_img = base64.b64encode(img.read()).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{b64_img}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    h1 {{
        font-size: 50px;
        text-align: center;
        color: white !important;
        font-weight: bold;
    }}
    h3, h4, h5, h6, label, p {{
        color: white !important;
        font-size: 20px !important;
        font-weight: bold;
    }}
    .stButton > button {{
        background-color: #0077b6;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        margin-top: 10px;
    }}
    </style>
    """, unsafe_allow_html=True)

# Set background
set_background("lung image.jpg")

# App title
st.markdown("<h1>🫁 Lung Cancer Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Result: Positive or Negative</h4><hr>", unsafe_allow_html=True)

# Input fields
yes_no_fields = [
    "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", "CHRONIC DISEASE",
    "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL CONSUMING", "COUGHING",
    "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "CHEST PAIN"
]

inputs = []

for feature in features:
    if feature == "GENDER":
        gender = st.selectbox("Gender", ["Male", "Female"])
        inputs.append(1 if gender == "Male" else 0)
    elif feature == "AGE":
        age = st.slider("Age", 10, 100, 30)
        inputs.append(age)
    elif feature in yes_no_fields:
        val = st.selectbox(feature.replace("_", " ").title(), ["No", "Yes"])
        inputs.append(1 if val == "Yes" else 0)

# Buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🩺 Predict"):
        prediction = model.predict([inputs])[0]
        proba = np.max(model.predict_proba([inputs])) * 100

        st.markdown("<h3>🧬 Prediction Result:</h3>", unsafe_allow_html=True)
        if prediction == 0:
            st.success(f"✅ **Negative Lung Cancer** ({proba:.2f}% confidence)")
            st.info("🟢 Health Tip: Keep up a healthy lifestyle! No cancer detected.")
        else:
            st.error(f"🚨 **Positive Lung Cancer Detected** ({proba:.2f}% confidence)")
            st.warning("📝 Recommendation: Consult an oncologist immediately.")
            st.markdown("### 🍎 Suggested Healthy Foods:")
            st.markdown("- Broccoli, Spinach, Berries")
            st.markdown("- Garlic, Ginger, Green Tea")
            st.markdown("- Omega-3 rich Fish")

with col2:
    if st.button("🔄 Clear"):
        st.experimental_rerun()

with col3:
    if st.button("❌ Exit"):
        st.markdown("### Thank you for using the Lung Cancer Predictor!")
        st.stop()

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'>Made with ❤️ by Hira Tariq | 2025</p>", unsafe_allow_html=True)
