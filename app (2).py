import streamlit as st
import joblib
import base64
import numpy as np
import pandas as pd

# Load the model and features
model = joblib.load("lung_model.joblib")
features = joblib.load("features.joblib")  # Must be a list of column names

# Set background image
def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        h1, h3, h4, label, p {{
            color: white !important;
        }}
        .stButton > button {{
            color: white;
            background-color: #0077b6;
            border-radius: 10px;
            font-size: 16px;
            margin: 10px 0;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("lung image.jpg")

# Title
st.markdown("<h1 style='text-align: center;'>🫁 Lung Cancer Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Classifying: Normal | Benign | Malignant</h3><hr>", unsafe_allow_html=True)

st.markdown("### 🔍 Enter Patient Details Below")

# Features that should be encoded as Yes/No
yes_no_fields = [
    "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE",
    "CHRONIC DISEASE", "FATIGUE ", "ALLERGY ", "WHEEZING", "ALCOHOL CONSUMING",
    "COUGHING", "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "CHEST PAIN"
]

inputs = []

# Dynamic input rendering
for col in features:
    clean_col = col.strip().upper()
    
    if clean_col == "GENDER":
        val = st.selectbox("Gender", ["Male", "Female"])
        inputs.append(1 if val == "Male" else 0)

    elif clean_col == "AGE":
        val = st.slider("Age", 15, 100, 30)
        inputs.append(val)

    elif clean_col in yes_no_fields:
        val = st.selectbox(col.replace("_", " ").strip().capitalize(), ["No", "Yes"])
        inputs.append(1 if val == "Yes" else 0)

    else:
        val = st.number_input(col.replace("_", " ").capitalize())
        inputs.append(val)

# Buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🩺 Predict"):
        prediction = model.predict([inputs])[0]
        prob = np.max(model.predict_proba([inputs])) * 100
        st.markdown("## 🧬 Prediction Result:")

        if prediction == 0:
            st.success(f"✅ Negative Lung Cancer — ({prob:.2f}% confidence)")
            st.info("🟢 Tip: Maintain a healthy lifestyle. Regular checkups recommended.")

        elif prediction == 1:
            st.warning(f"⚠️ Benign — Non-cancerous ({prob:.2f}% confidence)")
            st.info("🟠 Tip: Monitor symptoms. Follow preventive care.")

        else:
            st.error(f"🚨 Malignant — Cancer detected ({prob:.2f}% confidence)")
            st.warning("🔴 Recommendation: Consult a specialist immediately.")
            st.markdown("**🍎 Suggested Healthy Foods:**")
            st.markdown("- Broccoli, Spinach\n- Berries\n- Garlic and Ginger\n- Green Tea\n- Fish (Omega-3)")

with col2:
    if st.button("🔄 Clear"):
        st.experimental_rerun()

with col3:
    if st.button("❌ Exit"):
        st.markdown("### Thank you for using the Lung Cancer Classifier App.")
        st.stop()

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'>Made with ❤️ by Hira Tariq | 2025</p>", unsafe_allow_html=True)
