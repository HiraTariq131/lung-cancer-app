import streamlit as st
import joblib
import base64
import numpy as np

# Load model and features
model = joblib.load("lung_model.joblib")
features = joblib.load("features.joblib")

# Set background image
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        h1, h2, h3, h4, h5, h6, p, label {{
            color: white !important;
            font-weight: bold;
        }}
        .stButton > button {{
            color: white;
            background-color: #0288d1;
            border-radius: 12px;
            padding: 10px 20px;
            font-size: 16px;
        }}
        </style>
    """, unsafe_allow_html=True)

# Set your image file
set_background("lung image.jpg")

# Title
st.markdown("<h1 style='text-align: center;'>🫁 Lung Cancer Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Predict: Normal | Benign | Malignant</h4><hr>", unsafe_allow_html=True)
st.markdown("### 🔍 Enter Patient Details")

inputs = []

# Features to handle as Yes/No
yes_no_features = [
    "SMOKING", "ANXIETY", "CHRONIC DISEASE", "FATIGUE", "ALLERGY",
    "WHEEZING", "ALCOHOL", "COUGHING", "SHORTNESS OF BREATH",
    "YELLOW FINGERS", "PEER PRESSURE"
]

# Create input fields
for feature in features:
    if feature.upper() == "GENDER":
        gender = st.selectbox("Gender", ["Male", "Female"])
        inputs.append(1 if gender == "Male" else 0)
    elif feature.upper() == "AGE":
        age = st.slider("Age", 1, 100, 30)
        inputs.append(age)
    elif feature.upper() in yes_no_features:
        val = st.selectbox(feature.replace("_", " ").title(), ["No", "Yes"])
        inputs.append(1 if val == "Yes" else 0)
    else:
        val = st.number_input(feature.title(), format="%.2f")
        inputs.append(val)

# Buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🩺 Predict"):
        prediction = model.predict([inputs])[0]
        prob = np.max(model.predict_proba([inputs])) * 100

        st.markdown("## 🧬 Prediction Result:")
        if prediction == 0:
            st.success(f"✅ Negative Lung Cancer ({prob:.2f}% confidence)")
            st.info("✔️ Stay healthy! Keep up regular exercise and checkups.")
        elif prediction == 1:
            st.warning(f"⚠️ Benign Condition ({prob:.2f}% confidence)")
            st.info("🔍 Keep monitoring your symptoms regularly.")
        else:
            st.error(f"🚨 Positive Lung Cancer Detected ({prob:.2f}% confidence)")
            st.warning("🏥 Please consult a specialist immediately!")
            st.markdown("### 🍎 Recommended Healthy Foods:")
            st.markdown("- Leafy greens (Spinach, Kale)")
            st.markdown("- Berries (Blueberries, Strawberries)")
            st.markdown("- Garlic & Ginger")
            st.markdown("- Green Tea")
            st.markdown("- Omega-3 rich fish")

with col2:
    if st.button("🔄 Clear"):
        st.experimental_rerun()

with col3:
    if st.button("❌ Exit"):
        st.markdown("### Session ended. Thank you!")
        st.stop()

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'>Made with ❤️ by Hira Tariq | 2025</p>", unsafe_allow_html=True)
