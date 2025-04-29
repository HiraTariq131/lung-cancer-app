import streamlit as st
import joblib
import base64
import numpy as np

# Load model and features
model = joblib.load("lung_model.joblib")
features = joblib.load("features.joblib")

# Set custom background image
def set_background(image_path):
    with open(image_path, "rb") as img:
        b64_img = base64.b64encode(img.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{b64_img}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        html, body, [class*="css"] {{
            color: white !important;
            font-size: 22px !important;
            font-weight: bold !important;
        }}
        .stButton > button {{
            background-color: #0077b6;
            color: white;
            font-size: 20px;
            border-radius: 10px;
            padding: 0.6rem 1.5rem;
            margin-top: 10px;
        }}
        h1, h2, h3, h4, h5 {{
            color: white !important;
            text-align: center;
            font-weight: bold !important;
        }}
        </style>
    """, unsafe_allow_html=True)

# Set background image
set_background("lung image.jpg")

# Headline
st.markdown("<h1>🫁 Lung Cancer Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3>Result: Positive or Negative Only</h3><hr>", unsafe_allow_html=True)

# Yes/No input features
yes_no_features = [
    "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", "CHRONIC DISEASE",
    "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL CONSUMING", "COUGHING",
    "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "CHEST PAIN"
]

inputs = []

# Generate input fields
for feature in features:
    label = feature.replace("_", " ").title()
    if feature == "GENDER":
        gender = st.selectbox("Gender", ["Male", "Female"])
        inputs.append(1 if gender == "Male" else 0)
    elif feature == "AGE":
        age = st.slider("Age", 10, 100, 30)
        inputs.append(age)
    elif feature in yes_no_features:
        value = st.selectbox(label, ["No", "Yes"])
        inputs.append(1 if value == "Yes" else 0)

# Buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🩺 Predict"):
        prediction = model.predict([inputs])[0]
        proba = np.max(model.predict_proba([inputs])) * 100

        st.markdown("## 🧬 Prediction Result:")
        if prediction == 0:
            st.success(f"✅ Negative Lung Cancer ({proba:.2f}% confidence)")
            st.markdown("### 🟢 Health Tip:")
            st.markdown("- Keep a balanced diet 🥦")
            st.markdown("- Exercise regularly 🏃")
            st.markdown("- Avoid smoking and pollution 🚭")
        else:
            st.error(f"🚨 Positive Lung Cancer Detected ({proba:.2f}% confidence)")
            st.markdown("### 🔴 Recommendation:")
            st.markdown("- Consult an oncologist immediately 🩺")
            st.markdown("### 🍽️ Suggested Healthy Foods:")
            st.markdown("- Broccoli, Spinach, Garlic")
            st.markdown("- Berries, Green Tea")
            st.markdown("- Omega-3 Rich Fish")

with col2:
    if st.button("🔄 Clear"):
        st.experimental_rerun()

with col3:
    if st.button("❌ Exit"):
        st.markdown("### Thank you for using the Lung Cancer Classifier.")
        st.stop()

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Made with ❤️ by Hira Tariq | 2025</p>", unsafe_allow_html=True)
