import streamlit as st
import joblib
import base64
import sys

# Load model and features
model = joblib.load("lung_model.joblib")
features = joblib.load("features.joblib")

# Set background image
def set_background(image_path):
    with open(image_path, "rb") as img_file:
        img_bytes = img_file.read()
        encoded = base64.b64encode(img_bytes).decode()
    st.markdown(f"""
        <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded}");
                background-size: cover;
                background-attachment: fixed;
            }}
            .block-container {{
                background-color: rgba(0, 0, 0, 0.75);
                padding: 2rem;
                border-radius: 15px;
                color: white;
            }}
            h1, h2, h3, p, label {{
                color: white !important;
            }}
            .stButton > button {{
                color: white;
                background-color: #0d6efd;
                border-radius: 10px;
                padding: 0.5rem 1rem;
                font-size: 16px;
            }}
        </style>
    """, unsafe_allow_html=True)

# Apply background
set_background("lung image.jpg")

# Title
st.markdown("<h1 style='text-align: center;'>🫁 Lung Cancer Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Classifying: Normal | Benign | Malignant</h3><hr>", unsafe_allow_html=True)

# Input fields
st.markdown("### 🔍 Enter Patient Details Below")

inputs = []

for feature in features:
    f_lower = feature.lower()
    
    if f_lower in ['gender']:
        gender = st.selectbox("Gender", options=["Male", "Female"])
        inputs.append(1 if gender == "Male" else 0)

    elif f_lower == 'age':
        age = st.slider("Age", 1, 100, 30)
        inputs.append(age)

    elif f_lower in ['smoking', 'anxiety', 'chronic disease', 'fatigue', 'allergy', 'wheezing', 'alcohol', 'coughing', 'shortness of breath']:
        val = st.selectbox(f"{feature.capitalize()}", options=["No", "Yes"])
        inputs.append(1 if val == "Yes" else 0)

    else:
        val = st.number_input(f"{feature.capitalize()}", format="%.2f")
        inputs.append(val)

# Buttons layout
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🩺 Predict"):
        prediction = model.predict([inputs])[0]
        st.markdown("## 🧬 Prediction Result:")
        if prediction == 0:
            st.success("✅ **Normal** — No signs of cancer detected.")
        elif prediction == 1:
            st.warning("⚠️ **Benign** — Non-cancerous condition detected.")
        else:
            st.error("🚨 **Malignant** — Cancer detected. Immediate medical action advised!")

with col2:
    if st.button("🔄 Clear"):
        st.experimental_rerun()

with col3:
    if st.button("❌ Exit"):
        st.markdown("### Thank you for using the Lung Cancer App.")
        st.stop()

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'>Made with ❤️ by Hira Tariq | 2025</p>", unsafe_allow_html=True)
