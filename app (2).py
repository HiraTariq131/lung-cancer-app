import streamlit as st
import joblib
import base64

# Load your model
model = joblib.load("lung_model.joblib")
features = joblib.load("features.joblib")

# Set background using your image
def set_background(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

# Set your uploaded image as background
set_background("lung image.jpg")

# Main app title
st.markdown("<h1 style='text-align: center; color: white;'>Lung Cancer Detection App</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: white;'>Normal vs Benign vs Malignant</h4>", unsafe_allow_html=True)
st.markdown("---")

# Input fields
st.markdown("### 🔍 Enter Patient Details:")
user_input = []
for feature in features:
    val = st.number_input(f"{feature}", step=1.0, format="%.2f")
    user_input.append(val)

# Predict button
if st.button("Predict 🩺"):
    pred = model.predict([user_input])[0]
    st.markdown("### 🧬 Prediction Result:")
    if pred == 0:
        st.success("✅ Normal")
    elif pred == 1:
        st.warning("⚠️ Benign Tumor Detected")
    else:
        st.error("🚨 Malignant Tumor Detected")

# Add footer or any extra text
st.markdown("<hr style='border: 1px solid white;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'>© 2025 Lung Cancer Classifier | By Hira</p>", unsafe_allow_html=True)
