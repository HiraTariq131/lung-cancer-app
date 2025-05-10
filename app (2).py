import streamlit as st
import joblib
import base64
import numpy as np

# Load model and features
model = joblib.load("lung_model.joblib")
features = joblib.load("features.joblib")

# Set background image
def set_background(image_path):
    with open(image_path, "rb") as img:
        b64_img = base64.b64encode(img.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{b64_img}");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-position: between,center;
            height: 100vh;
        }}
        h1 {{
            color: white !important;
            font-size: 60px !important;
            font-weight: bold !important;
            text-align: center;
        }}
        h2, h3, h4, h5, h6, p, label, .stSelectbox label {{
            color: white !important;
            font-size: 24px !important;
            font-weight: bold !important;
        }}
        .stButton > button {{
            background-color: #0077b6;
            color: white;
            font-size: 24px;
            border-radius: 14px;
            padding: 0.6rem 1.2rem;
            margin: 10px 0;
        }}
        </style>
    """, unsafe_allow_html=True)

# Set background
set_background("lung hand image")

# Title
st.markdown("<h1>😽 Lung Cancer Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>📝 Predict: Positive or Negative</h1><hr>", unsafe_allow_html=True)

# Define Yes/No features
yes_no_features = [
    "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", "CHRONIC DISEASE",
    "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL CONSUMING", "COUGHING",
    "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "CHEST PAIN"
]

# Input fields
inputs = []
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
    else:
        value = st.selectbox(label, ["No", "Yes"])
        inputs.append(1 if value == "Yes" else 0)

# Buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🩺 Predict"):
        prediction = model.predict([inputs])[0]
        probability = np.max(model.predict_proba([inputs])) * 100

        st.markdown("<h2>🧬 Prediction Result:</h2>", unsafe_allow_html=True)

        if prediction == 0:
            st.markdown(f"<p style='color:white; font-size:26px;'><b>✅ Negative Lung Cancer</b> ({probability:.2f}% confidence)</p>", unsafe_allow_html=True)
            st.markdown("<p style='color:white;'>🥰 Stay healthy! No signs of lung cancer detected.</p>", unsafe_allow_html=True)
            st.markdown("<p style='color:white;'><b>🥗 Health Tip:</b> Eat fruits, veggies, stay active, avoid smoking.</p>", unsafe_allow_html=True)
            st.markdown("<p style='color:white;'>• Carrot, Cucumber, Berries</p>", unsafe_allow_html=True)
            st.markdown("<p style='color:white;'>• Apple, Orange, Green Tea</p>", unsafe_allow_html=True)
            st.markdown("<p style='color:white;'>• Omega-3 rich Fish</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='color:#ff4b4b; font-size:26px;'><b>🚨 Positive Lung Cancer</b> ({probability:.2f}% confidence)</p>", unsafe_allow_html=True)
            st.warning("📝 Please consult an oncologist immediately.")
            st.markdown("<p style='color:white;'><b>🍎 Healthy Food Suggestions:</b></p>", unsafe_allow_html=True)
            st.markdown("<p style='color:white;'>• Broccoli, Spinach, Berries</p>", unsafe_allow_html=True)
            st.markdown("<p style='color:white;'>• Garlic, Ginger, Green Tea</p>", unsafe_allow_html=True)
            st.markdown("<p style='color:white;'>• Omega-3 rich Fish</p>", unsafe_allow_html=True)

with col2:
    if st.button("🔄 Clear"):
        st.experimental_rerun()

with col3:
    if st.button("❌ Exit"):
        st.markdown("### Thank you for using the Lung Cancer Predictor App.")
        st.stop()

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; font-size: 18px;'>Made with ❤️ by Hira Tariq | 2025</p>", unsafe_allow_html=True)
