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
            background-position: center;
        }}
        h1 {{
            color: white !important;
            font-size: 60px !important;
            font-weight: bold !important;
            text-align: center;
        }}
        h2, h3, h4, h5, h6, p, label, .stSelectbox label {{
            color: white !important;
            font-size: 22px !important;
            font-weight: bold !important;
        }}
        .stButton > button {{
            background-color: #0077b6;
            color: white;
            font-size: 20px;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            margin: 10px 0;
        }}
        </style>
    """, unsafe_allow_html=True)

# Set background
set_background("lung image.jpg")

# Title
st.markdown("<h1>🫁 Lung Cancer Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>🔬 Predict: Positive or Negative Only</h3><hr>", unsafe_allow_html=True)

# Field categories
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
    else:
        value = st.selectbox(label, ["No", "Yes"])
        inputs.append(1 if value == "Yes" else 0)

# Button row
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🩺 Predict"):
        prediction = model.predict([inputs])[0]
        probability = np.max(model.predict_proba([inputs])) * 100

        st.markdown("<h2>🧬 Prediction Result:</h2>", unsafe_allow_html=True)
        if prediction == 0:
            st.success(f"✅ **Negative Lung Cancer** ({probability:.2f}% confidence)")
            st.info("🟢 Stay healthy! No signs of lung cancer detected.")
            st.markdown("**🥗 Health Tip:** Eat fruits, veggies, stay active, avoid smoking.")
        else:
            st.error(f"🚨 **Positive Lung Cancer** ({probability:.2f}% confidence)")
            st.warning("📝 See an oncologist immediately.")
            st.markdown("**🍎 Healthy Food Suggestions:**")
            st.markdown("- Broccoli, Spinach, Berries")
            st.markdown("- Garlic, Ginger, Green Tea")
            st.markdown("- Omega-3 rich Fish")

with col2:
    if st.button("🔄 Clear"):
        st.experimental_rerun()

with col3:
    if st.button("❌ Exit"):
        st.markdown("### Thank you for using the Lung Cancer Predictor App.")
        st.stop()

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; font-size: 18px;'>Made with ❤️ by Hira Tariq | 2025</p>", unsafe_allow_html=True)
