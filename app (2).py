import streamlit as st
import joblib
import base64
import numpy as np

# Load model and features
model = joblib.load("lung_model.joblib")
features = joblib.load("features.joblib")

# Set background image
def set_background(image_file):
    with open(image_file, "rb") as image:
        base64_image = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{base64_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center center;
            background-attachment: fixed;
            height: 100vh;
            width: 100vw;
            margin: 0;
            padding: 0;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background
set_background("blue lung image.jpg")


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
