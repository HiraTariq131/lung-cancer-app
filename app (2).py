import streamlit as st
import pickle
import numpy as np

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

st.set_page_config(layout="centered")

# Background image setup using HTML and CSS
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("https://raw.githubusercontent.com/yourusername/yourrepo/main/background.jpg");  /* Replace with your image URL */
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}
h1, h2, h3, h4, h5, h6, .stTextInput > label, .stSelectbox > label {{
    color: white !important;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: white;'>🩺 Lung Cancer Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: white;'>Classifying: Normal | Benign | Malignant</h3>", unsafe_allow_html=True)
st.markdown("<h4 style='color: white;'>🔍 Enter Patient Details Below</h4>", unsafe_allow_html=True)

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 1, 100, 25)
smoking = st.selectbox("Smoking", ["Yes", "No"])
yellow_fingers = st.selectbox("Yellow Fingers", ["Yes", "No"])
anxiety = st.selectbox("Anxiety", ["Yes", "No"])
peer_pressure = st.selectbox("Peer Pressure", ["Yes", "No"])
chronic_disease = st.selectbox("Chronic Disease", ["Yes", "No"])
fatigue = st.selectbox("Fatigue", ["Yes", "No"])
allergy = st.selectbox("Allergy", ["Yes", "No"])

# Button row
col1, col2, col3 = st.columns(3)
with col1:
    predict_button = st.button("🔍 Predict")
with col2:
    clear_button = st.button("🔄 Clear")
with col3:
    exit_button = st.button("❌ Exit")

def convert_to_binary(value):
    return 1 if value == "Yes" or value == "Male" else 0

if predict_button:
    # Convert all inputs to numerical
    data = np.array([
        convert_to_binary(gender),
        age,
        convert_to_binary(smoking),
        convert_to_binary(yellow_fingers),
        convert_to_binary(anxiety),
        convert_to_binary(peer_pressure),
        convert_to_binary(chronic_disease),
        convert_to_binary(fatigue),
        convert_to_binary(allergy)
    ]).reshape(1, -1)

    prediction = model.predict(data)[0]
    probability = np.max(model.predict_proba(data)) * 100

    if prediction == 0:
        st.success(f"✅ Result: **Negative Lung Cancer** ({probability:.2f}% confidence)")
        st.info("🟢 Health Tip: Continue with a healthy lifestyle. Regular checkups recommended.")
    else:
        st.error(f"⚠️ Result: **Positive Lung Cancer** ({probability:.2f}% confidence)")
        st.warning("📝 Recommendation: Consult a specialist immediately.")
        st.markdown("**🍎 Suggested Healthy Foods:**")
        st.markdown("- Broccoli, Spinach")
        st.markdown("- Berries")
        st.markdown("- Garlic and Ginger")
        st.markdown("- Green Tea")
        st.markdown("- Fish (Omega-3 rich)")

if clear_button:
    st.experimental_rerun()

if exit_button:
    st.stop()
