import streamlit as st
import joblib
import base64

# Load model and features
model = joblib.load("lung_model.joblib")
features = joblib.load("features.joblib")

# Set background image
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
            background-attachment: fixed;
            color: white;
        }}
        .block-container {{
            background-color: rgba(0, 0, 0, 0.6);
            padding: 2rem;
            border-radius: 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set your lung background
set_background("lung image.jpg")

# Header
st.markdown("<h1 style='text-align: center; color: white;'>🫁 Lung Cancer Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: white;'>Classifies: Normal, Benign, Malignant</h3>", unsafe_allow_html=True)
st.markdown(" ")

# Start input form
st.markdown("### 🔍 Enter Patient Details Below:")

user_input = []
for feature in features:
    if feature.lower() in ['gender', 'smoking', 'anxiety', 'chronic disease', 'fatigue', 'allergy', 'wheezing', 'alcohol', 'coughing']:
        option = st.selectbox(f"{feature.capitalize()}",
                              options=[0, 1],
                              format_func=lambda x: "Yes" if x == 1 else "No")
        user_input.append(option)
    elif feature.lower() == 'age':
        age = st.slider("Age", 1, 100, 30)
        user_input.append(age)
    elif feature.lower() == 'shortness of breath':
        option = st.selectbox("Shortness of Breath", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        user_input.append(option)
    else:
        val = st.number_input(f"{feature.capitalize()}", step=1.0, format="%.2f")
        user_input.append(val)

# Prediction Button
if st.button("🩺 Predict Result"):
    prediction = model.predict([user_input])[0]
    st.markdown("## 🧬 Result:")
    if prediction == 0:
        st.success("✅ Normal - No cancer detected.")
    elif prediction == 1:
        st.warning("⚠️ Benign - Non-cancerous tumor detected.")
    else:
        st.error("🚨 Malignant - Cancer detected. Immediate action recommended!")

# Footer
st.markdown("<hr style='border: 1px solid white;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'>© 2025 Lung Cancer App | Developed by Hira Tariq</p>", unsafe_allow_html=True)
