import streamlit as st
import pickle
import numpy as np
import os

# ------------------------------
# Load the saved model
# ------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "heart_disease_model.pkl")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# ------------------------------
# App Title
# ------------------------------
st.set_page_config(page_title="❤️ Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.markdown(
    """
    This app predicts whether a patient is **likely** to have heart disease  
    based on their medical details.  
    Fill in the information below and click **Predict** to see the result.
    """
)

# ------------------------------
# Input fields
# ------------------------------
st.subheader("📋 Patient Information")

age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])
restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)", [0, 1, 2])

# ------------------------------
# Prediction
# ------------------------------
if st.button("🔍 Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])

    prediction = model.predict(input_data)
    result = "Positive — High chance of Heart Disease 💔" if prediction[
                                                                0] == 1 else "Negative — Low chance of Heart Disease ❤️"

    # Show result with styling
    if prediction[0] == 1:
        st.error(result)
    else:
        st.success(result)

# ------------------------------
# Footer
# ------------------------------
st.markdown(
    """
    ---
    **Note:** This tool is for educational purposes only.  
    Consult a medical professional for accurate diagnosis.
    """
)
