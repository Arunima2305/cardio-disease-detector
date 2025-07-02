import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load('svm_model.pkl')

st.title("💓 Heart Disease Prediction App")

st.markdown("Enter the patient’s health data below to get a prediction.")

# Layout inputs in two columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    sex = st.selectbox("Sex", ["Male", "Female"])
    resting_bp = st.number_input("Resting Blood Pressure", min_value=50, max_value=250)
    cholesterol = st.number_input("Cholesterol", min_value=50, max_value=600)
    max_hr = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220)

with col2:
    chest_pain = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=6.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses an SVM model to predict the likelihood of heart disease 
    based on patient health indicators.
    """)

    st.write("👨‍⚕️ Developed for academic & educational purposes.")

input_dict = {
    'Age': [int(pd.cut([age], bins=[0, 35, 55, float('inf')], labels=[0, 1, 2])[0])],
    'RestingBP': [float(pd.cut([resting_bp], bins=[-float('inf'), 90, 120, 130, float('inf')], labels=[1, 0, 2, 3])[0])],
    'Cholesterol': [float(pd.cut([cholesterol], bins=[-float('inf'), 150, 200, float('inf')], labels=[1, 0, 2])[0])],
    'MaxHR': [float(pd.cut([max_hr], bins=[0, 90, 130, 160, float('inf')], labels=[1, 0, 2, 3])[0])],
    'Oldpeak': [float(pd.cut([oldpeak], bins=[-float('inf'), 0, 1, 2, float('inf')], labels=[0, 1, 2, 3])[0])],
    'FastingBS': [fasting_bs],
    'ST_Slope': [ {'Up': 0, 'Flat': 1, 'Down': 2}[st_slope] ],
    'Sex_M': [1 if sex == "Male" else 0],
    'ExerciseAngina_Y': [1 if exercise_angina == "Yes" else 0],
    'ChestPainType_ASY': [1 if chest_pain == "ASY" else 0],
    'ChestPainType_ATA': [1 if chest_pain == "ATA" else 0],
    'ChestPainType_NAP': [1 if chest_pain == "NAP" else 0],
    'ChestPainType_TA': [1 if chest_pain == "TA" else 0],
    'RestingECG_LVH': [1 if resting_ecg == "LVH" else 0],
    'RestingECG_Normal': [1 if resting_ecg == "Normal" else 0],
    'RestingECG_ST': [1 if resting_ecg == "ST" else 0],
}



input_df = pd.DataFrame(input_dict)

# ✅ Fix column order before prediction
columns_order = joblib.load('feature_columns.pkl')  # Load this file saved during training
input_df = input_df.reindex(columns=columns_order, fill_value=0)  # Ensure exact order

if st.button("🔍 Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("## 🧠 Prediction Result")
    if prediction == 0:
        st.success("✅ The patient is **not likely** to have heart disease.")
    else:
        st.error("⚠️ The patient is **likely** to have heart disease.")

    st.markdown(f"### 📊 Confidence Score: `{probability * 100:.2f}%`")


st.markdown("---")
st.markdown("<center>💖 Built with Streamlit | Version 1.0</center>", unsafe_allow_html=True)


